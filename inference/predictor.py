import torch
import numpy as np
import pandas as pd
import os
import json
import torch.serialization
import torch.nn as nn
import models.hybrid_model
import models.transformer_encoder
from models.hybrid_model import build_model
from scipy.stats import zscore

from utils.config import config
from utils.logger import logger
from data import database
from data.preprocessor import get_intermediate_data, create_final_sequences_and_scale, get_market_index
from utils.metrics import calculate_ece
from utils.model_artifacts import build_model_artifact_audit, list_numbered_model_paths
from sklearn.calibration import calibration_curve

from tslearn.metrics import soft_dtw
from utils.model_tracker import get_tracker

# --- Global Model Cache for Performance ---
_MODEL_CACHE = None
_MODEL_CACHE_PATHS = None
_ENSEMBLE_CONFIGS = None


def _load_ensemble_configs() -> list[dict]:
    """Load per-model configs from ensemble_configs.json (cached globally)."""
    global _ENSEMBLE_CONFIGS
    if _ENSEMBLE_CONFIGS is not None:
        return _ENSEMBLE_CONFIGS
    cfg_path = os.path.join("models", "ensemble_configs.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            _ENSEMBLE_CONFIGS = json.load(f).get("models", [])
    else:
        _ENSEMBLE_CONFIGS = []
    return _ENSEMBLE_CONFIGS


def _align_sequence_to_model_input(sequence_tensor: torch.Tensor, model, warn_once_keys: set | None = None) -> torch.Tensor:
    """Best-effort backward compatibility for old checkpoints with smaller input dims."""
    expected_dim = int(getattr(model, "input_dim", sequence_tensor.shape[-1]) or sequence_tensor.shape[-1])
    actual_dim = int(sequence_tensor.shape[-1])
    if expected_dim == actual_dim:
        return sequence_tensor

    key = (actual_dim, expected_dim)

    def _warn_once(msg: str):
        if warn_once_keys is None:
            logger.warning(msg)
            return
        if key not in warn_once_keys:
            warn_once_keys.add(key)
            logger.warning(msg)

    if expected_dim < actual_dim:
        _warn_once(
            f"[PredictorTrace] truncating input features from {actual_dim} -> {expected_dim} "
            f"for legacy checkpoint compatibility"
        )
        return sequence_tensor[..., :expected_dim]

    _warn_once(
        f"[PredictorTrace] padding input features from {actual_dim} -> {expected_dim} "
        f"for legacy checkpoint compatibility"
    )
    pad = torch.zeros(
        sequence_tensor.shape[0],
        sequence_tensor.shape[1],
        expected_dim - actual_dim,
        device=sequence_tensor.device,
        dtype=sequence_tensor.dtype,
    )
    return torch.cat([sequence_tensor, pad], dim=-1)


def _resolve_model_paths() -> list[str]:
    model_paths = sorted(list_numbered_model_paths())
    audit = build_model_artifact_audit()
    compatible_paths = [item["path"] for item in audit.get("models", []) if item.get("runtime_compatible")]
    if compatible_paths:
        if len(compatible_paths) != len(model_paths):
            logger.info(
                f"Selecting {len(compatible_paths)} runtime-compatible models out of {len(model_paths)} artifacts."
            )
        model_paths = compatible_paths
    elif model_paths:
        incompatible = [
            f"{item.get('model_key')}: {', '.join(item.get('issues', []))}"
            for item in audit.get("models", [])
            if item.get("issues")
        ]
        if incompatible:
            logger.warning(
                "[PredictorCompat] No runtime-compatible model artifacts found; falling back to legacy checkpoints: "
                + " | ".join(incompatible)
            )

    # Filter models marked as excluded in ensemble_configs.json
    import re as _re
    ensemble_cfgs = _load_ensemble_configs()
    if ensemble_cfgs:
        excluded_ids = {cfg["id"] for cfg in ensemble_cfgs if cfg.get("excluded", False)}
        if excluded_ids:
            before = len(model_paths)
            filtered = []
            for path in model_paths:
                m = _re.search(r'model_(\d+)\.pth', path)
                if m and int(m.group(1)) in excluded_ids:
                    reason = next(
                        (c.get("excluded_reason", "excluded in ensemble_configs.json")
                         for c in ensemble_cfgs if c["id"] == int(m.group(1))),
                        "excluded in ensemble_configs.json"
                    )
                    logger.info(f"[EnsembleFilter] Skipping model_{m.group(1)}: {reason}")
                    continue
                filtered.append(path)
            if len(filtered) != before:
                logger.info(f"[EnsembleFilter] {before} → {len(filtered)} models after exclusion filter.")
            model_paths = filtered

    ensemble_limit = getattr(config.Gan, "N_ENSEMBLE_MODELS", None)
    try:
        ensemble_limit = int(ensemble_limit) if ensemble_limit is not None else None
    except Exception:
        ensemble_limit = None

    if ensemble_limit is not None and ensemble_limit > 0 and len(model_paths) > ensemble_limit:
        logger.info(
            f"Limiting inference ensemble from {len(model_paths)} to {ensemble_limit} "
            f"based on config.Gan.N_ENSEMBLE_MODELS."
        )
        model_paths = model_paths[:ensemble_limit]

    return model_paths


def _detect_legacy_checkpoint_issues(model) -> list[str]:
    issues = []

    expected_input_dim = len(getattr(config.Data, "FEATURE_COLUMNS", []) or [])
    actual_input_dim = getattr(model, "input_dim", None)
    if expected_input_dim and actual_input_dim not in (None, expected_input_dim):
        issues.append(f"input_dim={actual_input_dim} (expected {expected_input_dim})")

    decoder = getattr(model, "decoder", None)
    decoder_class = decoder.__class__.__name__ if decoder is not None else None
    expected_decoder_mode = str(getattr(config.Gan, "DECODER_MODE", "") or "").lower()
    actual_decoder_mode = str(getattr(model, "decoder_mode", "") or "").lower()
    if expected_decoder_mode == "cvae" and decoder_class != "CVAEDecoder":
        issues.append(f"decoder={decoder_class or 'None'} (expected CVAEDecoder)")
    elif expected_decoder_mode == "gan" and decoder_class != "GANDecoder":
        issues.append(f"decoder={decoder_class or 'None'} (expected GANDecoder)")

    if not actual_decoder_mode:
        issues.append("decoder_mode metadata missing")

    if not getattr(model, "pe_mode", None):
        issues.append("pe_mode metadata missing")

    return issues


def _get_cached_models():
    """Load models once and cache them globally for faster predictions."""
    global _MODEL_CACHE, _MODEL_CACHE_PATHS
    
    model_paths = _resolve_model_paths()
    
    # Check if cache is valid (same model files)
    if _MODEL_CACHE is not None and _MODEL_CACHE_PATHS == model_paths:
        return _MODEL_CACHE
    
    # Load models and cache them
    logger.info(f"Loading {len(model_paths)} ensemble models into cache...")
    models = []
    for path in model_paths:
        try:
            model = torch.load(path, map_location=config.Device.DEVICE, weights_only=False)
            model = model.to(config.Device.DEVICE)
            model.eval()
            legacy_issues = _detect_legacy_checkpoint_issues(model)
            model_input_dim = getattr(model, "input_dim", None)
            if model_input_dim is not None and model_input_dim != len(config.Data.FEATURE_COLUMNS):
                logger.warning(
                    f"[Predictor] Model input_dim={model_input_dim} != config features={len(config.Data.FEATURE_COLUMNS)}. "
                    f"Old checkpoint — retrain needed."
                )
            if legacy_issues:
                logger.warning(
                    f"[PredictorCompat] Legacy/incompatible checkpoint detected for {path}: "
                    + "; ".join(legacy_issues)
                )
            models.append(model)
            logger.info(f"Cached model from {path}")
        except Exception as e:
            logger.error(f"Failed to load model from {path}: {e}")
            continue
    
    _MODEL_CACHE = models
    _MODEL_CACHE_PATHS = model_paths
    logger.info(f"Model cache initialized with {len(models)} models.")
    return models

def get_pattern_similarity(pattern1: np.ndarray, pattern2: np.ndarray) -> float:
    """
    Calculates the similarity between two price change patterns using Soft Dynamic Time Warping (soft-DTW)
    with z-score normalization.
    """
    if len(pattern1) == 0 or len(pattern2) == 0:
        return float('inf')

    # Z-score normalization to compare shapes
    pattern1_norm = zscore(pattern1)
    pattern2_norm = zscore(pattern2)

    # Reshape for tslearn: (sz, d)
    p1_reshaped = pattern1_norm.reshape(-1, 1)
    p2_reshaped = pattern2_norm.reshape(-1, 1)

    # Use gamma=0.1 as a starting point
    distance = soft_dtw(p1_reshaped, p2_reshaped, gamma=0.1)
    
    return distance



def run(markets: list, market_index_df: pd.DataFrame = None, historical_df: pd.DataFrame = None):
    """
    Makes ensembled, probabilistic predictions for a given list of markets.
    Can accept pre-loaded dataframes for efficient backtesting.
    """
    logger.info(f"--- Making ensembled predictions for {len(markets)} markets ---")

    # Use cached models instead of loading every time
    models = _get_cached_models()
    
    if not models:
        logger.error("No models available. Please run training first.")
        return []

    logger.info(f"Using {len(models)} cached ensemble models.")

    if market_index_df is None:
        logger.info("Market index not provided, calculating fresh...")
        market_index_df = get_market_index()
    
    # --- Refactoring for Shrunk Beta ---
    from data.preprocessor import get_crypto_factors
    crypto_factors_df, rank_4h_df = get_crypto_factors()

    intermediate_data = {}
    for market in markets:
        df, scaler = get_intermediate_data(market, market_index_df, historical_df=historical_df, crypto_factors_df=crypto_factors_df, rank_4h_df=rank_4h_df)
        if df is not None:
            intermediate_data[market] = {'df': df, 'scaler': scaler}

    if not intermediate_data:
        logger.warning("No markets had sufficient data for pre-computation.")
        return []

    beta_series_list = [data['df']['beta'].rename(market) for market, data in intermediate_data.items()]
    all_betas_df = pd.concat(beta_series_list, axis=1)
    cs_beta_mean = all_betas_df.mean(axis=1)
    beta_shrinkage_weight = float(getattr(config.Data, "BETA_SHRINKAGE_WEIGHT", 0.0) or 0.0)
    if beta_shrinkage_weight > 0:
        logger.info(f"Calculated cross-sectional mean beta for shrinkage (weight={beta_shrinkage_weight:.2f}).")

    all_predictions = []
    compat_warned_shapes = set()
    for market, data in intermediate_data.items():
        logger.info(f"[PredictorTrace] market={market} stage=start")
        df = data['df']
        scaler = data['scaler']

        if beta_shrinkage_weight > 0:
            own_weight = max(0.0, 1.0 - beta_shrinkage_weight)
            df['beta'] = own_weight * df['beta'] + beta_shrinkage_weight * cs_beta_mean
            df.loc[:, 'beta'] = df['beta'].fillna(0)

        X, y, scaler = create_final_sequences_and_scale(df, scaler)
        
        if X is None:
            logger.info(f"[PredictorTrace] market={market} stage=skip_no_sequences")
            continue
        logger.info(
            f"[PredictorTrace] market={market} stage=sequence_ready seq_shape={tuple(X.shape)}"
        )
        
        last_sequence = X[-1]
        sequence_tensor = torch.as_tensor(last_sequence, dtype=torch.float32, device=config.Device.DEVICE).unsqueeze(0)

        with torch.no_grad():
            individual_patterns = []
            gate_values = []
            model_directions = []
            attention_importances = []  # (n_mc, seq)
            model_confidences = []  # confidence_head outputs per model

            model_avg_patterns = []  # Store average pattern per model for weighted voting
            
            ensemble_cfgs = _load_ensemble_configs()
            for model_idx, model in enumerate(models):
                # Per-model lookback slicing: trim sequence to model's lookback_hours
                if model_idx < len(ensemble_cfgs):
                    lookback_h = ensemble_cfgs[model_idx].get("lookback_hours")
                else:
                    lookback_h = None
                if lookback_h is not None and lookback_h < sequence_tensor.shape[1]:
                    model_seq = sequence_tensor[:, -lookback_h:, :]
                else:
                    model_seq = sequence_tensor
                model_input = _align_sequence_to_model_input(model_seq, model, warn_once_keys=compat_warned_shapes)
                decoder_mode = getattr(model, "decoder_mode", "gan")
                if decoder_mode == "cvae":
                    model.eval()
                    out = model(model_input, return_explainability=True)
                    if isinstance(out, tuple):
                        _, explainability_info = out
                    else:
                        explainability_info = {}

                    gate_info = explainability_info.get("gate_info", {}) if isinstance(explainability_info, dict) else {}
                    gate_val = float(np.mean(gate_info.get("gate_values", [0.5]))) if isinstance(gate_info, dict) else 0.5
                    gate_values.append(gate_val)

                    attn_imp = explainability_info.get("attention_importance") if isinstance(explainability_info, dict) else None
                    if isinstance(attn_imp, torch.Tensor):
                        attn_imp = attn_imp.detach().cpu().numpy()
                    if attn_imp is not None:
                        attention_importances.append(np.array(attn_imp).flatten())

                    fused_context = explainability_info.get("fused_context") if isinstance(explainability_info, dict) else None
                    if fused_context is None:
                        fused_context = model(model_input, return_explainability=True)[1]["fused_context"]
                    # Extract confidence from confidence_head if available
                    if hasattr(model.decoder, 'confidence_head') and fused_context is not None:
                        conf_val = model.decoder.confidence_head(fused_context).squeeze().detach().cpu().item()
                        model_confidences.append(conf_val)

                    samples = model.decoder.sample(
                        fused_context,
                        n_samples=config.Recommender.MC_N_INFERENCES,
                    ).squeeze(0).detach().cpu().numpy()
                    model_preds_local = samples.tolist()
                    individual_patterns.extend(model_preds_local)
                else:
                    # MC-Dropout: Dropout stays active, BatchNorm in eval
                    model.train()
                    for module in model.modules():
                        if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                            module.eval()

                    model_preds_local = []
                    for _ in range(config.Recommender.MC_N_INFERENCES):
                        # Request explainability to get gate value (Trend vs Pattern)
                        out = model(model_input, return_explainability=True)

                        if isinstance(out, tuple):
                            pred_tensor = out[0]
                            explainability_info = out[1]
                            if isinstance(explainability_info, dict) and 'gate_info' in explainability_info:
                                gate_info = explainability_info['gate_info']
                                if isinstance(gate_info, torch.Tensor):
                                    gate_val = gate_info.detach().cpu().item()
                                elif isinstance(gate_info, dict) and 'gate_values' in gate_info:
                                    gate_val = float(np.mean(gate_info['gate_values']))
                                else:
                                    gate_val = 0.5
                            else:
                                gate_val = 0.5

                            if isinstance(explainability_info, dict) and 'attention_importance' in explainability_info:
                                attn_imp = explainability_info['attention_importance']
                                if isinstance(attn_imp, torch.Tensor):
                                    attn_imp = attn_imp.detach().cpu().numpy()
                                attention_importances.append(np.array(attn_imp).flatten())
                        else:
                            pred_tensor = out
                            gate_val = 0.5

                        predicted_pattern = pred_tensor.detach().cpu().numpy().flatten()
                        individual_patterns.append(predicted_pattern)
                        gate_values.append(gate_val)
                        model_preds_local.append(predicted_pattern)

                # Calculate direction for this specific model (for Consensus)
                avg_model_pattern = np.array(np.mean(model_preds_local, axis=0)).flatten()
                model_avg_patterns.append(avg_model_pattern)  # For weighted voting
                # Direction: End - Start
                direction = 1 if avg_model_pattern[-1] > avg_model_pattern[0] else -1
                model_directions.append(direction)
        logger.info(
            f"[PredictorTrace] market={market} stage=model_ensemble_complete "
            f"samples_n={len(individual_patterns)} models_n={len(models)}"
        )
        
        individual_patterns = np.array([np.array(p).flatten() for p in individual_patterns])

        # --- Weighted Ensemble Voting ---
        tracker = get_tracker(n_models=len(models))
        model_weights = tracker.get_weights()
        model_avg_patterns = np.array([np.array(p).flatten() for p in model_avg_patterns])
        
        # Weighted average of model predictions (instead of simple mean)
        final_pattern = np.average(model_avg_patterns, axis=0, weights=model_weights)
        
        # Calculate Consensus Score (Agreement % among models)
        n_up = model_directions.count(1)
        n_down = model_directions.count(-1)
        consensus_score = max(n_up, n_down) / len(models) # 1.0 = Unanimous, 0.6 = Split (3vs2)

        # Calculate Mean Gate Value
        final_gate_value = np.mean(gate_values)

        # Predictive standard deviation across all MC/ensemble samples
        # Simple, interpretable, no near-zero denominator issues
        pattern_std = np.std(individual_patterns, axis=0)
        final_uncertainty = float(np.mean(pattern_std))

        current_price = df.iloc[-1]['close']

        # PI_80: 80% prediction interval from MC-Dropout samples
        pi_low_80 = float(np.percentile(individual_patterns.sum(axis=1), 10))
        pi_high_80 = float(np.percentile(individual_patterns.sum(axis=1), 90))

        # attention_top3: mean attention importance across MC runs → top-3 timestep indices
        if attention_importances:
            try:
                # Filter to same-length arrays (different models may have different attention shapes)
                min_len = min(len(np.array(a).flatten()) for a in attention_importances)
                attn_uniform = [np.array(a).flatten()[:min_len] for a in attention_importances]
                mean_attn = np.mean(np.array(attn_uniform), axis=0)
                top3_idx = np.argsort(mean_attn)[::-1][:3].tolist()
            except Exception:
                top3_idx = []
        else:
            top3_idx = []

        # prototype_match: closest success pattern (DTW distance reused from recommender)
        from data.preprocessor import get_historical_success_patterns
        success_patterns = get_historical_success_patterns()
        if success_patterns.any():
            dists = [get_pattern_similarity(final_pattern, sp) for sp in success_patterns]
            best_proto_idx = int(np.argmin(dists))
            best_proto_sim = float(1.0 / (1.0 + np.min(dists)))  # similarity ∈ (0,1]
        else:
            best_proto_idx, best_proto_sim = -1, 0.0
        logger.info(
            f"[PredictorTrace] market={market} stage=prototype_complete "
            f"proto_id={best_proto_idx} proto_sim={best_proto_sim:.4f}"
        )

        # Model confidence: average of confidence_head outputs across ensemble
        model_confidence = float(np.mean(model_confidences)) if model_confidences else None

        all_predictions.append({
            "market": market,
            "predicted_pattern": final_pattern,
            "uncertainty": final_uncertainty,
            "model_confidence": model_confidence,  # from confidence_head (calibrated probability)
            "gate_value": final_gate_value,  # New: Trend(>0.6) vs Pattern(<0.4)
            "consensus_score": consensus_score, # New: Model Agreement
            "current_price": current_price,
            "individual_patterns": individual_patterns,
            "n_ensemble_models": len(models),
            "n_mc_inferences": config.Recommender.MC_N_INFERENCES,
            "pi_low_80": pi_low_80,
            "pi_high_80": pi_high_80,
            "attention_top3": top3_idx,
            "prototype_match": {"id": best_proto_idx, "similarity": best_proto_sim},
        })
        
        regime = "Trend" if final_gate_value > 0.6 else ("Pattern" if final_gate_value < 0.4 else "Hybrid")
        logger.info(f"Ensemble prediction for {market}: Unc={final_uncertainty:.2f}, Gate={final_gate_value:.2f}({regime}), Consensus={consensus_score:.2f}")

    logger.info(f"[PredictorTrace] predictor.run returning predictions_n={len(all_predictions)}")
    return all_predictions


def run_with_explainability(market: str, market_index_df=None, historical_df=None):
    """
    Makes a prediction with full explainability analysis.
    
    Returns:
        result: Dictionary with prediction and all explainability info
    """
    from analysis.explainability_analyzer import analyze_prediction
    
    models = _get_cached_models()
    if not models:
        logger.error("No models available")
        return None
    
    if market_index_df is None:
        market_index_df = get_market_index()
    
    df, scaler = get_intermediate_data(market, market_index_df, historical_df=historical_df)
    if df is None:
        return None
    
    X, y, scaler = create_final_sequences_and_scale(df, scaler)
    if X is None:
        return None
    
    last_sequence = X[-1]
    sequence_tensor = torch.as_tensor(last_sequence, dtype=torch.float32, device=config.Device.DEVICE).unsqueeze(0)
    
    # Use first model for explainability (representative)
    model = models[0]
    model.eval()
    
    result = analyze_prediction(model, sequence_tensor, market_name=market)
    result['market'] = market
    result['current_price'] = df.iloc[-1]['close']
    
    return result
