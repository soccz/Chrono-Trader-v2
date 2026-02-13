import torch
import numpy as np
import pandas as pd
import glob
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
from sklearn.calibration import calibration_curve

from tslearn.metrics import soft_dtw
from utils.model_tracker import get_tracker

# --- Global Model Cache for Performance ---
_MODEL_CACHE = None
_MODEL_CACHE_PATHS = None

def _get_cached_models():
    """Load models once and cache them globally for faster predictions."""
    global _MODEL_CACHE, _MODEL_CACHE_PATHS
    
    model_paths = sorted(glob.glob(os.path.join("models", "model_*.pth")))
    
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
    intermediate_data = {}
    for market in markets:
        df, scaler = get_intermediate_data(market, market_index_df, historical_df=historical_df)
        if df is not None:
            intermediate_data[market] = {'df': df, 'scaler': scaler}

    if not intermediate_data:
        logger.warning("No markets had sufficient data for pre-computation.")
        return []

    beta_series_list = [data['df']['beta'].rename(market) for market, data in intermediate_data.items()]
    all_betas_df = pd.concat(beta_series_list, axis=1)
    cs_beta_mean = all_betas_df.mean(axis=1)
    logger.info("Calculated cross-sectional mean beta for shrinkage.")

    all_predictions = []
    for market, data in intermediate_data.items():
        df = data['df']
        scaler = data['scaler']

        df['beta'] = 0.5 * df['beta'] + 0.5 * cs_beta_mean
        df.loc[:, 'beta'] = df['beta'].fillna(0)

        X, y, scaler = create_final_sequences_and_scale(df, scaler)
        
        if X is None:
            continue
        
        last_sequence = X[-1]
        sequence_tensor = torch.as_tensor(last_sequence, dtype=torch.float32, device=config.Device.DEVICE).unsqueeze(0)

        with torch.no_grad():
            individual_patterns = []
            gate_values = []
            model_directions = []

            model_avg_patterns = []  # Store average pattern per model for weighted voting
            
            for model_idx, model in enumerate(models):
                # MC-Dropout: Dropout stays active, BatchNorm in eval
                model.train()
                for module in model.modules():
                    if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                        module.eval()

                model_preds_local = []
                for _ in range(config.Recommender.MC_N_INFERENCES):
                    # Request explainability to get gate value (Trend vs Pattern)
                    # output: (prediction, explainability_dict)
                    out = model(sequence_tensor, return_explainability=True)
                    
                    # Handle return types (tuple vs tensor) safely
                    if isinstance(out, tuple):
                        pred_tensor = out[0]
                        explainability_info = out[1]  # This is a dict, not a tensor
                        
                        # Extract gate value from the dict
                        if isinstance(explainability_info, dict) and 'gate_info' in explainability_info:
                            gate_info = explainability_info['gate_info']
                            # gate_info could be a tensor or a dict itself
                            if isinstance(gate_info, torch.Tensor):
                                gate_val = gate_info.detach().cpu().item()
                            elif isinstance(gate_info, dict) and 'gate_values' in gate_info:
                                # gate_values is a numpy array, take mean
                                gate_val = float(np.mean(gate_info['gate_values']))
                            else:
                                gate_val = 0.5  # Default
                        else:
                            gate_val = 0.5
                    else:
                        pred_tensor = out
                        gate_val = 0.5

                    predicted_pattern = pred_tensor.detach().cpu().numpy().flatten()
                    
                    individual_patterns.append(predicted_pattern)
                    gate_values.append(gate_val)
                    model_preds_local.append(predicted_pattern)

                # Calculate direction for this specific model (for Consensus)
                avg_model_pattern = np.mean(model_preds_local, axis=0)
                model_avg_patterns.append(avg_model_pattern)  # For weighted voting
                # Direction: End - Start
                direction = 1 if avg_model_pattern[-1] > avg_model_pattern[0] else -1
                model_directions.append(direction)
        
        individual_patterns = np.array(individual_patterns)
        
        # --- Weighted Ensemble Voting ---
        tracker = get_tracker(n_models=len(models))
        model_weights = tracker.get_weights()
        model_avg_patterns = np.array(model_avg_patterns)
        
        # Weighted average of model predictions (instead of simple mean)
        final_pattern = np.average(model_avg_patterns, axis=0, weights=model_weights)
        
        # Calculate Consensus Score (Agreement % among models)
        n_up = model_directions.count(1)
        n_down = model_directions.count(-1)
        consensus_score = max(n_up, n_down) / len(models) # 1.0 = Unanimous, 0.6 = Split (3vs2)

        # Calculate Mean Gate Value
        final_gate_value = np.mean(gate_values)

        # Coefficient of Variation (CV = std/mean) normalization
        # Guard the denominator to avoid blow-ups when mean returns are near-zero.
        pattern_std = np.std(individual_patterns, axis=0)
        mean_abs = np.abs(np.mean(individual_patterns, axis=0))
        denom_floor = float(getattr(config.Recommender, "UNCERTAINTY_CV_DENOM_FLOOR", 0.002))
        denom = np.maximum(mean_abs, denom_floor) + 1e-8
        cv = pattern_std / denom
        final_uncertainty = np.mean(cv) * 100

        current_price = df.iloc[-1]['close']

        all_predictions.append({
            "market": market,
            "predicted_pattern": final_pattern,
            "uncertainty": final_uncertainty,
            "gate_value": final_gate_value,  # New: Trend(>0.6) vs Pattern(<0.4)
            "consensus_score": consensus_score, # New: Model Agreement
            "current_price": current_price,
            "individual_patterns": individual_patterns,
            "n_ensemble_models": len(models),
            "n_mc_inferences": config.Recommender.MC_N_INFERENCES
        })
        
        regime = "Trend" if final_gate_value > 0.6 else ("Pattern" if final_gate_value < 0.4 else "Hybrid")
        logger.info(f"Ensemble prediction for {market}: Unc={final_uncertainty:.2f}, Gate={final_gate_value:.2f}({regime}), Consensus={consensus_score:.2f}")

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
