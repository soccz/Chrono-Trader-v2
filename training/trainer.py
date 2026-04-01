import torch
import torch.optim as optim
import torch.nn as nn
import torch.autograd as autograd
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
import optuna
import functools
import json
import torch.serialization
import models.hybrid_model
import models.transformer_encoder
from scipy.stats import spearmanr
from sklearn.model_selection import TimeSeriesSplit


def _purged_walk_forward_splits(n: int, n_splits: int = 5, embargo: int = 6):
    """Purged Walk-Forward splits with embargo.
    Yields (train_idx, val_idx) where the last `embargo` samples before val are dropped.
    """
    fold_size = n // (n_splits + 1)
    splits = []
    for k in range(n_splits):
        train_end = fold_size * (k + 1)
        val_start = train_end + embargo
        val_end = val_start + fold_size
        if val_end > n:
            break
        train_idx = np.arange(0, train_end)
        val_idx = np.arange(val_start, val_end)
        splits.append((train_idx, val_idx))
    return splits

from utils.config import config
from utils.logger import logger
from data.preprocessor import (
    get_processed_data_for_training,
    get_market_index,
    get_intermediate_data,
    get_crypto_factors,
    create_train_val_sequences_and_scale,
)
from models.hybrid_model import build_model
from models.critic import build_critic

def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.getenv(name, str(default)) or str(default)).strip())
    except Exception:
        return int(default)

def _env_str(name: str, default: str = "") -> str:
    return str(os.getenv(name, default) or default).strip()

def _env_bool(name: str, default: bool = False) -> bool:
    v = str(os.getenv(name, "1" if default else "0") or ("1" if default else "0")).strip().lower()
    return v in ("1", "true", "yes", "y", "on")

def _write_optuna_checkpoint(study: optuna.Study, config_path: str) -> None:
    """Best-effort checkpoint so long Optuna runs survive interruptions."""
    try:
        best = getattr(study, "best_params", None)
        if not best:
            return
        ckpt = {
            "best_value": float(getattr(study, "best_value", float("nan"))),
            "best_params": dict(best),
            "n_trials": int(len(getattr(study, "trials", []) or [])),
        }
        os.makedirs(os.path.dirname(config_path) or ".", exist_ok=True)
        with open(config_path.replace(".json", "_checkpoint.json"), "w") as f:
            json.dump(ckpt, f, indent=2)
    except Exception as e:
        logger.debug(f"[Optuna] Failed to write checkpoint: {e}")

def _is_cuda() -> bool:
    return str(config.Device.DEVICE).startswith("cuda")

def _effective_batch_size(requested_batch_size: int) -> int:
    """
    Keep training stable on CPU-only environments by capping overly large batches.
    """
    if _is_cuda():
        return requested_batch_size
    cpu_cap = getattr(config.Gan, "MAX_BATCH_SIZE_CPU", 32)
    return max(8, min(requested_batch_size, cpu_cap))

def compute_gradient_penalty(critic, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN-GP and returns the gradient norm."""
    alpha = torch.rand(real_samples.size(0), 1).to(config.Device.DEVICE)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    critic_interpolates = critic(interpolates)
    gradients = autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(critic_interpolates.size()).to(config.Device.DEVICE),
        create_graph=True, retain_graph=True, only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    grad_norms = gradients.norm(2, dim=1)
    gradient_penalty = ((grad_norms - 1) ** 2).mean()
    return gradient_penalty, grad_norms.mean()

def objective(trial, X, y):
    """
    Optuna objective function using TimeSeriesSplit Cross-Validation to find hyperparameters 
    that MAXIMIZE the average uncertainty-error correlation, promoting generalization.
    It also penalizes trials where the adversarial loss is unstable.
    """
    # --- Hyperparameter Suggestions ---
    lr_g = trial.suggest_float("lr_g", 1e-5, 1e-3, log=True)
    lr_c = trial.suggest_float("lr_c", 1e-5, 1e-3, log=True)
    d_model = trial.suggest_categorical("d_model", [128, 256])
    n_layers = trial.suggest_int("n_layers", 2, 4)
    n_heads = trial.suggest_categorical("n_heads", [4, 8])
    batch_size = trial.suggest_categorical("batch_size", [32, 64])
    batch_size = _effective_batch_size(batch_size)
    lambda_recon_initial = trial.suggest_float("lambda_recon_initial", config.Gan.Dynamics.LAMBDA_RECON_MIN, config.Gan.Dynamics.LAMBDA_RECON_MAX, log=True)
    lambda_gp_initial = trial.suggest_float("lambda_gp_initial", config.Gan.Dynamics.LAMBDA_GP_MIN, config.Gan.Dynamics.LAMBDA_GP_MAX, log=True)
    lambda_ece = trial.suggest_float("lambda_ece", 0.05, 0.5) # Narrowed range based on feedback
    critic_base_iters = trial.suggest_int("critic_base_iters", 3, 10) # Widened range based on feedback
    dropout_p = trial.suggest_float("dropout_p", 0.05, 0.35)
    lambda_direction = trial.suggest_float("lambda_direction", 0.1, 1.0)

    # --- K-Fold Cross-Validation Setup ---
    n_splits = 5
    splits = _purged_walk_forward_splits(len(X), n_splits=n_splits, embargo=6)
    fold_correlations = []

    logger.info(f"Trial {trial.number}: Starting {n_splits}-Fold Purged Walk-Forward CV (embargo=6)...")

    for fold, (train_index, val_index) in enumerate(splits):
        logger.info(f"  - Fold {fold+1}/{n_splits}...")
        try:
            # --- Data Setup for current fold ---
            X_train_fold, X_val_fold = X[train_index], X[val_index]
            y_train_fold, y_val_fold = y[train_index], y[val_index]

            train_dataset = TensorDataset(torch.FloatTensor(X_train_fold), torch.FloatTensor(y_train_fold))
            val_dataset = TensorDataset(torch.FloatTensor(X_val_fold), torch.FloatTensor(y_val_fold))
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                pin_memory=_is_cuda(),
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=_is_cuda(),
            )

            # --- Model & Optimizer Setup ---
            decoder_mode = getattr(config.Gan, "DECODER_MODE", "gan")
            pe_mode = getattr(config.Gan, "PE_MODE", "cycle_pe")
            generator = build_model(
                d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                input_dim=X.shape[-1], noise_dim=config.Gan.NOISE_DIM,
                output_dim=y.shape[-1], dropout_p=dropout_p,
                decoder_mode=decoder_mode,
                pe_mode=pe_mode,
            )
            optimizer_G = optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.9))
            critic = None
            optimizer_C = None
            if decoder_mode == "gan":
                critic = build_critic()
                optimizer_C = optim.Adam(critic.parameters(), lr=lr_c, betas=(0.5, 0.9))
            recon_criterion = nn.MSELoss()
            
            # Use hyperparameters suggested by Optuna for this trial
            lambda_recon_run = float(lambda_recon_initial)
            lambda_gp_run = float(lambda_gp_initial)
            # Note: lambda_ece and critic_base_iters are already defined from the trial suggestions
            target_adv_ratio = float(config.Gan.Dynamics.TARGET_ADV_RATIO)

            # --- Training Loop (shortened for tuning) ---
            fold_adv_losses = [] # For stability penalty
            for epoch in range(15): # A reasonable number of epochs for a single trial
                generator.train()
                if critic is not None:
                    critic.train()
                for batch_idx, (real_sequences, real_paths) in enumerate(train_loader):
                    real_sequences, real_paths = real_sequences.to(config.Device.DEVICE), real_paths.to(config.Device.DEVICE)
                    if real_sequences.size(0) <= 1: continue

                    optimizer_G.zero_grad()
                    if decoder_mode == "cvae":
                        gen_paths = generator.forward_train(real_sequences, real_paths)
                        # gen_paths is now mu (mean prediction)
                        mu = getattr(generator, "_cvae_mu", gen_paths)
                        log_sigma = getattr(generator, "_cvae_log_sigma", None)
                        loss_G_adv = torch.tensor(0.0, device=config.Device.DEVICE)
                        grad_norm = torch.tensor(0.0, device=config.Device.DEVICE)
                        kl_loss = torch.tensor(0.0, device=config.Device.DEVICE)  # no KL
                    else:
                        # Use critic_base_iters suggested by Optuna
                        for _ in range(critic_base_iters):
                            optimizer_C.zero_grad()
                            fake_paths = generator(real_sequences).detach()
                            real_validity = critic(real_paths)
                            fake_validity_critic = critic(fake_paths)
                            gradient_penalty, grad_norm = compute_gradient_penalty(critic, real_paths.data, fake_paths.data)
                            loss_C = fake_validity_critic.mean() - real_validity.mean() + lambda_gp_run * gradient_penalty
                            loss_C.backward()
                            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
                            optimizer_C.step()

                        gen_paths = generator(real_sequences)
                        loss_G_adv = -critic(gen_paths).mean()
                        kl_loss = torch.tensor(0.0, device=config.Device.DEVICE)
                        mu = gen_paths
                        log_sigma = None
                    if log_sigma is not None:
                        # Gaussian NLL: log_sigma + 0.5 * (y - mu)^2 / sigma^2
                        loss_G_recon = (log_sigma + 0.5 * (real_paths - mu).pow(2) / (torch.exp(log_sigma).pow(2) + 1e-8)).mean()
                    else:
                        loss_G_recon = recon_criterion(gen_paths, real_paths)
                    # --- Focal-style direction loss (replaces broken sigmoid(mean_return) ECE) ---
                    pred_return = gen_paths.mean(dim=1)
                    real_return = real_paths.mean(dim=1)
                    fused_ctx = getattr(generator, '_cvae_fused_context', None)
                    if fused_ctx is not None and hasattr(generator, 'decoder') and hasattr(generator.decoder, 'confidence_head'):
                        pred_prob = generator.decoder.confidence_head(fused_ctx.detach()).squeeze(-1)
                    else:
                        pred_prob = torch.sigmoid(pred_return * 10.0)
                    target_prob = (real_return > 0).float()
                    pt = target_prob * pred_prob + (1 - target_prob) * (1 - pred_prob)
                    focal_weight = (1 - pt) ** 2
                    loss_G_ece = (focal_weight * nn.functional.binary_cross_entropy(pred_prob.clamp(1e-6, 1 - 1e-6), target_prob, reduction='none')).mean()

                    # --- Direction Balance Loss ---
                    pred_direction_prob = torch.sigmoid(gen_paths[:, -1] - gen_paths[:, 0])
                    direction_balance = torch.abs(pred_direction_prob.mean() - 0.5)
                    loss_G_balance = direction_balance * 0.1

                    # --- Directional Loss (penalizes wrong direction predictions) ---
                    pred_dir_logit = gen_paths[:, -1] - gen_paths[:, 0]
                    real_dir = (real_paths[:, -1] > real_paths[:, 0]).float()
                    loss_G_direction = torch.nn.functional.binary_cross_entropy_with_logits(
                        pred_dir_logit, real_dir
                    )

                    # --- Weighted MSE (larger movements get higher weight) ---
                    movement_magnitude = real_paths.abs().sum(dim=1)
                    weights = 1.0 + movement_magnitude / (movement_magnitude.mean() + 1e-8)
                    loss_G_recon_weighted = (weights * (gen_paths - real_paths).pow(2).mean(dim=1)).mean()

                    # --- Gate Regularization Loss ---
                    loss_G_gate_reg = getattr(generator, '_gate_reg_loss', torch.tensor(0.0).to(config.Device.DEVICE))

                    kl_weight = 0.0  # heteroscedastic: no KL term

                    loss_G = (
                        loss_G_adv
                        + lambda_recon_run * loss_G_recon  # NLL (includes log_sigma gradient)
                        + lambda_ece * loss_G_ece
                        + loss_G_balance
                        + loss_G_gate_reg * 0.5
                        + lambda_direction * loss_G_direction
                    )
                    loss_G.backward()
                    torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                    optimizer_G.step()

                    if decoder_mode == "gan":
                        # Track adversarial loss for stability penalty
                        fold_adv_losses.append(loss_G_adv.item())

                        # Dynamic balancing for reconstruction weight (respects config min/max)
                        adv_abs = loss_G_adv.abs().item()
                        recon_val = loss_G_recon.abs().item() + 1e-9
                        current_ratio = adv_abs / recon_val
                        if current_ratio < target_adv_ratio * 0.5:
                            lambda_recon_run = max(config.Gan.Dynamics.LAMBDA_RECON_MIN, lambda_recon_run * 0.9)
                        elif current_ratio > target_adv_ratio * 1.5:
                            lambda_recon_run = min(config.Gan.Dynamics.LAMBDA_RECON_MAX, lambda_recon_run * 1.1)

                        # Dynamic gradient penalty adjustment (respects config min/max)
                        if grad_norm.item() < 0.8:
                            lambda_gp_run = min(lambda_gp_run * 1.05, config.Gan.Dynamics.LAMBDA_GP_MAX)
                        elif grad_norm.item() > 1.2:
                            lambda_gp_run = max(lambda_gp_run * 0.95, config.Gan.Dynamics.LAMBDA_GP_MIN)

            # --- Evaluation for Uncertainty Correlation on the fold ---
            N_INFERENCES_TUNE = 30
            all_errors = []
            all_uncertainties = []

            if decoder_mode == "gan":
                generator.train() # Enable dropout for MC-Dropout
                for module in generator.modules():
                    if isinstance(module, torch.nn.BatchNorm1d):
                        module.eval()
            else:
                generator.eval()

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(config.Device.DEVICE), batch_y.to(config.Device.DEVICE)
                    
                    batch_mc_preds = []
                    for _ in range(N_INFERENCES_TUNE):
                        preds = generator(batch_X).cpu().numpy()
                        batch_mc_preds.append(preds)
                    
                    batch_mc_preds = np.array(batch_mc_preds)
                    
                    for i in range(batch_X.size(0)):
                        sample_preds = batch_mc_preds[:, i, :]
                        mean_pred = sample_preds.mean(axis=0)
                        uncertainty = np.sum(sample_preds.std(axis=0))
                        
                        true_path = batch_y[i].cpu().numpy()
                        error = np.mean(np.abs(mean_pred - true_path))
                        
                        all_errors.append(error)
                        all_uncertainties.append(uncertainty)

            correlation, _ = spearmanr(all_errors, all_uncertainties)
            score = correlation if not np.isnan(correlation) else -2.0 # Base score is correlation, penalize NaN

            # --- Apply Gentler Stability Penalty based on lossG_adv ---
            if fold_adv_losses:
                avg_adv_loss = np.mean(fold_adv_losses)
                lower_bound, upper_bound = config.Gan.Dynamics.TARGET_ADV_LOSS_RANGE
                penalty_buffer = 0.02  # Dead zone as suggested
                penalty = 0.0

                if avg_adv_loss < lower_bound - penalty_buffer:
                    penalty = abs(avg_adv_loss - (lower_bound - penalty_buffer))
                elif avg_adv_loss > upper_bound + penalty_buffer:
                    penalty = abs(avg_adv_loss - (upper_bound + penalty_buffer))

                if penalty > 0:
                    adjusted_penalty = penalty * config.Gan.Optuna.PENALTY_SCALING_FACTOR
                    score -= adjusted_penalty
                    logger.warning(f"  - Fold {fold+1} PENALIZED: Adv Loss ({avg_adv_loss:.3f}) was outside buffered target range [{lower_bound - penalty_buffer:.3f}, {upper_bound + penalty_buffer:.3f}]. Score adjusted by -{adjusted_penalty:.3f} (raw: -{penalty:.3f}).")
                else:
                    logger.info(f"  - Fold {fold+1} STABLE: Adv Loss ({avg_adv_loss:.3f}) is within target range.")

            fold_correlations.append(score)

        except Exception as e:
            logger.error(f"  - Fold {fold+1} failed with exception: {e}")
            fold_correlations.append(-2.0) # Penalize trial if a fold fails catastrophically
            continue # Move to the next fold

    # --- Final Objective Score ---
    average_correlation = np.mean(fold_correlations)
    logger.info(f"Trial {trial.number}: Average correlation across folds: {average_correlation:.4f}")

    trial.report(average_correlation, 0)
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return average_correlation

def run(markets: list = None, tune: bool = False, epochs: int = None):
    # Reproducibility baseline for more stable comparisons across runs
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    config_path = os.path.join("models", "model_config.json")
    if os.path.exists(config_path) and not tune:
        logger.info(f"Loading model configuration from {config_path}")
        with open(config_path, 'r') as f:
            best_params = json.load(f)
            # --- Remap Optuna params to config for actual training ---
            config.Gan.LEARNING_RATE_G = best_params.get('lr_g', config.Gan.LEARNING_RATE_G)
            config.Gan.LEARNING_RATE_C = best_params.get('lr_c', config.Gan.LEARNING_RATE_C)
            config.Gan.D_MODEL = best_params.get('d_model', config.Gan.D_MODEL)
            config.Gan.N_LAYERS = best_params.get('n_layers', config.Gan.N_LAYERS)
            config.Gan.N_HEADS = best_params.get('n_heads', config.Gan.N_HEADS)
            config.Gan.BATCH_SIZE = best_params.get('batch_size', config.Gan.BATCH_SIZE)
            config.Gan.DROPOUT_P = best_params.get('dropout_p', config.Gan.DROPOUT_P)
            # Remap GAN/ECE specific hyperparameters
            config.Gan.LAMBDA_RECON = best_params.get('lambda_recon_initial', config.Gan.LAMBDA_RECON)
            config.Gan.LAMBDA_GP = best_params.get('lambda_gp_initial', config.Gan.LAMBDA_GP)
            config.Gan.LAMBDA_ECE = best_params.get('lambda_ece', config.Gan.LAMBDA_ECE)
            config.Gan.CRITIC_BASE_ITERS = best_params.get('critic_base_iters', config.Gan.CRITIC_BASE_ITERS)
            logger.info(f"Loaded and remapped tuned hyperparameters: {best_params}")

    market_index_df = get_market_index()
    
    if tune:
        logger.info("--- Starting Hyperparameter Tuning with Optuna ---")
        # Use a mid-size alt for tuning to avoid OOM on BTC (62k rows)
        tune_market = _env_str("AETHER_OPTUNA_TUNE_MARKET", "KRW-XRP")
        X, y, _ = get_processed_data_for_training(tune_market, market_index_df)
        if X is None: return
        # Cap at recent 5000 samples to keep each trial fast
        MAX_TUNE_SAMPLES = int(_env_str("AETHER_OPTUNA_MAX_SAMPLES", "5000") or "5000")
        if len(X) > MAX_TUNE_SAMPLES:
            X, y = X[-MAX_TUNE_SAMPLES:], y[-MAX_TUNE_SAMPLES:]
            logger.info(f"Tuning data capped at {MAX_TUNE_SAMPLES} samples (most recent)")

        # IMPORTANT: We are now MAXIMIZING the correlation
        opt_trials = _env_int("AETHER_OPTUNA_TRIALS", 50)
        opt_timeout = _env_int("AETHER_OPTUNA_TIMEOUT_SEC", 0)
        opt_n_jobs = max(1, _env_int("AETHER_OPTUNA_N_JOBS", 1))
        opt_storage = _env_str("AETHER_OPTUNA_STORAGE", "")
        opt_study_name = _env_str("AETHER_OPTUNA_STUDY_NAME", "aether_gan_tune")
        opt_load_if_exists = _env_bool("AETHER_OPTUNA_LOAD_IF_EXISTS", True)

        create_kwargs = {"direction": "maximize"}
        if opt_storage:
            # Make sure target folder exists for sqlite:///analysis/....
            if opt_storage.startswith("sqlite:///"):
                try:
                    sqlite_path = opt_storage[len("sqlite:///") :]
                    os.makedirs(os.path.dirname(sqlite_path) or ".", exist_ok=True)
                except Exception:
                    pass
            create_kwargs.update(
                {
                    "storage": opt_storage,
                    "study_name": opt_study_name,
                    "load_if_exists": bool(opt_load_if_exists),
                }
            )

        study = optuna.create_study(**create_kwargs)
        objective_with_data = functools.partial(objective, X=X, y=y)
        study.optimize(
            objective_with_data,
            n_trials=int(opt_trials),
            timeout=(None if opt_timeout <= 0 else int(opt_timeout)),
            n_jobs=int(opt_n_jobs),
            gc_after_trial=True,
            catch=(Exception,),
            callbacks=[lambda s, t: _write_optuna_checkpoint(s, config_path)],
        )

        best_params = study.best_params
        logger.info(f"Tuning finished. Best parameters found: {best_params}")

        with open(config_path, 'w') as f: json.dump(best_params, f, indent=4)
        logger.info(f"Best parameters saved to {config_path}")

        # --- Remap Optuna params to config for the rest of the run ---
        config.Gan.LEARNING_RATE_G = best_params['lr_g']
        config.Gan.LEARNING_RATE_C = best_params['lr_c']
        config.Gan.D_MODEL = best_params['d_model']
        config.Gan.N_LAYERS = best_params['n_layers']
        config.Gan.N_HEADS = best_params['n_heads']
        config.Gan.BATCH_SIZE = best_params['batch_size']
        config.Gan.DROPOUT_P = best_params.get('dropout_p', config.Gan.DROPOUT_P)
        # Remap GAN/ECE specific hyperparameters
        config.Gan.LAMBDA_RECON = best_params['lambda_recon_initial']
        config.Gan.LAMBDA_GP = best_params['lambda_gp_initial']
        config.Gan.LAMBDA_ECE = best_params['lambda_ece']
        config.Gan.CRITIC_BASE_ITERS = best_params['critic_base_iters']
        logger.info("Configuration updated with best parameters for this run.")

    is_finetuning = markets is not None
    mode_log = "Fine-tuning" if is_finetuning else "Full Training"
    logger.info(f"--- Preparing for {mode_log} ---")

    if is_finetuning:
        target_markets = list(markets or [])
        lr = config.Gan.LEARNING_RATE_G / 10
    else: # Full Training on all target markets
        from data.database import get_top_markets_by_trading_value
        exclude_tokens = tuple(t.upper() for t in getattr(config.Data, "DYNAMIC_UNIVERSE_EXCLUDE", []))
        top_n = int(getattr(config.Data, "DYNAMIC_UNIVERSE_TOP_N", 100))
        dynamic_markets = get_top_markets_by_trading_value(limit=top_n, hours=24)
        dynamic_markets = [
            m for m in dynamic_markets
            if not any(tok in m.upper() for tok in exclude_tokens)
        ]
        target_markets = dynamic_markets or getattr(config.Data, "TRAIN_COINS_FALLBACK", config.Data.MARKET_INDEX_COINS)
        logger.info(f"Dynamic universe: {len(target_markets)} markets (top {top_n} by 24h volume)")
        lr = config.Gan.LEARNING_RATE_G

    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    crypto_factors_df, rank_4h_df = get_crypto_factors()
    for market in target_markets:
        df_market, _ = get_intermediate_data(
            market,
            market_index_df,
            historical_df=None,
            crypto_factors_df=crypto_factors_df,
            rank_4h_df=rank_4h_df,
        )
        if df_market is None:
            continue
        X_train_market, y_train_market, X_val_market, y_val_market, _ = create_train_val_sequences_and_scale(
            df_market,
            scaler=MinMaxScaler(),
            train_ratio=config.Gan.TRAIN_SPLIT,
        )
        if X_train_market is None or X_val_market is None:
            continue
        X_train_list.append(X_train_market)
        y_train_list.append(y_train_market)
        X_val_list.append(X_val_market)
        y_val_list.append(y_val_market)

    if not X_train_list or not X_val_list:
        logger.error("Training failed: no chronological train/validation sequences could be assembled.")
        return

    X_train_all = np.concatenate(X_train_list, axis=0)
    y_train_all = np.concatenate(y_train_list, axis=0)
    X_val_all = np.concatenate(X_val_list, axis=0)
    y_val_all = np.concatenate(y_val_list, axis=0)
    logger.info(
        f"Assembled chronological dataset with train X:{X_train_all.shape}, y:{y_train_all.shape} | "
        f"val X:{X_val_all.shape}, y:{y_val_all.shape}"
    )

    # Temporal holdout: carve out last 20% of val sequences as out-of-sample test set.
    # Val sequences are already sorted chronologically (after train), so tail = most recent.
    holdout_ratio = 0.20
    holdout_start = int(len(X_val_all) * (1 - holdout_ratio))
    X_holdout = X_val_all[holdout_start:]
    y_holdout = y_val_all[holdout_start:]
    X_val_all = X_val_all[:holdout_start]
    y_val_all = y_val_all[:holdout_start]
    logger.info(
        f"[HOLDOUT] Reserved last {holdout_ratio*100:.0f}% of val set as holdout: "
        f"holdout X:{X_holdout.shape}, y:{y_holdout.shape} | "
        f"remaining val X:{X_val_all.shape}, y:{y_val_all.shape}"
    )

    # If epochs is not passed as an argument, use config defaults
    if epochs is None:
        epochs = config.Gan.EPOCHS

    logger.info(
        "Model modes: decoder_mode=%s, pe_mode=%s",
        getattr(config.Gan, "DECODER_MODE", "gan"),
        getattr(config.Gan, "PE_MODE", "cycle_pe"),
    )

    effective_batch_size = _effective_batch_size(config.Gan.BATCH_SIZE)
    if effective_batch_size != config.Gan.BATCH_SIZE:
        logger.info(
            f"CPU environment detected: reducing batch size from "
            f"{config.Gan.BATCH_SIZE} to {effective_batch_size} for stability."
        )

    # --- Load Diverse Ensemble Configurations ---
    ensemble_config_path = os.path.join("models", "ensemble_configs.json")
    if os.path.exists(ensemble_config_path):
        with open(ensemble_config_path, 'r') as f:
            ensemble_configs = json.load(f)
        model_configs = ensemble_configs.get("models", [])
        logger.info(f"Loaded {len(model_configs)} diverse ensemble configurations from {ensemble_config_path}")
    else:
        # Fallback: generate default configs if ensemble_configs.json doesn't exist
        logger.warning(f"No ensemble_configs.json found. Using default uniform configuration.")
        model_configs = [
            {"id": i+1, "d_model": config.Gan.D_MODEL, "n_layers": config.Gan.N_LAYERS, 
             "n_heads": config.Gan.N_HEADS, "dropout_p": config.Gan.DROPOUT_P, 
             "cnn_mode": config.Gan.CNN_MODE, "bagging_ratio": 0.9, "name": f"Default Model {i+1}"}
            for i in range(config.Gan.N_ENSEMBLE_MODELS)
        ]

    # Respect configured ensemble size for compute-aware training.
    # If ensemble_configs.json contains more configs than we want to train, truncate deterministically.
    ensemble_limit = getattr(config.Gan, "N_ENSEMBLE_MODELS", None)
    try:
        ensemble_limit = int(ensemble_limit) if ensemble_limit is not None else None
    except Exception:
        ensemble_limit = None
    if ensemble_limit is not None and ensemble_limit > 0 and len(model_configs) > ensemble_limit:
        logger.info(
            f"Limiting ensemble training configs from {len(model_configs)} to {ensemble_limit} "
            f"based on config.Gan.N_ENSEMBLE_MODELS."
        )
        model_configs = model_configs[:ensemble_limit]

    for model_cfg in model_configs:
        model_id = model_cfg["id"]
        model_name = model_cfg.get("name", f"Model {model_id}")
        d_model = model_cfg.get("d_model", config.Gan.D_MODEL)
        n_layers = model_cfg.get("n_layers", config.Gan.N_LAYERS)
        n_heads = model_cfg.get("n_heads", config.Gan.N_HEADS)
        dropout_p = model_cfg.get("dropout_p", config.Gan.DROPOUT_P)
        cnn_mode = model_cfg.get("cnn_mode", config.Gan.CNN_MODE)
        bagging_ratio = model_cfg.get("bagging_ratio", 0.9)
        
        lookback_hours = model_cfg.get("lookback_hours", None)
        cnn_kernel_size = model_cfg.get("cnn_kernel_size", None)

        logger.info(f"\n--- Training Ensemble Model {model_id}/{len(model_configs)}: {model_name} ---")
        logger.info(f"    d_model={d_model}, n_layers={n_layers}, n_heads={n_heads}, dropout={dropout_p}, cnn={cnn_mode}, bagging={bagging_ratio}, lookback={lookback_hours}")

        # Temporarily override CNN_MODE for this model
        original_cnn_mode = config.Gan.CNN_MODE
        config.Gan.CNN_MODE = cnn_mode

        # Per-model lookback slicing: trim sequences temporally if lookback_hours < SEQUENCE_LENGTH
        if lookback_hours is not None and lookback_hours < X_train_all.shape[1]:
            X_train = X_train_all[:, -lookback_hours:, :]
            X_val = X_val_all[:, -lookback_hours:, :]
            X_holdout_model = X_holdout[:, -lookback_hours:, :]
            logger.info(f"    Lookback slicing: {X_train_all.shape[1]}h -> {lookback_hours}h (seq_len={X_train.shape[1]})")
        else:
            X_train = X_train_all
            X_val = X_val_all
            X_holdout_model = X_holdout
        y_train = y_train_all
        y_val = y_val_all

        full_train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        n_samples = len(full_train_dataset)
        subset_size = int(n_samples * bagging_ratio)
        bagging_rng = np.random.default_rng(seed=42 + model_id)
        subset_indices = bagging_rng.choice(n_samples, subset_size, replace=True)
        train_subset = Subset(full_train_dataset, subset_indices)
        logger.info(f"Model {model_id} will be trained on a random subset of {len(train_subset)} samples (Bagging with ratio {bagging_ratio}).")

        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
        train_loader = DataLoader(
            train_subset,
            batch_size=effective_batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=_is_cuda(),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=effective_batch_size,
            shuffle=False,
            pin_memory=_is_cuda(),
        )

        model_save_path = os.path.join("models", f"model_{model_id}.pth")
        if is_finetuning and os.path.exists(model_save_path):
            try:
                generator = torch.load(model_save_path, map_location=config.Device.DEVICE, weights_only=False)
                loaded_input_dim = getattr(generator, "input_dim", None)
                expected_input_dim = X_train.shape[-1]
                if loaded_input_dim is not None and loaded_input_dim != expected_input_dim:
                    raise ValueError(
                        f"Checkpoint input_dim mismatch (loaded={loaded_input_dim}, expected={expected_input_dim})"
                    )
                logger.info(f"Loaded model {model_id} for fine-tuning.")
            except Exception as e:
                logger.error(f"Could not load model {model_id}: {e}. Rebuilding.")
                generator = build_model(
                    d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                    input_dim=X_train.shape[-1], noise_dim=config.Gan.NOISE_DIM,
                    output_dim=y_train.shape[-1], dropout_p=dropout_p,
                    decoder_mode=getattr(config.Gan, "DECODER_MODE", "gan"),
                    pe_mode=getattr(config.Gan, "PE_MODE", "cycle_pe"),
                )
        else:
            generator = build_model(
                d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                input_dim=X_train.shape[-1], noise_dim=config.Gan.NOISE_DIM,
                output_dim=y_train.shape[-1], dropout_p=dropout_p,
                decoder_mode=getattr(config.Gan, "DECODER_MODE", "gan"),
                pe_mode=getattr(config.Gan, "PE_MODE", "cycle_pe"),
            )
        
        decoder_mode = getattr(config.Gan, "DECODER_MODE", "gan")
        # Use separate learning rates for Generator and Critic (TTUR)
        lr_g = lr # The 'lr' variable is now correctly set to LEARNING_RATE_G
        lr_c = config.Gan.LEARNING_RATE_C if not is_finetuning else config.Gan.LEARNING_RATE_C / 10
        optimizer_G = optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.9))
        critic = None
        optimizer_C = None
        if decoder_mode == "gan":
            critic = build_critic()
            optimizer_C = optim.Adam(critic.parameters(), lr=lr_c, betas=(0.5, 0.9))
        
        # Learning Rate Schedulers (Cosine Annealing for smooth decay)
        scheduler_G = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=epochs, eta_min=lr_g * 0.01)
        scheduler_C = None
        if optimizer_C is not None:
            scheduler_C = optim.lr_scheduler.CosineAnnealingLR(optimizer_C, T_max=epochs, eta_min=lr_c * 0.01)
        
        if optimizer_C is not None:
            logger.info(f"Optimizers created. Generator LR: {lr_g:.6f}, Critic LR: {lr_c:.6f}")
        else:
            logger.info(f"Optimizer created. Generator LR: {lr_g:.6f} (CVAE mode)")
        logger.info(f"LR Schedulers: CosineAnnealingLR with T_max={epochs}")
        recon_criterion = nn.MSELoss()
        best_val_recon_loss = float('inf')
        lambda_recon_run = config.Gan.LAMBDA_RECON
        lambda_gp_run = config.Gan.LAMBDA_GP
        lambda_ece = config.Gan.LAMBDA_ECE
        lambda_direction = config.Gan.LAMBDA_DIRECTION
        target_adv_ratio = config.Gan.Dynamics.TARGET_ADV_RATIO
        lambda_recon_min = config.Gan.Dynamics.LAMBDA_RECON_MIN
        lambda_recon_max = config.Gan.Dynamics.LAMBDA_RECON_MAX
        critic_base_iters = config.Gan.CRITIC_BASE_ITERS
        critic_max_iters = config.Gan.Dynamics.CRITIC_MAX_ITERS
        critic_min_iters = config.Gan.Dynamics.CRITIC_MIN_ITERS

        loss_G_adv = torch.tensor(1.0)
        total_steps = 0

        # --- Initialize for Auto-stopping ---
        rules = config.Gan.AutoStop.GAN_STOP_RULES
        metric_history = {'ratio': [], 'grad_norm': []}
        strong_stop_counters = {'ratio_sustained': 0, 'grad_norm_sustained': 0}
        stop_training_flag = False

        for epoch in range(epochs):
            generator.train()
            if critic is not None:
                critic.train()
            for batch_idx, (real_sequences, real_paths) in enumerate(train_loader):
                real_sequences, real_paths = real_sequences.to(config.Device.DEVICE), real_paths.to(config.Device.DEVICE)
                if real_sequences.size(0) <= 1: continue

                optimizer_G.zero_grad()
                if decoder_mode == "gan":
                    c_iters = critic_base_iters
                    if total_steps > config.Gan.Dynamics.GAN_WARMUP_STEPS:
                        if loss_G_adv.item() < config.Gan.Dynamics.TARGET_ADV_LOSS:
                            c_iters = min(critic_max_iters, c_iters + 2)
                        else:
                            c_iters = max(critic_min_iters, c_iters - 1)

                    # Critic updates (WGAN-GP)
                    for _ in range(c_iters):
                        optimizer_C.zero_grad()
                        fake_paths = generator(real_sequences).detach()
                        real_validity = critic(real_paths)
                        fake_validity_critic = critic(fake_paths)
                        gradient_penalty, grad_norm = compute_gradient_penalty(
                            critic, real_paths.data, fake_paths.data
                        )
                        loss_C = fake_validity_critic.mean() - real_validity.mean() + lambda_gp_run * gradient_penalty
                        loss_C.backward()
                        torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
                        optimizer_C.step()

                    # Generator update (exactly once per batch for stability)
                    gen_paths = generator(real_sequences)
                    loss_G_adv = -critic(gen_paths).mean()
                    kl_loss = torch.tensor(0.0, device=config.Device.DEVICE)
                    mu = gen_paths
                    log_sigma = None
                else:
                    c_iters = 0
                    grad_norm = torch.tensor(0.0, device=config.Device.DEVICE)
                    gradient_penalty = torch.tensor(0.0, device=config.Device.DEVICE)
                    loss_C = torch.tensor(0.0, device=config.Device.DEVICE)
                    real_validity = torch.tensor(0.0, device=config.Device.DEVICE)
                    fake_validity_critic = torch.tensor(0.0, device=config.Device.DEVICE)
                    gen_paths = generator.forward_train(real_sequences, real_paths)
                    # gen_paths is now mu (mean prediction)
                    mu = getattr(generator, "_cvae_mu", gen_paths)
                    log_sigma = getattr(generator, "_cvae_log_sigma", None)
                    loss_G_adv = torch.tensor(0.0, device=config.Device.DEVICE)
                    kl_loss = torch.tensor(0.0, device=config.Device.DEVICE)  # no KL
                if log_sigma is not None:
                    # Gaussian NLL: log_sigma + 0.5 * (y - mu)^2 / sigma^2
                    loss_G_recon = (log_sigma + 0.5 * (real_paths - mu).pow(2) / (torch.exp(log_sigma).pow(2) + 1e-8)).mean()
                else:
                    loss_G_recon = recon_criterion(gen_paths, real_paths)

                # --- Focal-style direction loss (replaces broken sigmoid(mean_return) ECE) ---
                pred_return = gen_paths.mean(dim=1)
                real_return = real_paths.mean(dim=1)
                fused_ctx = getattr(generator, '_cvae_fused_context', None)
                if fused_ctx is not None and hasattr(generator, 'decoder') and hasattr(generator.decoder, 'confidence_head'):
                    pred_prob = generator.decoder.confidence_head(fused_ctx.detach()).squeeze(-1)
                else:
                    pred_prob = torch.sigmoid(pred_return * 10.0)  # sharper sigmoid with scaling
                target_prob = (real_return > 0).float()
                # Focal-style: down-weight easy examples
                pt = target_prob * pred_prob + (1 - target_prob) * (1 - pred_prob)
                focal_weight = (1 - pt) ** 2
                loss_G_ece = (focal_weight * nn.functional.binary_cross_entropy(pred_prob.clamp(1e-6, 1 - 1e-6), target_prob, reduction='none')).mean()

                # --- Direction Balance Loss ---
                pred_direction_prob = torch.sigmoid(gen_paths[:, -1] - gen_paths[:, 0])
                direction_balance = torch.abs(pred_direction_prob.mean() - 0.5)
                loss_G_balance = direction_balance * 0.1

                # --- Directional Loss (differentiable) ---
                real_direction = (real_paths[:, -1] > real_paths[:, 0]).float()
                direction_logits = gen_paths[:, -1] - gen_paths[:, 0]
                loss_G_direction = torch.nn.functional.binary_cross_entropy_with_logits(
                    direction_logits, real_direction
                )

                # --- Weighted MSE (larger movements get higher weight) ---
                movement_magnitude = real_paths.abs().sum(dim=1)
                weights = 1.0 + movement_magnitude / (movement_magnitude.mean() + 1e-8)
                loss_G_recon_weighted = (weights * (gen_paths - real_paths).pow(2).mean(dim=1)).mean()

                # --- Gate Regularization Loss ---
                loss_G_gate_reg = getattr(generator, '_gate_reg_loss', torch.tensor(0.0).to(config.Device.DEVICE))

                kl_weight = 0.0  # heteroscedastic: no KL term

                loss_G = (
                    loss_G_adv
                    + lambda_recon_run * loss_G_recon  # NLL (includes log_sigma gradient)
                    + lambda_ece * loss_G_ece
                    + loss_G_balance
                    + loss_G_gate_reg * 0.5
                    + lambda_direction * loss_G_direction
                )
                loss_G.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                optimizer_G.step()

                total_steps += 1

                if decoder_mode == "gan":
                    # Keep metrics updated for auto-stop checks
                    monitor_recon_floor = max(abs(loss_G_recon.item()), 1e-2)
                    ratio_value = float(loss_G_adv.item() / monitor_recon_floor)
                    metric_history['ratio'].append(ratio_value)
                    metric_history['grad_norm'].append(grad_norm.item())

                    if batch_idx % 20 == 0:
                        w_est = real_validity.mean() - fake_validity_critic.mean()
                        logger.info(
                            f"Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx}/{len(train_loader)}] | "
                            f"C_iters: {c_iters} | W_est:{w_est:.4f} | ||∇D||:{grad_norm:.3f} | GP:{gradient_penalty:.3f} | "
                            f"lossC:{loss_C:.3f} | lossG_adv:{loss_G_adv:.3f} | lossG_recon:{loss_G_recon:.3f} | "
                            f"lossG_ece:{loss_G_ece:.4f} | ratio:{ratio_value:.3f}"
                        )

                    # --- GAN Stability Check (Auto-stopping) ---
                    if total_steps > rules['warmup_steps'] and total_steps % rules['check_interval'] == 0:
                        window = rules['moving_avg_window']
                        if len(metric_history['ratio']) >= window:
                            avg_ratio = np.mean(metric_history['ratio'][-window:])
                            avg_grad_norm = np.mean(metric_history['grad_norm'][-window:])

                            if not (rules['warnings']['ratio_range'][0] <= avg_ratio <= rules['warnings']['ratio_range'][1]):
                                logger.warning(f"[Auto-stop] Warning: Moving avg of ratio ({avg_ratio:.3f}) is outside the warning range {rules['warnings']['ratio_range']}.")
                            if not (rules['warnings']['grad_norm_range'][0] <= avg_grad_norm <= rules['warnings']['grad_norm_range'][1]):
                                logger.warning(f"[Auto-stop] Warning: Moving avg of grad_norm ({avg_grad_norm:.3f}) is outside the warning range {rules['warnings']['grad_norm_range']}.")

                            if avg_ratio < rules['strong_stop']['ratio_lower_bound']:
                                strong_stop_counters['ratio_sustained'] += rules['check_interval']
                            else:
                                strong_stop_counters['ratio_sustained'] = 0

                            if not (rules['strong_stop']['grad_norm_range'][0] <= avg_grad_norm <= rules['strong_stop']['grad_norm_range'][1]):
                                strong_stop_counters['grad_norm_sustained'] += rules['check_interval']
                            else:
                                strong_stop_counters['grad_norm_sustained'] = 0

                            if strong_stop_counters['ratio_sustained'] >= rules['strong_stop']['sustained_steps']:
                                logger.error(f"[Auto-stop] STOP: GAN ratio has been below {rules['strong_stop']['ratio_lower_bound']} for {strong_stop_counters['ratio_sustained']} steps. Stopping training.")
                                stop_training_flag = True

                            if strong_stop_counters['grad_norm_sustained'] >= rules['strong_stop']['sustained_steps']:
                                logger.error(f"[Auto-stop] STOP: Grad norm has been outside {rules['strong_stop']['grad_norm_range']} for {strong_stop_counters['grad_norm_sustained']} steps. Stopping training.")
                                stop_training_flag = True

                    adv_abs = loss_G_adv.abs().item()
                    recon_val = loss_G_recon.abs().item() + 1e-9
                    current_ratio = adv_abs / recon_val
                    if current_ratio < target_adv_ratio * 0.5:
                        lambda_recon_run = max(lambda_recon_min, lambda_recon_run * 0.9)
                    elif current_ratio > target_adv_ratio * 1.5:
                        lambda_recon_run = min(lambda_recon_max, lambda_recon_run * 1.1)

                    if grad_norm.item() < 0.8:
                        lambda_gp_run = min(lambda_gp_run * 1.05, config.Gan.Dynamics.LAMBDA_GP_MAX)
                    elif grad_norm.item() > 1.2:
                        lambda_gp_run = max(lambda_gp_run * 0.95, config.Gan.Dynamics.LAMBDA_GP_MIN)
                elif batch_idx % 20 == 0:
                    log_sigma_mean = log_sigma.mean().item() if log_sigma is not None else 0.0
                    logger.info(
                        f"Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx}/{len(train_loader)}] | "
                        f"HETERO | lossG_nll:{loss_G_recon:.3f} | lossG_ece:{loss_G_ece:.4f} | log_sigma_mean:{log_sigma_mean:.4f}"
                    )

            if stop_training_flag:
                logger.warning(f"Epoch [{epoch+1}/{epochs}] | Stopping early due to instability.")
                break

            if decoder_mode == "gan":
                logger.info(
                    f"Epoch [{epoch+1}/{epochs}] | Dynamic weights -> lambda_recon:{lambda_recon_run:.3f}, lambda_gp:{lambda_gp_run:.3f}, lambda_ece:{lambda_ece:.3f}"
                )

            generator.eval()
            val_recon_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(config.Device.DEVICE), batch_y.to(config.Device.DEVICE)
                    if decoder_mode == "cvae":
                        outputs = generator(batch_X)  # prior sampling, no y -- true inference behavior
                        loss = recon_criterion(outputs, batch_y)
                    else:
                        outputs = generator(batch_X)
                        loss = recon_criterion(outputs, batch_y)
                    val_recon_loss += loss.item()
            val_recon_loss /= len(val_loader)

            logger.info(f"Epoch [{epoch+1}/{epochs}] | Model [{model_id}] | Val Recon Loss: {val_recon_loss:.6f}")
            
            # Step LR Schedulers
            scheduler_G.step()
            if scheduler_C is not None:
                scheduler_C.step()

            if val_recon_loss < best_val_recon_loss:
                best_val_recon_loss = val_recon_loss
                try:
                    abs_path = os.path.abspath(model_save_path)
                    logger.info(f"Attempting to save model {model_id} to absolute path: {abs_path}")
                    torch.save(generator, abs_path)
                    logger.info(f"Successfully executed torch.save for model {model_id}. Please verify file timestamp.")
                    
                    # --- 모델 메타데이터 저장 ---
                    metadata_path = os.path.join("models", "model_metadata.json")
                    try:
                        from datetime import datetime as dt
                        feature_columns = list(getattr(config.Data, "FEATURE_COLUMNS", []) or [])
                        if os.path.exists(metadata_path):
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)
                        else:
                            metadata = {"version": "2.0", "models": {}}
                        metadata["version"] = "2.0"
                        
                        metadata["models"][f"model_{model_id}"] = {
                            "created_at": dt.now().isoformat(),
                            "val_loss": round(val_recon_loss, 6),
                            "epochs": epochs,
                            "is_finetune": is_finetuning,
                            "input_dim": int(X_train.shape[-1]),
                            "output_dim": int(y_train.shape[-1]),
                            "future_window_size": int(getattr(config.Data, "FUTURE_WINDOW_SIZE", y_train.shape[-1]) or y_train.shape[-1]),
                            "feature_count": len(feature_columns),
                            "feature_columns": feature_columns,
                            "context_dim": int(getattr(config.Data, "CONTEXT_DIM", 0) or 0),
                            "decoder_mode": str(getattr(config.Gan, "DECODER_MODE", "gan") or "gan"),
                            "pe_mode": str(getattr(config.Gan, "PE_MODE", "cycle_pe") or "cycle_pe"),
                            "strip_context_from_main_input": bool(
                                getattr(config.Gan, "STRIP_CONTEXT_FROM_MAIN_INPUT", False)
                            ),
                            "hyperparams": {
                                "d_model": d_model,
                                "n_layers": n_layers,
                                "n_heads": n_heads,
                                "dropout_p": dropout_p,
                                "cnn_mode": cnn_mode
                            }
                        }
                        
                        with open(metadata_path, 'w') as f:
                            json.dump(metadata, f, indent=2)
                        logger.info(f"Updated model metadata for model_{model_id}")
                    except Exception as meta_e:
                        logger.warning(f"Failed to save model metadata: {meta_e}")
                        
                except Exception as e:
                    logger.error(f"CRITICAL: Failed to save model {model_id} to {abs_path}. Error: {e}", exc_info=True)

        # --- Holdout evaluation (out-of-sample, chronologically last 20% of val) ---
        if len(X_holdout_model) > 0:
            try:
                generator.eval()
                with torch.no_grad():
                    X_holdout_tensor = torch.FloatTensor(X_holdout_model).to(config.Device.DEVICE)
                    y_holdout_tensor = torch.FloatTensor(y_holdout).to(config.Device.DEVICE)
                    holdout_pred = generator(X_holdout_tensor)
                    holdout_loss = recon_criterion(holdout_pred, y_holdout_tensor).item()
                logger.info(
                    f"[HOLDOUT] Model {model_id} | Holdout Loss: {holdout_loss:.6f} "
                    f"| Val Loss (best): {best_val_recon_loss:.6f} "
                    f"| Generalization gap: {holdout_loss - best_val_recon_loss:+.6f}"
                )
            except Exception as holdout_e:
                logger.warning(f"[HOLDOUT] Model {model_id} | Holdout evaluation failed: {holdout_e}")

        # Restore original CNN_MODE after this model finishes (always, even on exception)
        config.Gan.CNN_MODE = original_cnn_mode
        logger.info(f"Finished training model {model_id} ({model_name}). Best validation reconstruction loss: {best_val_recon_loss:.6f}")

    logger.info(f"=== {mode_log} Finished ===")
