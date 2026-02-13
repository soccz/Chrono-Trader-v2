import torch
import torch.optim as optim
import torch.nn as nn
import torch.autograd as autograd
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import train_test_split
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

from utils.config import config
from utils.logger import logger
from data.preprocessor import get_processed_data_for_training, get_market_index
from models.hybrid_model import build_model
from models.critic import build_critic

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
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_correlations = []
    
    logger.info(f"Trial {trial.number}: Starting {n_splits}-Fold TimeSeries Cross-Validation...")

    for fold, (train_index, val_index) in enumerate(tscv.split(X)):
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
            generator = build_model(d_model=d_model, n_heads=n_heads, n_layers=n_layers, input_dim=X.shape[-1], noise_dim=config.Gan.NOISE_DIM, output_dim=y.shape[-1], dropout_p=dropout_p)
            critic = build_critic()
            optimizer_G = optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.9))
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
                critic.train()
                for batch_idx, (real_sequences, real_paths) in enumerate(train_loader):
                    real_sequences, real_paths = real_sequences.to(config.Device.DEVICE), real_paths.to(config.Device.DEVICE)
                    if real_sequences.size(0) <= 1: continue

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

                    optimizer_G.zero_grad()
                    gen_paths = generator(real_sequences)
                    loss_G_adv = -critic(gen_paths).mean()
                    loss_G_recon = recon_criterion(gen_paths, real_paths)
                    pred_return = gen_paths.sum(dim=1)
                    real_return = real_paths.sum(dim=1)
                    pred_prob = torch.sigmoid(pred_return)
                    target_prob = (real_return > 0).float()
                    loss_G_ece = ((pred_prob - target_prob) ** 2).mean()

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

                    loss_G = loss_G_adv + lambda_recon_run * loss_G_recon_weighted + lambda_ece * loss_G_ece + lambda_direction * loss_G_direction
                    loss_G.backward()
                    torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                    optimizer_G.step()

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

            generator.train() # Enable dropout for MC-Dropout
            for module in generator.modules():
                if isinstance(module, torch.nn.BatchNorm1d):
                    module.eval()

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
        X, y, _ = get_processed_data_for_training(config.Data.MARKET_INDEX_COINS[0], market_index_df)
        if X is None: return

        # IMPORTANT: We are now MAXIMIZING the correlation
        study = optuna.create_study(direction='maximize') 
        objective_with_data = functools.partial(objective, X=X, y=y)
        study.optimize(objective_with_data, n_trials=50)

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
        X_list, y_list = [], []
        for market in markets:
            X_market, y_market, _ = get_processed_data_for_training(market, market_index_df)
            if X_market is not None: X_list.append(X_market); y_list.append(y_market)
        if not X_list: logger.error("No data for fine-tuning."); return
        X, y = np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)
        lr = config.Gan.LEARNING_RATE_G / 10
    else: # Full Training on all target markets
        full_train_markets = getattr(config.Data, "TRAIN_COINS", config.Data.MARKET_INDEX_COINS)
        logger.info(f"Starting full training data assembly for markets: {full_train_markets}")
        X_list, y_list = [], []
        for market in full_train_markets:
            X_market, y_market, _ = get_processed_data_for_training(market, market_index_df)
            if X_market is not None:
                X_list.append(X_market)
                y_list.append(y_market)
        
        if not X_list:
            logger.error("Training failed: No data could be assembled for any target market.")
            return

        X, y = np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)
        logger.info(f"Assembled full training dataset with shape X: {X.shape}, y: {y.shape}")
        lr = config.Gan.LEARNING_RATE_G

    # If epochs is not passed as an argument, use config defaults
    if epochs is None:
        epochs = config.Gan.EPOCHS

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
        
        logger.info(f"\n--- Training Ensemble Model {model_id}/{len(model_configs)}: {model_name} ---")
        logger.info(f"    d_model={d_model}, n_layers={n_layers}, n_heads={n_heads}, dropout={dropout_p}, cnn={cnn_mode}, bagging={bagging_ratio}")
        
        # Temporarily override CNN_MODE for this model
        original_cnn_mode = config.Gan.CNN_MODE
        config.Gan.CNN_MODE = cnn_mode
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=1 - config.Gan.TRAIN_SPLIT, random_state=42 + model_id)
        
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
                expected_input_dim = X.shape[-1]
                if loaded_input_dim is not None and loaded_input_dim != expected_input_dim:
                    raise ValueError(
                        f"Checkpoint input_dim mismatch (loaded={loaded_input_dim}, expected={expected_input_dim})"
                    )
                logger.info(f"Loaded model {model_id} for fine-tuning.")
            except Exception as e:
                logger.error(f"Could not load model {model_id}: {e}. Rebuilding.")
                generator = build_model(d_model=d_model, n_heads=n_heads, n_layers=n_layers, input_dim=X.shape[-1], noise_dim=config.Gan.NOISE_DIM, output_dim=y.shape[-1], dropout_p=dropout_p)
        else:
            generator = build_model(d_model=d_model, n_heads=n_heads, n_layers=n_layers, input_dim=X.shape[-1], noise_dim=config.Gan.NOISE_DIM, output_dim=y.shape[-1], dropout_p=dropout_p)
        
        critic = build_critic()
        # Use separate learning rates for Generator and Critic (TTUR)
        lr_g = lr # The 'lr' variable is now correctly set to LEARNING_RATE_G
        lr_c = config.Gan.LEARNING_RATE_C if not is_finetuning else config.Gan.LEARNING_RATE_C / 10
        optimizer_G = optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.9))
        optimizer_C = optim.Adam(critic.parameters(), lr=lr_c, betas=(0.5, 0.9))
        
        # Learning Rate Schedulers (Cosine Annealing for smooth decay)
        scheduler_G = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=epochs, eta_min=lr_g * 0.01)
        scheduler_C = optim.lr_scheduler.CosineAnnealingLR(optimizer_C, T_max=epochs, eta_min=lr_c * 0.01)
        
        logger.info(f"Optimizers created. Generator LR: {lr_g:.6f}, Critic LR: {lr_c:.6f}")
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
            critic.train()
            for batch_idx, (real_sequences, real_paths) in enumerate(train_loader):
                real_sequences, real_paths = real_sequences.to(config.Device.DEVICE), real_paths.to(config.Device.DEVICE)
                if real_sequences.size(0) <= 1: continue

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
                optimizer_G.zero_grad()
                gen_paths = generator(real_sequences)
                loss_G_adv = -critic(gen_paths).mean()
                loss_G_recon = recon_criterion(gen_paths, real_paths)

                pred_return = gen_paths.sum(dim=1)
                real_return = real_paths.sum(dim=1)
                pred_prob = torch.sigmoid(pred_return)
                target_prob = (real_return > 0).float()
                loss_G_ece = ((pred_prob - target_prob) ** 2).mean()

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
                
                loss_G = (
                    loss_G_adv
                    + lambda_recon_run * loss_G_recon_weighted
                    + lambda_ece * loss_G_ece
                    + loss_G_balance
                    + loss_G_gate_reg * 0.5
                    + lambda_direction * loss_G_direction
                )
                loss_G.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                optimizer_G.step()

                total_steps += 1

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

                        # Check Warning Conditions
                        if not (rules['warnings']['ratio_range'][0] <= avg_ratio <= rules['warnings']['ratio_range'][1]):
                            logger.warning(f"[Auto-stop] Warning: Moving avg of ratio ({avg_ratio:.3f}) is outside the warning range {rules['warnings']['ratio_range']}.")
                        if not (rules['warnings']['grad_norm_range'][0] <= avg_grad_norm <= rules['warnings']['grad_norm_range'][1]):
                            logger.warning(f"[Auto-stop] Warning: Moving avg of grad_norm ({avg_grad_norm:.3f}) is outside the warning range {rules['warnings']['grad_norm_range']}.")

                        # Check Strong Stop Conditions
                        if avg_ratio < rules['strong_stop']['ratio_lower_bound']:
                            strong_stop_counters['ratio_sustained'] += rules['check_interval']
                        else:
                            strong_stop_counters['ratio_sustained'] = 0
                        
                        if not (rules['strong_stop']['grad_norm_range'][0] <= avg_grad_norm <= rules['strong_stop']['grad_norm_range'][1]):
                            strong_stop_counters['grad_norm_sustained'] += rules['check_interval']
                        else:
                            strong_stop_counters['grad_norm_sustained'] = 0

                        # Trigger Stop if Necessary
                        if strong_stop_counters['ratio_sustained'] >= rules['strong_stop']['sustained_steps']:
                            logger.error(f"[Auto-stop] STOP: GAN ratio has been below {rules['strong_stop']['ratio_lower_bound']} for {strong_stop_counters['ratio_sustained']} steps. Stopping training.")
                            stop_training_flag = True
                        
                        if strong_stop_counters['grad_norm_sustained'] >= rules['strong_stop']['sustained_steps']:
                            logger.error(f"[Auto-stop] STOP: Grad norm has been outside {rules['strong_stop']['grad_norm_range']} for {strong_stop_counters['grad_norm_sustained']} steps. Stopping training.")
                            stop_training_flag = True

                # Dynamic balancing for reconstruction weight
                adv_abs = loss_G_adv.abs().item()
                recon_val = loss_G_recon.abs().item() + 1e-9
                current_ratio = adv_abs / recon_val
                if current_ratio < target_adv_ratio * 0.5:
                    lambda_recon_run = max(lambda_recon_min, lambda_recon_run * 0.9)
                elif current_ratio > target_adv_ratio * 1.5:
                    lambda_recon_run = min(lambda_recon_max, lambda_recon_run * 1.1)

                # Light gradient penalty adjustment based on critic grad norm
                if grad_norm.item() < 0.8:
                    lambda_gp_run = min(lambda_gp_run * 1.05, config.Gan.Dynamics.LAMBDA_GP_MAX)
                elif grad_norm.item() > 1.2:
                    lambda_gp_run = max(lambda_gp_run * 0.95, config.Gan.Dynamics.LAMBDA_GP_MIN)

            if stop_training_flag:
                logger.warning(f"Epoch [{epoch+1}/{epochs}] | Stopping early due to instability.")
                break

            logger.info(
                f"Epoch [{epoch+1}/{epochs}] | Dynamic weights -> lambda_recon:{lambda_recon_run:.3f}, lambda_gp:{lambda_gp_run:.3f}, lambda_ece:{lambda_ece:.3f}"
            )

            generator.eval()
            val_recon_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(config.Device.DEVICE), batch_y.to(config.Device.DEVICE)
                    outputs = generator(batch_X)
                    loss = recon_criterion(outputs, batch_y)
                    val_recon_loss += loss.item()
            val_recon_loss /= len(val_loader)

            logger.info(f"Epoch [{epoch+1}/{epochs}] | Model [{model_id}] | Val Recon Loss: {val_recon_loss:.6f}")
            
            # Step LR Schedulers
            scheduler_G.step()
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
                        if os.path.exists(metadata_path):
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)
                        else:
                            metadata = {"version": "1.0", "models": {}}
                        
                        metadata["models"][f"model_{model_id}"] = {
                            "created_at": dt.now().isoformat(),
                            "val_loss": round(val_recon_loss, 6),
                            "epochs": epochs,
                            "is_finetune": is_finetuning,
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

        # Restore original CNN_MODE after this model finishes
        config.Gan.CNN_MODE = original_cnn_mode
        logger.info(f"Finished training model {model_id} ({model_name}). Best validation reconstruction loss: {best_val_recon_loss:.6f}")

    logger.info(f"=== {mode_log} Finished ===")
