import torch
import os

class Config:
    # --- General ---
    APP_NAME = "Chrono-Trader"
    DB_PATH = os.path.join("data", "crypto_data.db")
    LOG_DIR = "logs"

    # --- Device ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Data ---
    TARGET_MARKETS = ["KRW-BTC", "KRW-ETH"]
    SEQUENCE_LENGTH = 168
    IMAGE_SIZE = 168

    # --- Model ---
    MODEL_PATH = "models/model_1.pth"
    D_MODEL = 128
    N_HEADS = 8
    N_LAYERS = 3
    DROPOUT_P = 0.1
    GAN_NOISE_DIM = 32
    CNN_MODE = '1D'

    # --- GAN Training Specifics ---
    GAN_TARGET_ADV_LOSS = -0.1 # Target for generator adversarial loss (adjusted for WGAN-GP)
    GAN_TARGET_ADV_LOSS_RANGE = [-0.3, -0.05] # Acceptable range for lossG_adv
    PENALTY_SCALING_FACTOR = 0.2 # Factor to scale the stability penalty in Optuna trials
    GAN_WARMUP_STEPS = 500      # Steps before adaptive training kicks in
    LAMBDA_GP = 10              # Gradient penalty lambda
    LAMBDA_RECON = 100          # Weight of reconstruction loss in generator
    TARGET_ADV_RATIO = 1.0      # Desired |adv| : recon ratio for generator loss balancing
    LAMBDA_RECON_MIN = 1.0
    LAMBDA_RECON_MAX = 100.0
    CRITIC_BASE_ITERS = 7
    CRITIC_MAX_ITERS = 10
    CRITIC_MIN_ITERS = 5
    CRITIC_LEARNING_RATE_MULTIPLIER = 1.0 # Multiplier for critic's learning rate relative to generator's
    LAMBDA_ECE = 0.1            # Weight for calibration loss term
    LAMBDA_RECON_INITIAL = 100  # Initial weight of reconstruction loss for Optuna
    LAMBDA_GP_INITIAL = 10      # Initial gradient penalty lambda for Optuna
    CRITIC_BASE_ITERS_INITIAL = 7 # Initial critic iterations for Optuna
    LAMBDA_GP_MIN = 1.0         # Minimum lambda_gp for dynamic adjustment
    LAMBDA_GP_MAX = 10.0        # Maximum lambda_gp for dynamic adjustment

    # --- Auto-stopping Rules for GAN Training ---
    GAN_STOP_RULES = {
        'warmup_steps': 500,
        'check_interval': 100, # Check every 100 steps
        'moving_avg_window': 50,
        'warnings': {
            'grad_norm_range': [0.8, 1.2],
            'ratio_range': [-5.0, 15.0]
        },
        'strong_stop': {
            'grad_norm_range': [0.5, 1.5],
            'ratio_lower_bound': -10.0,
            'sustained_steps': 200 # Stop if condition holds for this many steps
        }
    }

    # --- Training ---
    EPOCHS = 100
    BATCH_SIZE = 32
    LEARNING_RATE_G = 0.0001  # Learning rate for the generator
    LEARNING_RATE_C = 0.0002  # Learning rate for the critic (TTUR)
    TRAIN_SPLIT = 0.9
    N_ENSEMBLE_MODELS = 3

    # --- Inference & Trading ---
    PATTERN_LOOKBACK_HOURS = 24 # Hours for pattern matching in daily mode
    MC_N_INFERENCES = 20 # Number of inferences for MC Dropout
    MAX_POSITIONS = 5
    KELLY_FRACTION = 0.2
    UNCERTAINTY_THRESHOLD = 7.5 # Max uncertainty score to allow a trade
    # Mode-specific liquidity/volume thresholds
    LIQUIDITY_THRESHOLDS = {
        'live': 1_000_000_000,      # 10억원 for live screening
        'backtest': 50_000_000       # 5천만원 for backtest screening (lowered)
    }
    DTW_THRESHOLD = 1.5 # Max DTW distance to consider a pattern similar
    MIN_SIGNAL_RETURN = 0.02 # Minimum compounded return magnitude to issue a trade signal

config = Config()
