import torch
import os

class Config:
    """
    Central configuration class for the Chrono-Trader project.
    Utilizes nested classes to group related parameters for better organization.
    """
    PATTERN_LOOKBACK_HOURS = 24

    # --- General Application Settings ---
    class General:
        APP_NAME = "AETHER: Quant AI"
        DB_PATH = os.path.join("data", "crypto_data.db")
        LOG_DIR = "logs"
        # Path to the main GAN model for pattern prediction
        MODEL_PATH = "models/model_1.pth"
        MODEL_PATH_SHORT = "models/model_short.pth"
        REC_TAG = "standard"  # Tag for recommendation files (standard/short)

    # --- Device Configuration ---
    class Device:
        # Automatically select CUDA if available, otherwise fallback to CPU
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Data Preprocessing and Feature Engineering ---
    class Data:
        # Markets to use for building the primary market index
        MARKET_INDEX_COINS = ["KRW-BTC", "KRW-ETH"]
        # The sequence length (in hours) for the main GAN model's input
        SEQUENCE_LENGTH = 168
        # The future window size (in hours) to be predicted by the main model
        FUTURE_WINDOW_SIZE = 6
        # The size of the image representation of the time series
        IMAGE_SIZE = 168
        
        # Feature columns used for model input (ORDER MATTERS!)
        FEATURE_COLUMNS = [
            'close', 'volume', 'rsi', 'macd', 'macdsignal', 'macdhist', 'adx', 'obv',
            'market_index_return', 'historical_similarity',  # Context features (indices 8, 9)
            'bb_upper', 'bb_middle', 'bb_lower', 'volume_ma',
            'volatility_24h', 'volatility_7d', 'volume_volatility',
            'alpha', 'beta'
        ]
        # Index of market_index_return in FEATURE_COLUMNS (for contextual PE)
        # Context dimensions: market_index_return (idx 8) + historical_similarity (idx 9)
        MARKET_INDEX_FEATURE_IDX = 8  # 0-indexed position
        HISTORICAL_SIMILARITY_FEATURE_IDX = 9  # 0-indexed position (right after market_index)
        CONTEXT_DIM = 2  # Number of context features for Contextual PE

    # --- Pattern Library Generation ---
    class Pattern:
        # Length of patterns (in hours) to be clustered
        LENGTH = 24
        # Number of representative patterns to generate
        N_CLUSTERS = 50

    # --- Recommendation Engine ---
    class Recommender:
        # How many past hours of trading value to consider for liquidity checks
        LIQUIDITY_LOOKBACK_HOURS = 24
        # Mode-specific liquidity/volume thresholds (in KRW)
        LIQUIDITY_THRESHOLDS = {
            'live': 1_000_000_000,      # 10억원 for live screening
            'backtest': 50_000_000       # 5천만원 for backtest screening
        }
        # Minimum absolute expected return to consider a signal valid
        MIN_SIGNAL_RETURN = 0.001  # 0.1% - lowered to match model's conservative predictions
        # Base uncertainty score threshold for accepting a trade
        UNCERTAINTY_THRESHOLD = 7.5
        # Multiplier for the uncertainty threshold for counter-trend trades (e.g., 0.7 makes it 30% stricter)
        COUNTER_TREND_UNCERTAINTY_MULTIPLIER = 0.7
        # Short and long window SMA periods for determining market regime
        REGIME_SMA_SHORT = 20
        REGIME_SMA_LONG = 60
        # Confidence score boost for trades that align with the market trend
        TREND_CONFIDENCE_BOOST = 1.05
        # Maximum DTW distance to consider a pattern similar to historical success patterns
        DTW_THRESHOLD = 1.5
        # Minimum return for a pattern to be considered a "success pattern"
        SUCCESS_PATTERN_MIN_RETURN = 0.15
        # Number of Monte Carlo inferences for uncertainty estimation
        MC_N_INFERENCES = 20
        # Maximum number of concurrent positions to hold
        MAX_POSITIONS = 5
        # Kelly fraction for position sizing
        KELLY_FRACTION = 0.2
        # Minimum probability for a pump signal to be considered significant
        PUMP_PROBABILITY_THRESHOLD = 0.2

    # --- GAN Model & Training Hyperparameters ---
    class Gan:
        # Model Architecture
        D_MODEL = 128
        N_HEADS = 8
        N_LAYERS = 3
        DROPOUT_P = 0.1
        NOISE_DIM = 32
        CNN_MODE = '1D'

        # General Training
        EPOCHS = 100
        BATCH_SIZE = 64 # Optimized for RTX 3070 (8GB VRAM)
        TRAIN_SPLIT = 0.9
        N_ENSEMBLE_MODELS = 3

        # Learning Rates
        LEARNING_RATE_G = 0.0001  # Generator
        LEARNING_RATE_C = 0.0002  # Critic (TTUR)
        # Weight for the Expected Calibration Error loss term
        LAMBDA_ECE = 0.1

        # Loss Weights & Dynamics (for WGAN-GP)
        LAMBDA_GP = 10              # Gradient penalty lambda
        LAMBDA_RECON = 100          # Reconstruction loss weight
        CRITIC_BASE_ITERS = 7

        # --- Sub-group for advanced dynamic adjustments ---
        class Dynamics:
            TARGET_ADV_LOSS = -0.1
            TARGET_ADV_LOSS_RANGE = [-0.3, -0.05]
            GAN_WARMUP_STEPS = 500
            TARGET_ADV_RATIO = 1.0
            LAMBDA_RECON_MIN = 1.0
            LAMBDA_RECON_MAX = 100.0
            CRITIC_MAX_ITERS = 10
            CRITIC_MIN_ITERS = 5
            LAMBDA_GP_MIN = 1.0
            LAMBDA_GP_MAX = 10.0

        # --- Sub-group for Optuna-specific initial values ---
        class Optuna:
            PENALTY_SCALING_FACTOR = 0.2
            LAMBDA_RECON_INITIAL = 100
            LAMBDA_GP_INITIAL = 10
            CRITIC_BASE_ITERS_INITIAL = 7

        # --- Sub-group for auto-stopping rules ---
        class AutoStop:
            GAN_STOP_RULES = {
                'warmup_steps': 500, 'check_interval': 100, 'moving_avg_window': 50,
                'warnings': {'grad_norm_range': [0.8, 1.2], 'ratio_range': [-5.0, 15.0]},
                'strong_stop': {'grad_norm_range': [0.5, 1.5], 'ratio_lower_bound': -10.0, 'sustained_steps': 200}
            }

config = Config()