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
        # Timezone used for human-facing reports (e.g., Telegram). Cron scheduling should still be set explicitly.
        REPORT_TIMEZONE = os.getenv("AETHER_REPORT_TIMEZONE", "Asia/Seoul")

    # --- Device Configuration ---
    class Device:
        # Automatically select CUDA if available, otherwise fallback to CPU
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Data Preprocessing and Feature Engineering ---
    class Data:
        # Markets to use for building the primary market index
        MARKET_INDEX_COINS = ["KRW-BTC", "KRW-ETH"]
        # Dynamic universe: top N markets by 24h trading value (replaces TRAIN_COINS hardcoding)
        DYNAMIC_UNIVERSE_TOP_N = 100
        DYNAMIC_UNIVERSE_MIN_LISTING_DAYS = 14
        DYNAMIC_UNIVERSE_EXCLUDE = ["USDT", "BUSD", "DAI", "USDC", "UP", "DOWN", "BEAR", "BULL"]
        # Fallback static list used only when DB is unavailable
        TRAIN_COINS_FALLBACK = [
            "KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-SOL", "KRW-DOGE",
            "KRW-ADA", "KRW-AVAX", "KRW-DOT", "KRW-MATIC", "KRW-LINK"
        ]
        # The sequence length (in hours) for the main GAN model's input
        SEQUENCE_LENGTH = 168
        # The future window size (in hours) to be predicted by the main model
        FUTURE_WINDOW_SIZE = 12  # changed from 3h → 12h for medium-term prediction horizon
        # The size of the image representation of the time series
        IMAGE_SIZE = 168
        
        # Rolling beta window candidates (hours) — selected by Optuna walk-forward
        BETA_ROLLING_WINDOW = 168  # default; Optuna tunes over [72, 168, 336]
        # Training currently uses the raw rolling beta. Keep inference aligned by
        # default; enable shrinkage explicitly only after re-training with it.
        BETA_SHRINKAGE_WEIGHT = 0.0
        # BTC regime hysteresis band (fraction of MA)
        REGIME_HYSTERESIS = 0.02
        # BTC regime RV quantile lookback (hours ~180 days)
        REGIME_RV_LOOKBACK = 4320
        COLLECTOR_REQUEST_MAX_RETRIES = 3
        COLLECTOR_REQUEST_BACKOFF_SEC = 1.0
        COLLECTOR_PAGE_SLEEP_SEC = 0.5

        # Feature columns used for model input (ORDER MATTERS!)
        # Phase 1: 32 → 26 features (reduction)
        # Phase 2: 26 → 32 features (cross-market pattern transfer)
        FEATURE_COLUMNS = [
            'rsi', 'macd', 'macdhist', 'adx',                    # technical (4)
            'market_index_return', 'historical_similarity',        # context/PE (2)
            'bb_middle', 'bb_position', 'volume_ma',               # structure (3)
            'volatility_24h', 'volume_volatility',                 # volatility (2)
            'alpha', 'beta',                                       # factor (2)
            'price_position', 'volume_ratio', 'return_skew_24h',   # relative (3)
            'cross_corr_btc',                                      # cross-market (1)
            'factor_size', 'factor_mom', 'factor_vol', 'factor_liq', # factors (4)
            'btc_regime',                                          # regime (1)
            'btc_regime_rv', 'btc_ma_distance',                    # regime detail (2)
            'factor_mom_x_bull', 'factor_liq_x_bear',              # interaction (2)
            'breadth_ratio', 'top_n_return_1h', 'top_n_return_4h', # pattern transfer (3)
            'rank_return_4h', 'net_volume_flow', 'volume_breadth',  # pattern transfer (3)
            'kimchi_premium', 'binance_lead_return_1h', 'global_volume_ratio',  # binance cross-market (3)
        ]
        # Index of market_index_return in FEATURE_COLUMNS (for contextual PE)
        # Context dimensions: market_index_return (idx 4) + historical_similarity (idx 5)
        MARKET_INDEX_FEATURE_IDX = 4  # 0-indexed position
        HISTORICAL_SIMILARITY_FEATURE_IDX = 5  # 0-indexed position (right after market_index)
        CONTEXT_DIM = 2  # Number of context features for Contextual PE
        NON_SCALED_FEATURE_COLUMNS = ['btc_regime']

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
        MIN_SIGNAL_RETURN = 0.002  # 0.2% - scaled for 12h horizon (2x of 3h baseline)
        # Execution cost assumptions.
        TRADE_FEE_PER_LEG = float(os.getenv("AETHER_TRADE_FEE_PER_LEG", "0.0005"))
        SLIPPAGE_PER_LEG = float(os.getenv("AETHER_SLIPPAGE_PER_LEG", "0.0003"))
        # Live intraday long is treated as a spot entry signal, so Step 1 can
        # budget entry-side cost only. Realized backtest PnL still subtracts full
        # round-trip cost for realism.
        LIVE_INTRADAY_LONG_STEP1_COST_LEGS = float(os.getenv("AETHER_LIVE_INTRADAY_LONG_STEP1_COST_LEGS", "1.0"))
        # Probabilistic downside guard for Step 1.
        # The guard is applied as a soft downside penalty instead of a hard veto so
        # the acceptance score can trade off expected edge vs. downside risk more generally.
        PI_LOW_80_MIN_LIVE = float(os.getenv("AETHER_PI_LOW_80_MIN_LIVE", "0.0"))
        PI_LOW_80_MIN_INTRADAY = float(os.getenv("AETHER_PI_LOW_80_MIN_INTRADAY", "-0.008"))  # scaled 2x for 12h horizon
        # Step 1 downside-risk penalty multiplier:
        # step1_score = net_alpha - penalty * max(0, pi_guard_floor - directional_pi_guard)
        STEP1_PI_GUARD_PENALTY = float(os.getenv("AETHER_STEP1_PI_GUARD_PENALTY", "0.35"))
        # Live intraday longs need a lighter downside penalty because they are
        # entry-side only in spot execution and already carry tighter cost budgets.
        STEP1_PI_GUARD_PENALTY_INTRADAY_LONG = float(
            os.getenv("AETHER_STEP1_PI_GUARD_PENALTY_INTRADAY_LONG", "0.10")
        )
        # Base uncertainty score threshold (mean predictive std of 12h returns).
        # Typical range: 0.005-0.05. Dynamic threshold adapts from batch quantile.
        UNCERTAINTY_THRESHOLD = 0.03
        # Use current-batch uncertainty distribution to adapt threshold robustly.
        ENABLE_DYNAMIC_UNCERTAINTY_THRESHOLD = True
        # Keep roughly this quantile of lower-uncertainty candidates before other filters.
        DYNAMIC_UNCERTAINTY_QUANTILE = 0.65
        # Clamp adaptive threshold to avoid overreaction to outliers.
        DYNAMIC_UNCERTAINTY_MIN_MULTIPLIER = 0.8
        DYNAMIC_UNCERTAINTY_MAX_MULTIPLIER = 4.0
        # Multiplier for the uncertainty threshold for counter-trend trades (e.g., 0.7 makes it 30% stricter)
        COUNTER_TREND_UNCERTAINTY_MULTIPLIER = 0.7
        # Minimum ensemble agreement required to keep a candidate.
        MIN_CONSENSUS_SCORE = 0.6
        # Intraday can tolerate slightly lower agreement because Step 1 already
        # enforces positive net alpha and a probabilistic downside floor.
        MIN_CONSENSUS_SCORE_INTRADAY = float(os.getenv("AETHER_MIN_CONSENSUS_SCORE_INTRADAY", "0.55"))
        # Counter-trend trades require stronger model agreement.
        COUNTER_TREND_MIN_CONSENSUS_SCORE = 0.8
        COUNTER_TREND_MIN_CONSENSUS_SCORE_INTRADAY = float(
            os.getenv("AETHER_COUNTER_TREND_MIN_CONSENSUS_SCORE_INTRADAY", "0.6")
        )
        # Telegram alert volume control.
        # Intraday runs every 4h, but Telegram should remain a sparse execution channel.
        TELEGRAM_ALERT_CAP_INTRADAY_PER_DAY = int(
            os.getenv("AETHER_TELEGRAM_ALERT_CAP_INTRADAY_PER_DAY", "2")
        )
        TELEGRAM_ALERT_CAP_MORNING_PER_DAY = int(
            os.getenv("AETHER_TELEGRAM_ALERT_CAP_MORNING_PER_DAY", "1")
        )
        # Prevent CV-based uncertainty blow-ups when mean predicted return is near-zero.
        # (returns are per-step returns, e.g. 0.002 = 0.2%)
        UNCERTAINTY_CV_DENOM_FLOOR = 0.002
        # Live execution requirement: ensure at least N recommendations are produced each run.
        MIN_RECOMMENDATIONS_LIVE = 1
        # Repo-root live ops target KRW spot execution.
        # Short signals may still be emitted as directional warnings, but they are
        # watch-only by default unless this override is enabled explicitly.
        LIVE_ALLOW_SHORT_EXECUTION = os.getenv("AETHER_LIVE_ALLOW_SHORT_EXECUTION", "0").strip().lower() in (
            "1", "true", "yes", "y"
        )
        # MinRec behavior:
        # - "trade": force a minimal-risk trade candidate if possible (still can degrade to watch if impossible)
        # - "watch": never force a trade; output a watch-only item when no recs survive
        MIN_REC_MODE = os.getenv("AETHER_MIN_REC_MODE", "trade").strip().lower()
        # Do not force neutral/no-signal items.
        MIN_REC_DISALLOW_NEUTRAL = True
        # Minimum absolute expected return for forced trade candidates.
        MIN_REC_MIN_ABS_EXPECTED_RETURN = float(os.getenv("AETHER_MIN_REC_MIN_ABS_EXPECTED_RETURN", "0.002"))
        # Minimum consensus for forced trade candidates (relaxes ensemble noise).
        MIN_REC_MIN_CONSENSUS_SCORE = float(os.getenv("AETHER_MIN_REC_MIN_CONSENSUS_SCORE", "0.5"))
        # Maximum uncertainty multiple for forced trade candidates. (relative to base UNCERTAINTY_THRESHOLD)
        MIN_REC_MAX_UNCERTAINTY_MULTIPLIER = float(os.getenv("AETHER_MIN_REC_MAX_UNCERTAINTY_MULTIPLIER", "6.0"))
        # If no forced trade candidate is safe enough, produce a watch-only item instead of nothing.
        MIN_REC_ALLOW_WATCH_ONLY_FALLBACK = True
        # Keep liquidity as a hard constraint even in min-rec fallback.
        MIN_REC_FALLBACK_ALLOW_LOW_LIQUIDITY = False
        # Consider data stale if last candle is older than this many hours (screener fallback).
        SCREENER_MAX_STALENESS_HOURS_LIVE = 24
        # Hard gate for scheduled inference runs (intraday/morning): if DB is too stale, abort and alert.
        # These can be overridden via env vars to match your collection cadence.
        DATA_FRESHNESS_MAX_LAG_HOURS_INTRADAY = float(os.getenv("AETHER_FRESHNESS_MAX_LAG_HOURS_INTRADAY", "6"))
        DATA_FRESHNESS_MAX_LAG_HOURS_MORNING = float(os.getenv("AETHER_FRESHNESS_MAX_LAG_HOURS_MORNING", "12"))
        FAIL_ON_STALE_DATA_LIVE = True
        STALE_DATA_ALERT_SEND_TELEGRAM = True
        # Ops safety: force all outputs to be "watch-only" regardless of signal quality.
        # Intended for stale/offline override paths in scheduled automation.
        RUNTIME_WATCH_ONLY = os.getenv("AETHER_RUNTIME_WATCH_ONLY", "0").strip() in ("1", "true", "yes", "y")

        # If only a single candidate survives to uncertainty stage, allow it to pass even if above threshold.
        # This reduces excessive MinRec forcing when the pipeline already narrowed to 1 liquid candidate.
        ALLOW_SINGLE_CANDIDATE_UNCERTAINTY_BYPASS = True
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
        # Safety switch: by default, do NOT force-fill recommendations that failed filters.
        FORCED_TOPK_ENABLED = False
        # Optional backtest override (kept False for strict realism).
        FORCED_TOPK_BACKTEST_ENABLED = False
        # Even when forced fallback is enabled, never revive these failed conditions.
        FORCED_TOPK_EXCLUDE_FAILED_REASONS = (
            "High Uncertainty",
            "Low Liquidity",
        )

    # --- GAN Model & Training Hyperparameters ---
    class Gan:
        # Model Architecture
        D_MODEL = 128
        N_HEADS = 8
        N_LAYERS = 3
        DROPOUT_P = 0.1
        NOISE_DIM = 32
        CNN_MODE = '1D'
        USE_CAUSAL_MASK = False  # encoder sees full sequence; attention-guided CNN needs unmasked attn
        DECODER_MODE = 'cvae'
        PE_MODE = 'cycle_pe_full'
        STRIP_CONTEXT_FROM_MAIN_INPUT = True
        CVAE_KL_BETA = 0.5  # Target β after annealing warmup

        # General Training
        EPOCHS = 100
        BATCH_SIZE = 64 # Optimized for RTX 3070 (8GB VRAM)
        MAX_BATCH_SIZE_CPU = 32  # CPU-only fallback cap for stability
        TRAIN_SPLIT = 0.9
        N_ENSEMBLE_MODELS = 4

        # Learning Rates
        LEARNING_RATE_G = 0.0001  # Generator
        LEARNING_RATE_C = 0.0002  # Critic (TTUR)
        # Weight for the Expected Calibration Error loss term
        LAMBDA_ECE = 0.1
        # Weight for directional accuracy loss (penalizes wrong direction predictions)
        LAMBDA_DIRECTION = 0.3

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
