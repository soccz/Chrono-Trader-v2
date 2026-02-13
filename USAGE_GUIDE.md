# Chrono-Trader Usage Guide

This guide provides detailed instructions on how to use the different modes of the Chrono-Trader application.

## ⏱ Scheduled Ops (`intraday`, `morning-report`)

Production scheduling should use the ops entrypoint `scripts/run_scheduled.py` (it runs `refresh-db` then inference, enforces watchdogs/freshness, and guarantees >=1 output per run via MinRec/watch-only fallback).

Recommended on-host scheduler:
- systemd (user timers): see `deploy/systemd/README.md`

Manual run examples:
```bash
# Refresh recent DB candles (network required unless DB is already fresh)
python main.py --mode refresh-db --refresh_days 3 --offline_ok

# Intraday scheduled inference (best run via scripts/run_scheduled.py)
python main.py --mode intraday --min_k 1 --limit 8

# Morning report scheduled inference
python main.py --mode morning-report --min_k 1 --limit 8
```

Operational contract (exit codes/artifacts/fallback behavior):
- `OPS_ACCEPTANCE.md`

## 🚀 Daily Prediction (`daily`)

This is the primary mode for daily operation. It performs a lightweight fine-tuning on the latest data and generates new predictions.

**Purpose:**
-   Quickly update the model with the most recent market trends.
-   Generate new trade recommendations and potential pump predictions.
-   Designed to be run automatically every day.

**Command:**
```bash
python main.py --mode daily
```

**Options:**
-   `--daily_epochs <N>`: Specify the number of epochs for the light fine-tuning. Default is `2`. A small number is recommended to keep the process fast and prevent overfitting to short-term noise.

**Example:**
```bash
# Run daily mode with 3 epochs for fine-tuning
python main.py --mode daily --daily_epochs 3
```

---

## 🏋️‍♂️ Full Model Training (`train`)

This mode is for performing a deep, extensive training of the models. It should be run periodically (e.g., weekly or monthly) or when you believe the market dynamics have significantly changed.

**Purpose:**
-   Train the models from the ground up or perform a heavy re-training on a large dataset.
-   Find optimal hyperparameters using Optuna (optional).
-   This is a time-consuming process.

**Command:**
```bash
python main.py --mode train
```

**Options:**
-   `--epochs <N>`: Override training epochs (defaults to `config.Gan.EPOCHS`).
-   `--tune`: Run Optuna hyperparameter tuning before the main training begins.
-   `--no_collect`: Skip network data collection and train from the existing DB only (recommended for offline/stability).
-   `--offline_ok`: Continue despite collection failures (used mainly for ops/offline environments).

**Optuna (advanced) via env vars:**
- `AETHER_OPTUNA_TRIALS`: number of trials (default: `50`)
- `AETHER_OPTUNA_STORAGE`: Optuna storage URL (recommended: `sqlite:///analysis/optuna_full.db`)
- `AETHER_OPTUNA_STUDY_NAME`: study name (default: `aether_gan_tune`)
- `AETHER_OPTUNA_LOAD_IF_EXISTS=1`: resume an existing study
- `AETHER_OPTUNA_TIMEOUT_SEC`: optional wall-clock timeout
- `AETHER_OPTUNA_N_JOBS`: parallelism (recommend `1` on CPU)

**Example:**
```bash
# Run a full training for 30 epochs
python main.py --mode train --epochs 30

# Run hyperparameter tuning followed by training with the best parameters
python main.py --mode train --tune --no_collect --offline_ok
```

---

## 📊 Backtesting (`backtest`)

This mode allows you to evaluate the model's performance on historical data.

**Purpose:**
-   Simulate trading over a past period to gauge the model's effectiveness.
-   Calculate key performance metrics like Win Rate, Sharpe Ratio, and Max Drawdown.

**Command:**
```bash
python main.py --mode backtest --days <N>
```

**Options:**
-   `--days <N>`: **(Required)** The number of past days to run the backtest over.

**Example:**
```bash
# Run a backtest over the last 30 days
python main.py --mode backtest --days 30
```

**Faster evaluation / automation:**
```bash
# Backtest suite writes JSON artifacts into analysis/
python scripts/eval_suite.py --days 30 --stride_hours 4 --tag eval_30d_4h --no_telegram
```

---

## 🛠️ Other Utility Modes

-   **`init_db`**: Initializes the database schema. Run this once when setting up the project for the first time.
    ```bash
    python main.py --mode init_db
    ```
-   **`collect-all`**: Collects historical data for all markets defined in the configuration.
    ```bash
    python main.py --mode collect-all --days 90
    ```
-   **`train-pump`**: Specifically trains the pump prediction model.
    ```bash
    python main.py --mode train-pump
    ```
-   **`find-pumps`**: Runs only the pump prediction logic without any training.
    ```bash
    python main.py --mode find-pumps
    ```
