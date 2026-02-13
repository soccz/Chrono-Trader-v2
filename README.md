# AETHER (Chrono-Trader)

Explainable, production-minded crypto forecasting and recommendation system:
- Feature engineering for market regime + cross-sectional factors
- Hybrid model: Transformer (global) + CNN (local) fused by an explainable gate
- Probabilistic forecasting via GAN-style generator + uncertainty estimation
- Scheduled ops that never hang, enforce freshness gates, degrade safely, and always emit at least one output item per run

This repository is both a research sandbox and an ops-ready pipeline.

Not financial advice.

## Docs

- Architecture: `PROJECT_ARCHITECTURE.md`
- Ops contract (exit codes, artifacts, freshness, MinRec): `OPS_ACCEPTANCE.md`
- Evaluation protocol: `EVAL_PROTOCOL.md`
- systemd scheduling: `deploy/systemd/README.md`
- Usage (EN): `USAGE_GUIDE.md`
- Usage (KR): `사용가이드.md`

## Architecture (High Level)

### End-to-End Pipeline

```mermaid
flowchart LR
  Upbit[(Upbit API)] --> Collector[data/collector.py]
  Collector --> DB[(SQLite: data/crypto_data.db)]

  DB --> Preprocess[data/preprocessor.py]
  Preprocess --> Predictor[inference/predictor.py]
  Predictor --> Recommender[inference/recommender.py]
  Recommender --> CSV[recommendations/*.csv]
  Recommender --> Metrics[analysis/run_markets_metrics_*.jsonl]

  Scheduler[scripts/run_scheduled.py] --> Refresh[main.py --mode refresh-db]
  Scheduler --> Infer[main.py --mode intraday|morning-report]
  Metrics --> Health[scripts/ops_healthcheck.py]
  Health --> Alert[Telegram (optional)]
```

### Model (Hybrid + Explainable Gate)

```mermaid
flowchart TB
  X[168h sequence x 27 features] --> T[Transformer Encoder]
  X --> C[CNN stack (local patterns)]
  T --> F[Explainable Gated Fusion]
  C --> F
  F --> G[Generator (GAN-style decoder)]
  G --> Y[6-step return path + uncertainty]
```

## Quickstart (Local)

### Install

```bash
python -m pip install -r requirements.txt
```

### Initialize DB (first time)

```bash
python main.py --mode init_db
```

### Collect data (network required)

```bash
python main.py --mode refresh-db --refresh_days 7
```

### Run inference (manual)

```bash
python main.py --mode intraday --min_k 1 --limit 8
python main.py --mode morning-report --min_k 1 --limit 8
```

## Scheduled Ops (Production)

Preferred entrypoint:
- `scripts/run_scheduled.py`

Preferred scheduler:
- systemd user timers in `deploy/systemd/`

Why:
- freshness gate + watch-only fallback
- watchdog timeouts
- overlap protection
- minimum output guarantee (MinRec + synthetic watch-only fallback)

Details:
- `OPS_ACCEPTANCE.md`
- `deploy/systemd/README.md`

## Training + Optuna (Long Runs)

### Full training (no tuning)

```bash
python main.py --mode train --no_collect --offline_ok
```

### Optuna tuning (resumable via SQLite)

```bash
export AETHER_OPTUNA_TRIALS=200
export AETHER_OPTUNA_STORAGE=sqlite:///analysis/optuna_full.db
export AETHER_OPTUNA_STUDY_NAME=aether_optuna_full
export AETHER_OPTUNA_LOAD_IF_EXISTS=1
python main.py --mode train --tune --no_collect --offline_ok
```

### End-to-end long pipeline (detach)

Runs:
1) Optuna + training
2) backtest suite
3) (optional) ablation suite

```bash
python scripts/optuna_full_run.py --tag long --optuna_trials 200 --no_telegram --run_ablation --detach
tail -n 200 logs/optuna_full_long_*.log
```

## Evaluation

```bash
# Backtest (main)
python main.py --mode backtest --days 30

# Suite runner (writes JSON artifacts into analysis/)
python scripts/eval_suite.py --days 30 --stride_hours 4 --tag eval_30d_4h --no_telegram

# Ablation (factors/context)
python scripts/ablation_suite.py --days 7 --stride_hours 4 --tag abl_7d_4h --include_no_context --no_telegram
```

## Repo Layout

- `main.py`: CLI entrypoint (train / refresh-db / intraday / morning-report / backtest)
- `data/`: DB + collectors + feature engineering
- `models/`: model definitions + ensemble configs
- `inference/`: predictor + recommender funnel
- `training/`: trainer + evaluator (backtest engine)
- `scripts/`: ops runners, validation, healthcheck, eval/ablation, automation
- `deploy/systemd/`: user timers/services for scheduled ops
