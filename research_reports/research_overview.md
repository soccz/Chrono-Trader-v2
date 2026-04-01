## Chrono-Trader Research Overview

This file is a research reference note, not the runtime source of truth.
For the live system, prefer `OPS_ACCEPTANCE.md`, `README.md`, and `PROJECT_ARCHITECTURE.md`.
`aaa/` remains the dedicated paper/research-writing track when a publication-oriented narrative is needed.

### 1. Motivation

Crypto spot markets suffer from regime shifts, low-liquidity pockets, and pump-and-dump events. Chrono-Trader approaches this by:

- Generating **short-horizon return trajectories** via an attention-conditioned probabilistic decoder instead of one-step predictions.
- Detecting **micro pump events** with gradient-boosted classification.
- Applying a recommender that filters by liquidity, uncertainty (MC Dropout), and DTW pattern similarity to output long/short ideas.

### 2. Current Architecture (as implemented)

| Stage | Core idea | Main files |
|-------|-----------|------------|
| Data ingestion | Upbit hourly OHLCV, dynamic index, technical indicators, pump features | `data/collector.py`, `data/preprocessor.py` |
| Sequence generator | Transformer encoder + attention-guided TCN/CNN + probabilistic decoder → short-horizon return path | `models/hybrid_model.py`, `models/transformer_encoder.py`, `models/gan_decoder.py`, `models/cvae_decoder.py` |
| Pump classifier | XGBoost multi-class pump probability | `training/pump_trainer.py`, `inference/pump_predictor.py` |
| Recommender | Liquidity → uncertainty → DTW filters → compounded-return threshold | `inference/recommender.py` |
| Evaluation | Backtest pipeline + calibration diagnostics | `training/evaluator.py`, `analysis/validate_uncertainty.py` |

### 3. Training Strategy

1. **Hybrid probabilistic forecaster**
   - Input: 168×runtime feature tensor with macro context, α/β, factors, and regime features.
   - Encoder: attention for global flow + attention-guided TCN/CNN for local shape.
   - Decoder: GAN/CVAE family for path distribution generation.
   - Training emphasis: reconstruction, calibration, uncertainty usefulness, and stable sampling.

2. **Pump detection**: XGBoost with class-balanced weighting, multi-class labels from forward high.

3. **Sampling / MC inference**: repeated forward passes or decoder sampling, uncertainty used for filtering and ECE reporting.

4. **DTW Pattern Filter**: soft-DTW vs cached “success patterns” to prune implausible signals.

### 4. Observed Issues

- `lossG_adv` heavily negative; λ_recon saturates → generator collapses to regression.
- Uncertainty Spearman(|error|, σ) near zero → dropout variance uninformative.
- Recommender outputs mainly short signals with large negative returns.
- Pump classifier lift modest.
- Execution realism absent (no slippage/latency modeling).

### 5. Research Roadmap

#### 5.1 Stabilize Hybrid Probabilistic Forecaster (Priority)
- Two-time-scale updates (lr critic = 2× generator), adaptive critic iterations.
- Replace heuristic λ with GradNorm/AdaLoss to target `|L_adv| : L_recon ≈ 0.5–1.0`.
- Spectral norm + light DiffAug to curb mode collapse.
- Dashboard logging: `W_est`, `||∇D||`, `loss` ratios, collapse score. Auto rollback when metrics drift (e.g., `||∇D||` outside 0.8–1.2 for 500 steps).

#### 5.2 MC Dropout & Calibration
- Sweep dropout p and MC samples (30/50/100) and monitor ECE, Spearman corr.
- Compare Brier-style calibration term vs differentiable ECE histogram.
- Plot uncertainty cutoff vs backtest return to choose thresholds.

#### 5.3 Pattern Discovery (DTW)
- Soft-DTW with Sakoe–Chiba band; choose τ by maximizing Precision/Profit@K on validation.
- Track per-lag contribution; remove lags with poor performance.
- Switch from top-K to threshold-based candidate selection.

#### 5.4 Alpha/Beta & Features
- EWMA beta with Vasicek shrinkage, Huber regression for robustness.
- Mask β where std error high; drop α/β features stuck in bottom 30% importance.
- Upgrade multi-scale CNN (inception kernels, dilations, depthwise separable + channel attention).

#### 5.5 Pump Classifier & Ensemble Diversity
- Optuna 50-trial tuning (time-aware split) with scale_pos_weight.
- Bagging via feature subsets; snapshot ensembles/SWA for diversity.
- Log ensemble correlation/disagreement to keep Spearman mean ≤ 0.8.

#### 5.6 Execution Realism
- Price API retry with stale-data skip.
- Slippage model (spread, depth, volatility) integrated into backtester.
- Abort trades when slippage or spread exceeds preset budget.

### 6. 48-Hour Sprint Outline

**Day 1 – Diagnostics**
1. WGAN dashboard (W_est, ||∇D||, ratio, collapse score).
2. MC Dropout study (N=30 vs 50) logging ECE & Spearman.
3. DTW distribution logging + τ grid search.
4. Implement EWMA β + shrinkage, monitor β standard error.

**Day 2 – Improvements**
1. Critic iteration & λ_gp/λ_recon sweep → pick best combo.
2. CNN kernel/dilation mini-grid; evaluate val AUC + backtest.
3. XGBoost Optuna tuning with early stopping.
4. Ensemble diversity test (lookback/feature subsets, snapshot ensemble).
5. Plug slippage model into backtester; skip trades breaching risk limits.

### 7. Key Metrics to Track
- Training: `W_est`, `||∇D||`, `lossC`, `lossG_adv`, `lossG_recon`, `lossG_ece`, `λ_gp`, `λ_recon`, collapse score.
- Calibration: ECE, Brier score, Spearman(|error|, σ), low/high-uncertainty t-tests.
- Recommender: Precision@K, Profit@K, liquidity filter hit rate, DTW τ stats.
- Backtest: win rate, Sharpe, drawdown, hit rate at target horizon, profit factor.
- Ensemble: pairwise correlation, disagreement rate, oracle upper bound difference.

### 8. Path to Publication
1. **Stabilize the probabilistic decoder** so generated trajectories are realistic and uncertainty is useful.
2. **Calibrated uncertainty** achieving ECE < 0.05 and Spearman ≥ 0.3, leading to improved filtered returns.
3. **Backtest lift** over baselines (positive profit or drawdown reduction) under realistic execution.
4. **Ablation studies** for each component (sequence vs scalar, uncertainty loss, pump filter, DTW patterns).
5. **Execution realism** documented (slippage, liquidity, latency handling).

Achieving these milestones will position Chrono-Trader as a credible research contribution on probabilistic crypto forecasting with calibrated uncertainty.
