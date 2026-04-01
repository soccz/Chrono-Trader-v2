# Evaluation Protocol (v1)

Goal: before changing models/features (FF3/FF5-like factors, encoder/CNN fusion, GAN distribution head),
freeze a consistent evaluation loop so improvements are measurable and regressions are caught.

## Rules (Non-Negotiable)

- Always record the git commit hash for an evaluation run.
- Never compare results across different data windows without stating the window explicitly.
- Keep evaluation settings (days, stride, filters, costs) stable unless you intentionally change them and note it.

## Baseline (Fast, Daily)

Use a quick backtest to catch obvious breakage and measure direction + calibration trends.

- Command:
  - `AETHER_BACKTEST_STRIDE_HOURS=4 python main.py --mode backtest --days 7 --no_telegram`
- Expected artifacts:
  - `analysis/backtest_summary_latest.json`
  - `analysis/backtest_summary_main_YYYYmmdd_HHMMSS.json`
  - `analysis/backtest_gate_values_main_YYYYmmdd_HHMMSS.csv`

## Full Backtest (Slower, Weekly)

- Command:
  - `AETHER_BACKTEST_STRIDE_HOURS=1 python main.py --mode backtest --days 30 --no_telegram`

## Metrics To Track

- `n_trades` (must be > 0)
- `win_rate_pct`
- `avg_return_per_trade`
- `sharpe_annualized`
- `max_drawdown_pct`
- `ece` (calibration)
- `spearman_abs_error_uncertainty` (uncertainty usefulness)

## Acceptance Gate (Suggested)

Do not proceed to full training runs if:

- `n_trades == 0`
- metrics files are missing
- ops contract breaks (scheduled run does not produce >=1 output artifact)

