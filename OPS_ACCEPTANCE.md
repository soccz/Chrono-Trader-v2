# Ops Acceptance Criteria (Contract)

This document defines the operational contract for scheduled runs. The goal is predictable daily automation that never hangs, always produces at least one actionable output artifact, and degrades safely when data/network is bad.

Repo-root ops target KRW spot usage. In live scheduled runs, `Short` may remain as a directional/model signal, but it must not become an executable position unless explicitly enabled.

## Modes

- `refresh-db`: Collects recent candles for a diversified watchlist (DB warmup job).
- `intraday`: Inference-only scheduled run (every 4h).
- `morning-report`: Inference-only scheduled run (daily 08:00 KST).

Recommended scheduler entrypoint:
- `scripts/run_scheduled.py` (runs `refresh-db` then scheduled inference).

## Exit Codes

The pipeline must treat these exit codes as a contract:

- `0`: success (artifacts produced or safe no-op when appropriate)
- `2`: stale/offline abort (freshness gate triggered, or refresh-db offline without `--offline_ok`)
- `3`: watchdog timeout (process exceeded `--timeout_sec` for that mode)

## Artifacts (Must Exist On Success)

For each scheduled inference run (`intraday`, `morning-report`) that exits `0`:

- A CSV in `recommendations/` is created for that run.
  - **Required columns** (pipeline logic depends on these): `market`, `signal`, `expected_return`, `net_alpha`, `pi_low_80`, `pi_high_80`, `attention_top3`, `prototype_match`, `confidence`, `consensus_score`, `position_size`, `status`, `trend_alignment`, `decision_score`, `gate_value`, `volatility`, `dtw_distance`.
  - **Informational columns** (present for display/debugging, not used by downstream gates): `strategy`, `current_price`, `pattern`, `reason`.
- A metrics payload is written:
  - `analysis/run_markets_metrics_{mode}.json` (latest)
  - `analysis/run_markets_metrics_{mode}.jsonl` (append-only history)

## Recommendation Funnel Contract

A recommendation is only eligible for `position_size > 0` if it survives every step below, applied **in this order**:

- **Step 0: Tradeable gate** — Market is not delisted, not a stablecoin, not a leveraged token (`UP`/`DOWN`/`BEAR`/`BULL`).
- **Step 1: Direction consistency + net_alpha + step1_score (combined gate)** —
   - ≥ 66 % of pattern-based directional signals must agree with the predicted direction.
   - `net_alpha > 0` under the active cost budget (round-trip default; entry-side for live intraday long).
   - `step1_score = net_alpha - lambda * max(0, pi_guard_floor - directional_PI_guard)` must be > 0.
   - Any candidate that fails any of these three checks is rejected at this step.
- **Step 1.5: Regime / Lead-Lag adjustment** — SMA crossover regime detection; BTC correlation boost applied to scores. Not a filter but modifies scores before downstream gates.
- **Step 2: Liquidity** — 24h trading value ≥ threshold (1 B KRW live, 50 M KRW backtest).
- **Step 3: Min Expected Return** — `|expected_return| ≥ 0.1 %`.
- **Step 3.5: Consensus** — consensus score ≥ mode-specific minimum (default 0.6, intraday 0.55, counter-trend 0.8).
- **Step 4: Uncertainty** — uncertainty below adaptive threshold (batch 65th-quantile cutoff; counter-trend uses 0.7× multiplier).
- **Step 5: DTW Pattern similarity** — DTW distance ≤ 1.5.
- **Short block + final selection** — `Short` signals blocked for KRW spot live ops unless explicitly enabled; final eligible set assembled.
- **MinRec / Watch-only guarantee** — If all filters remove every candidate, MinRec produces at least one `Watch` item with `position_size=0.0`.

## Dynamic Universe

- Training and inference universe: top-N markets by 24h trading value (default N=100).
- Excludes: stablecoins, leveraged tokens (`UP`/`DOWN`/`BEAR`/`BULL`), listings < 14 days.
- Fallback to `TRAIN_COINS_FALLBACK` if DB is unavailable.
- Universe is re-evaluated at each training run and each scheduled inference run.

## Freshness Gate Policy

- Default behavior: if DB is too stale for the scheduled mode, abort with exit code `2`.
- Safe fallback behavior (scheduler): if inference exits `2`, rerun once with:
  - `--allow_stale_data`
  - `AETHER_RUNTIME_WATCH_ONLY=1` and `AETHER_MIN_REC_MODE=watch`
  - Goal: still produce at least 1 output, but never suggest a position size.

Important: `refresh-db` failure alone must not force watch-only; the DB may still be fresh.

## Minimum Output Policy

- Live runs must produce at least `config.Recommender.MIN_RECOMMENDATIONS_LIVE` items.
- If strict filters remove everything, MinRec ensures at least one output:
  - Prefer a safe forced trade when possible.
  - Otherwise output `Watch (MinRec)` with `position_size=0.0`.
- When `AETHER_RUNTIME_WATCH_ONLY=1`, all outputs become `Watch (Runtime)` with `position_size=0.0`.

## Watchdogs

- `refresh-db`, `intraday`, `morning-report` must be wrapped in watchdog timers.
- If watchdog fires: exit code `3`.

## Optional: Speed/Stability Switch

- `--skip_pattern_followers` (morning-report only): skips pattern followers (DTW-heavy).
- `--skip_pump_radar` (morning-report only): skips pump radar.
- `--skip_aux` (morning-report only): skips both pattern followers + pump radar.
  - Use for ops validation or when compute is constrained.

## Model / CV Contract

- Hyperparameter tuning uses **Purged Walk-Forward CV** with 6h embargo between train and validation folds.
- `model_config.json` must be regenerated after any change to features, horizon, or CV method.
- Prediction horizon: **3h** (sum of log-returns t+1 ~ t+3).
- Prediction target: **residual return** (coin return minus beta × BTC return).
