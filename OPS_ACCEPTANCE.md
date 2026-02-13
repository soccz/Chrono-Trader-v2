# Ops Acceptance Criteria (Contract)

This document defines the operational contract for scheduled runs. The goal is predictable daily automation that never hangs, always produces at least one actionable output artifact, and degrades safely when data/network is bad.

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
- A metrics payload is written:
  - `analysis/run_markets_metrics_{mode}.json` (latest)
  - `analysis/run_markets_metrics_{mode}.jsonl` (append-only history)

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
