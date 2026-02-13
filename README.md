# AETHER (Chrono-Trader v2)

Explainable, ops-oriented crypto forecasting and recommendation system.

This README intentionally focuses on ideas, design logic, and research direction (paper-style). For operational usage details, see `USAGE_GUIDE.md` / `사용가이드.md`.

Not financial advice.

## Abstract

Markets exhibit both trend-level structure and local motif repetition. In crypto, macro drift often appears first in leaders (e.g., BTC/ETH) and then propagates to followers, while idiosyncratic local patterns still dominate many moves.
AETHER is built to represent these two forces simultaneously and to produce recommendations that are (a) uncertainty-aware and (b) operationally robust under degraded data/network conditions.

Core idea: treat the problem as **pattern localization in time** ("where are we in a known historical motif?") while conditioning on **macro context** (BTC/ETH index), then route decisions through an **explainable hybrid model** and a strict recommendation funnel.

## Problem Setting

Given an hourly sequence window (168h) of engineered features for a market, predict a multi-step future return path (6h horizon) and produce a small set of ranked recommendations.

Design constraints:
- Regimes change; features must capture both macro and micro structure.
- Predictions must carry uncertainty, and filtering must respect it.
- Scheduled runs must not hang, must enforce data freshness, and must degrade safely.
- Each scheduled run must emit at least one output item (trade or watch-only), to keep automation consistent.

## Method Overview

### Context Features (Macro + Memory)

We form a macro tape proxy and a "pattern memory" signal:
- **Market index return**: a market-cap-like index from BTC/ETH (proxy for the global trend).
- **Historical similarity**: similarity of the recent macro-return window against a memory bank of past windows (pattern localization).

We augment with market-relative and cross-sectional structure:
- **alpha/beta** style features (market-relative movement)
- **crypto FF-style factors** (size/momentum/volatility/liquidity premia proxies)

Source of truth for the full feature set: `utils/config.py` (`config.Data.FEATURE_COLUMNS`).

### Hybrid Representation Learning (Global + Local)

We encode the same sequence through two complementary views:

```text
Input (168h x 27 features)
  |-> Transformer encoder (global structure / regime-level dependencies)
  |-> CNN stack (local motifs / shape primitives)
  `-> Explainable gated fusion (route / weight the two views)
  `-> Generator (GAN-style decoder) -> 6-step return path
```

The fusion gate is treated as an interpretable variable: when the system leans on local vs global representations, we can attribute which representation dominated the output.

### Uncertainty-Aware Decision Funnel

Predictions do not directly become trades. They pass through a staged funnel where uncertainty and operational constraints act as hard filters:
- tradeable validation
- regime/lead-lag heuristics (macro + micro)
- liquidity constraints
- expected-return constraints
- uncertainty constraints
- (optional) similarity / pattern checks

This separation (modeling vs decision) makes the system easier to stabilize in production.

## Ops Contract (Production Constraint)

In unattended scheduling, failure modes are first-class. AETHER enforces:
- watchdog timeouts
- freshness gates (stale DB abort with a distinct exit code)
- safe fallback to watch-only on stale/offline runs
- minimum-output policy (MinRec) so each scheduled run emits at least one item

Operational contract and exit codes: `OPS_ACCEPTANCE.md`.

## Strengths

- **Hybrid bias**: global trend modeling and local motif extraction are not forced into a single inductive bias.
- **Explainable routing**: the gate exposes which representation dominated, improving debuggability.
- **Uncertainty as a constraint**: filtering is explicitly uncertainty-aware rather than post-hoc.
- **Ops resilience**: stable scheduled automation (timeouts, freshness gates, safe fallback, minimum-output guarantee).

## Limitations (Current)

- **Compute cost**: Optuna trials are expensive on CPU (5-fold CV, MC-Dropout).
- **Backtest realism**: simplified fills/slippage and incomplete microstructure effects may overstate performance.
- **Context fragility**: macro context features can help or hurt depending on regime and alignment; leakage/clock alignment must be audited continuously.
- **Exchange dependency**: data collection and tradeability are tailored to Upbit-style market data.

## Roadmap (Research Direction)

1. **Macro context redesign**
   - multi-scale context (daily/4h/1h) embeddings, regime state separation, and explicit delay/propagation modeling.
2. **Factor model stabilization**
   - robust factor construction under thin liquidity, shrinkage/regularization, and regime-conditioned factor weighting.
3. **Calibration + probabilistic outputs**
   - tighter coupling between uncertainty calibration (ECE), decision thresholds, and realized outcomes.
4. **Evaluation hardening**
   - leak checks, fixed end-time backtests for fair ablations, cost models (fees/slippage), and walk-forward stress suites.
5. **Operational autonomy**
   - continuous health reporting, automated rollback to prior weights, and better market rotation criteria for "top-N" refresh pools.

## Where To Look Next (Source of Truth)

- Full architecture definition: `PROJECT_ARCHITECTURE.md`
- Learning/eval contract: `EVAL_PROTOCOL.md`
- Ops contract: `OPS_ACCEPTANCE.md`
