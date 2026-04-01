# AETHER (Chrono-Trader v2)

Probabilistic, ops-oriented crypto forecasting and recommendation system for live use.

This repository has two tracks:
- repo root: the live crypto prediction, recommendation, dashboard, and ops system
- `aaa/`: the paper-writing and research-packaging track

This README is the root-system overview. It should describe the production-facing architecture first. For operational usage details, see `USAGE_GUIDE.md` / `사용가이드.md`.

Not financial advice.

## Abstract

Markets exhibit both trend-level structure and local motif repetition. In crypto, macro drift often appears first in leaders (e.g., BTC/ETH) and then propagates to followers, while idiosyncratic local patterns still dominate many moves.
AETHER is built to represent these two forces simultaneously and to produce recommendations that are (a) uncertainty-aware and (b) operationally robust under degraded data/network conditions.

Core idea:
- model level: use an attention + TCN hybrid encoder to read both macro flow and local motifs
- output level: generate a future return distribution rather than a point estimate
- system level: turn that distribution into strategy-split operations for trend, pattern-follower, intraday, and morning runs
- execution level: repo-root live ops target KRW spot, so `Short` stays informational/watch-only unless explicitly enabled

## 1. Introduction

Most single-model approaches implicitly choose an inductive bias: either prioritize global dependencies (trend/regime) or focus on local motifs (shape-level patterns). In practice, crypto price action alternates: sometimes the macro tape dominates, and sometimes local microstructure motifs dominate.

AETHER explicitly represents both views through an attention + TCN hybrid and exposes a routing variable (gate) to support debugging and iterative refinement. Separately, it treats operational failure modes (stale data, network loss, long-running tasks) as first-class constraints of the system.

## Problem Setting

Given an hourly sequence window (168h) of engineered features for a market, predict a multi-step future return path (**12h horizon**) and produce a small set of ranked recommendations.

Design constraints:
- Regimes change; features must capture both macro and micro structure.
- Predictions must carry uncertainty, and filtering must respect it.
- Scheduled runs must not hang, must enforce data freshness, and must degrade safely.
- Each scheduled run must emit at least one output item (trade or watch-only), to keep automation consistent.
- Training universe is dynamic (top-N by 24h volume) rather than a fixed coin list.

## Method Overview

High-level block diagram:

```text
Upbit/DB candles
  -> feature engineering (macro + factors + technicals)
  -> hybrid encoder: Transformer (global) + CNN (local)
  -> explainable gated fusion + FiLM regime conditioning
  -> generator: multi-step return path (12h)
  -> uncertainty estimation (PI_80) + staged recommendation funnel
  -> net-alpha filter (gross - fee - slippage) + soft downside-risk score
  -> ranked outputs with attention_top3 + prototype_match (trade or watch-only)
```

### Context Features (Macro + Memory)

We form a macro tape proxy and a "pattern memory" signal:
- **Market index return**: a market-cap-like index from BTC/ETH (proxy for the global trend).
- **Historical similarity**: similarity of the recent macro-return window against a memory bank of past windows (pattern localization).

We augment with market-relative and cross-sectional structure:
- **rolling closed-form beta/alpha** (vectorized, window W ∈ {72, 168, 336}h tuned by Optuna)
- **residual return** as prediction target (coin return minus beta × BTC return)
- **4-state BTC regime** (Bull/Bear × quiet/volatile via 200h MA + realized volatility quantiles)
- **FF5-style factors × regime interactions** (e.g., `factor_mom_x_bull`, `factor_liq_x_bear`)
- **cross-sectional rank normalization** across the dynamic universe

Source of truth for the full feature set: `utils/config.py` (`config.Data.FEATURE_COLUMNS`).

### Hybrid Representation Learning (Global + Local)

We encode the same sequence through two complementary views:

```text
Input (168h x 32 features)
  |-> Transformer encoder (global structure / regime-level dependencies)
  |     attention_weights[-1] -> (B, 168) importance row
  |-> Attention-guided CNN (input weighted by softmax(attention_importance))
  |     local motifs / shape primitives
  `-> Explainable gated fusion (route / weight the two views)
  `-> FiLM conditioning on 4-state BTC regime
  `-> Probabilistic decoder (GAN/CVAE path) -> 3-step residual return path
```

The fusion gate is treated as an interpretable variable: when the system leans on local vs global representations, we can attribute which representation dominated the output. `attention_top3` exposes the three most influential timesteps per prediction.

### Uncertainty-Aware Decision Funnel

Predictions do not directly become trades. They pass through a staged funnel where uncertainty and operational constraints act as hard filters:
- tradeable validation
- regime/lead-lag heuristics (macro + micro)
- liquidity constraints
- **net-alpha filter**: live `intraday` long is treated as a spot entry signal, so Step 1 budgets entry-side cost for admission; realized evaluation still subtracts full round-trip cost
- **downside-risk score**: `step1_score = net_alpha - lambda * max(0, pi_guard_floor - directional_PI_guard)` so downside tail remains penalized without a brittle hard veto
- uncertainty constraints
- similarity / pattern checks (DTW vs success patterns)

Each recommendation carries: `net_alpha`, `PI_80` interval, `attention_top3` (top-3 influential timesteps), and `prototype_match` (nearest success pattern ID + similarity score).

For repo-root live ops, KRW spot is the execution contract. `Short` can still be emitted as a model/view signal, but it degrades to watch-only rather than becoming an executable position by default.

This separation (modeling vs decision) makes the system easier to stabilize in production.

## Ops Contract (Production Constraint)

In unattended scheduling, failure modes are first-class. AETHER enforces:
- watchdog timeouts
- freshness gates (stale DB abort with a distinct exit code)
- safe fallback to watch-only on stale/offline runs
- minimum-output policy (MinRec) so each scheduled run emits at least one item

Operational contract and exit codes: `OPS_ACCEPTANCE.md`.

## Related Work (Conceptual Positioning)

This project is conceptually adjacent to:
- **Time-series Transformers**: global dependency modeling and long-range context extraction.
- **CNN/TCN-style local motif extractors**: pattern primitives and shape-level inductive bias.
- **Gating / mixture-of-experts routing**: conditional computation or conditional blending between representations.
- **Probabilistic forecasting**: generating distributions or scenarios rather than single-point estimates.
- **Uncertainty estimation and calibration**: MC-dropout-style epistemic proxies and post-hoc calibration metrics (e.g., ECE).
- **Factor models**: market-relative decomposition and cross-sectional premia signals (FF-style construction adapted to crypto).

The intended contribution is not a new theorem; it is a coherent integration where representation learning, decision filtering, and operational reliability are aligned under a single contract.

## Assumptions

The system (and the evaluation) implicitly assumes:
- **Timestamp integrity**: candle timestamps are aligned and monotonic; joins across markets and index series use the same time basis.
- **No leakage**: features use only information available at decision time (especially similarity/memory features).
- **Stable symbol universe**: market listings/delistings and “tradeable” status are handled consistently.
- **Sufficient liquidity**: liquidity thresholds approximate tradability; extreme slippage is not fully modeled.
- **Operational environment**: scheduled jobs run on a host with stable clock/timezone configuration and persistent storage.

## Threats to Validity

Threats that can invalidate conclusions or inflate backtest performance:
- **Backtest bias**: simplified fills, slippage, and fee modeling; ignoring partial fills and orderbook impact.
- **Selection bias**: screening top markets by recent activity can change the effective distribution over time.
- **Hyperparameter overfitting**: long Optuna runs can overfit to a particular window/split configuration.
- **Non-stationarity**: regime changes may invalidate tuned parameters quickly; performance may not transfer.
- **Data quality drift**: missing candles, API/DB staleness, and symbol mapping changes can alter feature semantics.
- **Metric mismatch**: optimizing proxy objectives (e.g., uncertainty correlation) may not align with trading utility.

## Strengths

- **Hybrid bias**: global trend modeling and local motif extraction are not forced into a single inductive bias.
- **Attention-guided CNN**: Transformer attention directly steers CNN input weighting, coupling the two paths.
- **Regime conditioning**: FiLM layers condition the fused representation on the 4-state BTC regime.
- **Explainable routing**: gate + `attention_top3` + `prototype_match` expose why a recommendation was made.
- **Cost-aware filtering**: net-alpha and PI_80 gates prevent recommendations that don't survive fees/slippage.
- **Dynamic universe**: training and inference use the top-N markets by 24h volume, not a fixed list.
- **Leak-free CV**: Purged Walk-Forward with 6h embargo prevents look-ahead contamination in Optuna tuning.
- **Uncertainty as a constraint**: filtering is explicitly uncertainty-aware rather than post-hoc.
- **Ops resilience**: stable scheduled automation (timeouts, freshness gates, safe fallback, minimum-output guarantee).

## Limitations (Current)

- **Compute cost**: Optuna trials are expensive on CPU (Purged Walk-Forward CV × MC-Dropout).
- **Backtest realism**: simplified fills/slippage and incomplete microstructure effects may overstate performance.
- **Context fragility**: macro context features can help or hurt depending on regime and alignment; leakage/clock alignment must be audited continuously.
- **Short data history**: dynamic universe alts have ~6 months of data; regime/factor signals need longer history to stabilize.
- **Exchange dependency**: data collection and tradeability are tailored to Upbit-style market data.

## Current Status (2026-04)

IC analysis on holdout (Oct 2025 – Jan 2026, 17 markets) showed raw IC = −0.06, Short IC = −0.16.
The GAN+Transformer phase is complete. Short 58% win rate was market beta (BTC −22% during test period), not alpha.
Signal limit is feature quality and predictability at 12h, not model architecture.

**Next direction: cross-sectional factor model.**
Rank 200+ coins by momentum/liquidity/volatility factors, long top decile / short bottom decile, neutralize market beta.
IC measurable in minutes (vs 22h backtest cycle). Existing features (`rank_return_4h`, `breadth_ratio`, etc.) are reusable.

The ops infrastructure (scheduler, dashboard, data pipeline) remains in place and carries over to the new strategy.

---

## Roadmap (Research Direction)

1. **Cross-sectional factor model** *(next immediate step)*
   - Rank-based long/short on momentum, liquidity, volatility factors
   - XGBoost or Ridge regression on cross-sectional ranks
   - IC as primary evaluation metric
2. **Macro context redesign**
   - multi-scale context (daily/4h/1h) embeddings, regime state separation, and explicit delay/propagation modeling.
2. **Factor model stabilization**
   - robust factor construction under thin liquidity, shrinkage/regularization, and regime-conditioned factor weighting.
3. **Calibration + probabilistic outputs**
   - tighter coupling between uncertainty calibration (ECE), decision thresholds, and realized outcomes.
4. **Evaluation hardening**
   - fixed end-time backtests for fair ablations, realized cost models (fees/slippage), and walk-forward stress suites.
5. **Operational autonomy**
   - continuous health reporting, automated rollback to prior weights, and better market rotation criteria for top-N refresh pools.
6. **Data depth**
   - BTC/ETH backfill to 2019 (2 halvings); alt backfill to 2021 where available; regime/factor signals require ≥2y to stabilize.

## Where To Look Next (Source of Truth)

- Full architecture definition: `PROJECT_ARCHITECTURE.md`
- Learning/eval contract: `EVAL_PROTOCOL.md`
- Ops contract: `OPS_ACCEPTANCE.md`
