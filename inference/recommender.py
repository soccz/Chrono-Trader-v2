import json
import pandas as pd
from utils.telegram_bot import send_alert
import logging
import numpy as np
import os
from datetime import datetime, timezone

from utils.config import config
from utils.logger import logger
from data.database import get_trading_values_for_markets
from data.database import load_data
from data.collector import get_current_price
from utils.price_cache import get_tradeable_markets

_LAST_RUN_DIAGNOSTICS = None
_LAST_RUN_DIAGNOSTICS_PATH = None


def _configured_pattern_horizon() -> int:
    try:
        horizon = int(getattr(config.Data, "FUTURE_WINDOW_SIZE", 0) or 0)
    except Exception:
        horizon = 0
    return max(1, horizon)


def _coerce_pattern_array(pattern, fallback_horizon: int | None = None) -> np.ndarray:
    if pattern is None:
        return np.zeros(int(fallback_horizon or _configured_pattern_horizon()), dtype=float)
    try:
        arr = np.array(pattern, dtype=float).reshape(-1)
        if arr.size > 0 and np.isfinite(arr).any():
            return arr
    except Exception:
        pass
    return np.zeros(int(fallback_horizon or _configured_pattern_horizon()), dtype=float)


def _get_latest_close_map_from_db(markets: list) -> dict:
    """
    Best-effort DB fallback for current_price when API is offline.
    Returns {market: last_close}.
    """
    try:
        markets = [str(m) for m in (markets or []) if m]
        if not markets:
            return {}
        placeholders = ", ".join("?" for _ in markets)
        q = (
            "SELECT t.market, t.close "
            "FROM crypto_data t "
            "JOIN ("
            f"  SELECT market, MAX(timestamp) AS max_ts FROM crypto_data WHERE market IN ({placeholders}) GROUP BY market"
            ") m "
            "ON t.market = m.market AND t.timestamp = m.max_ts"
        )
        df = load_data(q, params=list(markets))
        if df is None or df.empty:
            return {}
        out = {}
        for _, r in df.iterrows():
            try:
                mk = str(r.get("market"))
                cl = float(r.get("close"))
                if mk and np.isfinite(cl) and cl > 0:
                    out[mk] = cl
            except Exception:
                continue
        return out
    except Exception:
        return {}


def _resolve_pi_low_80_floor(mode: str, strategy: str | None = None) -> float:
    try:
        base = float(getattr(config.Recommender, "PI_LOW_80_MIN_LIVE", 0.0) or 0.0)
    except Exception:
        base = 0.0

    if str(mode or "").strip().lower() != "live":
        return base

    if str(strategy or "").strip().lower() == "intraday":
        try:
            return float(getattr(config.Recommender, "PI_LOW_80_MIN_INTRADAY", base) or base)
        except Exception:
            return base

    return base


def _compute_directional_trade_metrics(
    expected_return: float,
    pi_low_80: float,
    pi_high_80: float,
    pi_low_floor: float,
) -> dict:
    signal = "Long" if expected_return > 0 else ("Short" if expected_return < 0 else "Neutral")
    directional_return = float(expected_return)
    directional_pi_guard = float(pi_low_80)

    if signal == "Short":
        directional_return = float(-expected_return)
        directional_pi_guard = float(-pi_high_80)

    return {
        "signal": signal,
        "directional_return": directional_return,
        "directional_pi_guard": directional_pi_guard,
        "pi_guard_floor": float(pi_low_floor),
    }


def _compute_step1_score(
    net_alpha: float,
    directional_pi_guard: float,
    pi_guard_floor: float,
    *,
    mode: str | None = None,
    strategy: str | None = None,
    signal: str | None = None,
) -> dict:
    try:
        penalty = float(getattr(config.Recommender, "STEP1_PI_GUARD_PENALTY", 0.35) or 0.35)
    except Exception:
        penalty = 0.35
    if (
        str(mode or "").strip().lower() == "live"
        and str(strategy or "").strip().lower() == "intraday"
        and str(signal or "").strip().lower() == "long"
    ):
        try:
            penalty = float(getattr(config.Recommender, "STEP1_PI_GUARD_PENALTY_INTRADAY_LONG", penalty) or penalty)
        except Exception:
            pass
    penalty = max(0.0, penalty)
    pi_guard_shortfall = max(0.0, float(pi_guard_floor) - float(directional_pi_guard))
    score = float(net_alpha) - penalty * pi_guard_shortfall
    return {
        "step1_score": score,
        "pi_guard_shortfall": pi_guard_shortfall,
        "pi_guard_penalty": penalty,
    }


def _compute_trade_quality_score(trade: dict) -> float:
    """
    Rank candidates by a long-only quality score when we are in live intraday spot mode.

    For live intraday longs we want ranking to reflect actual directional edge and
    expected return, not just raw confidence. For other cases, preserve the historical
    confidence × consensus sort.
    """
    try:
        confidence = float(trade.get("confidence", 0.0) or 0.0)
    except Exception:
        confidence = 0.0
    try:
        consensus = float(trade.get("consensus_score", 0.0) or 0.0)
    except Exception:
        consensus = 0.0
    try:
        step1_score = max(0.0, float(trade.get("step1_score", 0.0) or 0.0))
    except Exception:
        step1_score = 0.0
    try:
        expected_return = max(0.0, float(trade.get("expected_return", 0.0) or 0.0))
    except Exception:
        expected_return = 0.0

    confidence = max(0.0, min(1.0, confidence))
    confidence = max(0.5, confidence)
    consensus = max(0.0, min(1.0, consensus))
    base_quality = confidence * (0.5 + 0.5 * consensus)

    mode = str(trade.get("mode", "") or "").strip().lower()
    strategy = str(trade.get("strategy", "") or "").strip().lower()
    signal = str(trade.get("signal", "") or "").strip().lower()
    if mode == "live" and strategy == "intraday" and signal == "long":
        trend_alignment = str(trade.get("trend_alignment", "") or "").strip().lower()
        if trend_alignment == "following":
            trend_multiplier = 1.15
        elif trend_alignment == "counter":
            trend_multiplier = 0.75
        else:
            trend_multiplier = 0.90
        long_quality = (0.6 * step1_score) + (0.4 * expected_return)
        consensus_multiplier = 0.65 + 0.35 * consensus
        return long_quality * consensus_multiplier * trend_multiplier
    return base_quality


def _live_short_execution_allowed(mode: str) -> bool:
    if str(mode or "").strip().lower() != "live":
        return True
    return bool(getattr(config.Recommender, "LIVE_ALLOW_SHORT_EXECUTION", False))


def _per_leg_trade_cost() -> float:
    fee = float(getattr(config.Recommender, "TRADE_FEE_PER_LEG", 0.0005) or 0.0005)
    slippage = float(getattr(config.Recommender, "SLIPPAGE_PER_LEG", 0.0003) or 0.0003)
    return max(0.0, fee) + max(0.0, slippage)


def _resolve_step1_cost_budget(mode: str, strategy: str | None, signal: str) -> float:
    per_leg_cost = _per_leg_trade_cost()
    if str(mode or "").strip().lower() == "live" and str(strategy or "").strip().lower() == "intraday" and str(signal).strip().lower() == "long":
        try:
            legs = float(getattr(config.Recommender, "LIVE_INTRADAY_LONG_STEP1_COST_LEGS", 1.0) or 1.0)
        except Exception:
            legs = 1.0
        return max(0.0, legs) * per_leg_cost
    return 2.0 * per_leg_cost


def _resolve_min_consensus(mode: str, strategy: str | None = None, trend_alignment: str | None = None) -> float:
    # Backtest: 순수 방향 정확도 측정을 위해 consensus 필터 비활성화
    if str(mode or "").strip().lower() == "backtest":
        return 0.0

    strategy_name = str(strategy or "").strip().lower()
    trend_name = str(trend_alignment or "").strip().lower()
    is_counter = trend_name == "counter"

    if str(mode or "").strip().lower() == "live" and strategy_name == "intraday":
        attr = "COUNTER_TREND_MIN_CONSENSUS_SCORE_INTRADAY" if is_counter else "MIN_CONSENSUS_SCORE_INTRADAY"
    else:
        attr = "COUNTER_TREND_MIN_CONSENSUS_SCORE" if is_counter else "MIN_CONSENSUS_SCORE"

    default = 0.8 if is_counter else 0.6
    try:
        return float(getattr(config.Recommender, attr, default) or default)
    except Exception:
        return float(default)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.floating, float)):
        val = float(value)
        return val if np.isfinite(val) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _summarize_numeric_list(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
    }


def _funnel_status_counts(funnel_data: list) -> dict:
    counts = {}
    for trade in funnel_data or []:
        status = str(trade.get("status", "unknown") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: kv[0]))


def _capture_funnel_snapshot(name: str, funnel_data: list) -> dict:
    active = [t for t in (funnel_data or []) if str(t.get("status")) == "Initial Candidate"]
    active_sorted = sorted(active, key=_trade_sort_key, reverse=True)
    top_active = []
    for trade in active_sorted[:10]:
        top_active.append({
            "market": trade.get("market"),
            "strategy": trade.get("strategy"),
            "signal": trade.get("signal"),
            "expected_return": trade.get("expected_return"),
            "net_alpha": trade.get("net_alpha"),
            "step1_score": trade.get("step1_score"),
            "pi_guard_shortfall": trade.get("pi_guard_shortfall"),
            "pi_low_80": trade.get("pi_low_80"),
            "pi_low_80_floor": trade.get("pi_low_80_floor"),
            "consensus_score": trade.get("consensus_score"),
            "min_consensus_required": trade.get("min_consensus_required"),
            "uncertainty": trade.get("uncertainty"),
            "trend_alignment": trade.get("trend_alignment"),
            "status": trade.get("status"),
        })
    return {
        "step": str(name),
        "active_n": len(active),
        "status_counts": _funnel_status_counts(funnel_data),
        "top_active": _json_safe(top_active),
    }


def _set_last_run_diagnostics(payload: dict, path: str | None) -> None:
    global _LAST_RUN_DIAGNOSTICS, _LAST_RUN_DIAGNOSTICS_PATH
    _LAST_RUN_DIAGNOSTICS = payload
    _LAST_RUN_DIAGNOSTICS_PATH = path


def get_last_run_diagnostics() -> dict | None:
    return _LAST_RUN_DIAGNOSTICS


def get_last_run_diagnostics_path() -> str | None:
    return _LAST_RUN_DIAGNOSTICS_PATH


def _write_recommender_diagnostics(payload: dict, rec_tag: str) -> str | None:
    try:
        analysis_dir = "analysis"
        os.makedirs(analysis_dir, exist_ok=True)
        tag = str(rec_tag or "default").strip() or "default"
        path = os.path.join(analysis_dir, f"recommender_diagnostics_{tag}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_json_safe(payload), f, ensure_ascii=False, indent=2)
        return path
    except Exception as e:
        logger.warning(f"Failed to write recommender diagnostics: {e}")
        return None


def _synthesize_watch_only_recommendation(predictions: list, reason: str) -> dict:
    """
    Absolute last-resort fallback to satisfy ops contract: emit >=1 item per run.
    Always returns a watch-only recommendation (position_size=0.0).
    """
    prefer = "KRW-BTC"
    pred = None
    if predictions:
        for p in predictions:
            if p.get("market") == prefer:
                pred = p
                break
        if pred is None:
            pred = predictions[0]

    market = prefer
    if pred is not None and pred.get("market"):
        market = str(pred.get("market"))

    # Best-effort price: prediction -> DB last close -> None
    current_price = None
    if pred is not None:
        try:
            cp = pred.get("current_price")
            if cp is not None and float(cp) > 0:
                current_price = float(cp)
        except Exception:
            current_price = None
    if current_price is None and market:
        fb = _get_latest_close_map_from_db([market])
        try:
            cp = fb.get(market)
            if cp is not None and float(cp) > 0:
                current_price = float(cp)
        except Exception:
            current_price = None

    # Best-effort pattern: prediction -> zeros using the active runtime horizon.
    pattern = _coerce_pattern_array(
        pred.get("predicted_pattern") if isinstance(pred, dict) else None,
        fallback_horizon=_configured_pattern_horizon(),
    )

    expected_return = float(np.prod(1 + pattern) - 1) if pattern.size else 0.0
    signal = "Long" if expected_return > 0 else ("Short" if expected_return < 0 else "Neutral")

    return {
        "market": market,
        "signal": signal,
        "strategy": (pred.get("strategy") if isinstance(pred, dict) else None) or "fallback",
        "expected_return": expected_return,
        "confidence": 0.0,
        "uncertainty": float(pred.get("uncertainty")) if isinstance(pred, dict) and pred.get("uncertainty") is not None else float("nan"),
        "current_price": current_price,
        "pattern": pattern,
        "dtw_distance": 999.0,
        "volatility": float(np.std(pattern)) if pattern.size else 0.0,
        "position_size": 0.0,
        "status": "Watch (Fallback)",
        "fallback_reason": str(reason or "fallback"),
        "gate_value": float(pred.get("gate_value", 0.5)) if isinstance(pred, dict) else 0.5,
        "consensus_score": float(pred.get("consensus_score", 0.0)) if isinstance(pred, dict) else 0.0,
    }


def _trade_sort_key(trade: dict):
    """Ranking key: quality first, then consensus, then expected magnitude."""
    quality = _compute_trade_quality_score(trade)
    consensus = float(trade.get('consensus_score', 0.0))
    expected = abs(float(trade.get('expected_return', 0.0)))
    return (quality, consensus, expected)


def _compute_uncertainty_threshold(funnel_data: list, base_threshold: float) -> float:
    """Compute dynamic uncertainty threshold from current batch if enabled."""
    if not getattr(config.Recommender, "ENABLE_DYNAMIC_UNCERTAINTY_THRESHOLD", False):
        return float(base_threshold)

    active_unc = []
    for trade in funnel_data:
        if trade.get('status') == 'Initial Candidate':
            try:
                u = float(trade.get('uncertainty', np.nan))
            except Exception:
                u = np.nan
            if np.isfinite(u):
                active_unc.append(u)

    if len(active_unc) < 2:
        return float(base_threshold)

    quantile = float(getattr(config.Recommender, "DYNAMIC_UNCERTAINTY_QUANTILE", 0.65))
    quantile = max(0.1, min(0.95, quantile))
    raw_q = float(np.quantile(np.array(active_unc, dtype=float), quantile))

    min_mult = float(getattr(config.Recommender, "DYNAMIC_UNCERTAINTY_MIN_MULTIPLIER", 0.8))
    max_mult = float(getattr(config.Recommender, "DYNAMIC_UNCERTAINTY_MAX_MULTIPLIER", 4.0))
    lo = float(base_threshold) * min_mult
    hi = float(base_threshold) * max_mult
    adaptive = float(np.clip(raw_q, lo, hi))
    unc_arr = np.array(active_unc, dtype=float)

    logger.info(
        f"[Uncertainty] Adaptive threshold={adaptive:.4f} "
        f"(base={float(base_threshold):.4f}, q{int(quantile*100)}={raw_q:.4f}, "
        f"min={unc_arr.min():.4f}, median={np.median(unc_arr):.4f}, max={unc_arr.max():.4f}, n={len(active_unc)})"
    )
    return adaptive


def _get_market_index_safe() -> pd.DataFrame:
    try:
        from data.preprocessor import get_market_index
        return get_market_index()
    except Exception as exc:
        logger.warning(f"[Recommender] Market index unavailable, skipping regime filter: {exc}")
        return pd.DataFrame()


def _get_historical_success_patterns_safe() -> np.ndarray:
    try:
        from data.preprocessor import get_historical_success_patterns
        return get_historical_success_patterns()
    except Exception as exc:
        logger.warning(f"[Recommender] Success patterns unavailable, skipping DTW filter: {exc}")
        return np.array([])


def _get_pattern_similarity_safe(pattern: np.ndarray, success_pattern: np.ndarray) -> float:
    try:
        from inference.predictor import get_pattern_similarity
        return float(get_pattern_similarity(pattern, success_pattern))
    except Exception as exc:
        logger.warning(f"[Recommender] Pattern similarity unavailable, keeping candidate: {exc}")
        return 0.0


def get_market_index() -> pd.DataFrame:
    return _get_market_index_safe()


def get_historical_success_patterns() -> np.ndarray:
    return _get_historical_success_patterns_safe()


def get_pattern_similarity(pattern: np.ndarray, success_pattern: np.ndarray) -> float:
    return _get_pattern_similarity_safe(pattern, success_pattern)


def _log_recommendation_table(stage_name: str, trades: list):
    """Helper function to log the state of trades at each funnel step in a detailed table."""
    logger.info(f"\n--- {stage_name} ({len(trades)} candidates) ---")
    if not trades:
        logger.info("  No candidates to display.")
        return

    # Sort by confidence for consistent logging
    sorted_trades = sorted(trades, key=lambda x: x.get('confidence', 0), reverse=True)
    
    horizon = max(
        _configured_pattern_horizon(),
        max((len(np.array(trade.get("pattern", []), dtype=float).reshape(-1)) for trade in sorted_trades), default=0),
    )
    expected_return_header = f"Exp. ({horizon}H)"
    pattern_headers = [f"H+{i}" for i in range(1, horizon + 1)]
    headers = ["Market", "Signal", "Strategy", expected_return_header, "Conf.", *pattern_headers, "Status"]
    
    # Dynamically calculate column widths
    col_widths = {h: len(h) for h in headers}
    for trade in sorted_trades:
        col_widths["Market"] = max(col_widths["Market"], len(trade.get('market', '')))
        col_widths["Strategy"] = max(col_widths["Strategy"], len(trade.get('strategy', '')))
        col_widths["Signal"] = max(col_widths["Signal"], len(trade.get('signal', '')))
    
    # Set fixed widths for numeric/status columns
    col_widths["Market"] = max(col_widths["Market"], 6) + 2
    col_widths["Signal"] = max(col_widths["Signal"], 6) + 2
    col_widths["Strategy"] = max(col_widths["Strategy"], 8) + 2
    col_widths[expected_return_header] = max(len(expected_return_header), 11)
    col_widths["Conf."] = 8
    for h in pattern_headers:
        col_widths[h] = 8
    col_widths["Status"] = 45

    # Print Header
    header_line = " | ".join([f"{h:<{col_widths[h]}}" for h in headers])
    separator = "-+-".join(["-" * col_widths[h] for h in headers])
    logger.info(header_line)
    logger.info(separator)

    # Print Rows
    for trade in sorted_trades:
        status_color = "\033[92m" if trade['status'] == 'Recommended' else "\033[91m" if 'Failed' in trade['status'] else "\033[0m"
        
        pattern = _coerce_pattern_array(trade.get("pattern"), fallback_horizon=horizon)
        pattern_cells = []
        for idx in range(horizon):
            if idx < pattern.size:
                pattern_cells.append(f"{pattern[idx]:>+7.2%}")
            else:
                pattern_cells.append(f"{'':>7}")
        pattern_cols = " | ".join(pattern_cells)

        row_str = f"{trade['market']:<{col_widths['Market']}} | " \
                  f"{trade.get('signal', 'N/A'):<{col_widths['Signal']}} | " \
                  f"{trade['strategy']:<{col_widths['Strategy']}} | " \
                  f"{trade['expected_return']:>+10.2%} | " \
                  f"{trade['confidence']:>7.2%} | " \
                  f"{pattern_cols} | " \
                  f"{status_color}{trade['status']:<{col_widths['Status']}}\033[0m"
        logger.info(row_str)


def run(
    predictions: list,
    historical_data: pd.DataFrame = None,
    mode: str = 'live',
    min_k: int = 3,
    *,
    reference_time: datetime | None = None,
    emit_artifacts: bool = True,
    collect_diagnostics: bool = False,
):
    """
    Analyzes predictions through a multi-stage filtering funnel and presents
    a clear, visual table of the entire process.
    Returns the final list of recommended trades (dicts).
    """
    logger.info("=== Starting Recommendation Generation Funnel ===")
    rec_tag = str(getattr(config.General, "REC_TAG", mode) or mode)
    diagnostics_steps = []
    consensus_probe = []
    step1_reject_counts = {
        "tradeable_predictions_n": 0,
        "missing_price": 0,
        "direction_inconsistent": 0,
        "net_alpha_only": 0,
        "pi_guard_only": 0,
        "both_fail": 0,
        "passed": 0,
    }
    step1_signal_counts = {
        "total_long": 0,
        "total_short": 0,
        "total_neutral": 0,
        "passed_long": 0,
        "passed_short": 0,
        "passed_neutral": 0,
        "failed_step1_long": 0,
        "failed_step1_short": 0,
        "failed_step1_neutral": 0,
        "blocked_live_short_execution": 0,
    }
    step1_signal_reject_counts = {
        "long": {
            "missing_price": 0,
            "direction_inconsistent": 0,
            "net_alpha_only": 0,
            "pi_guard_only": 0,
            "both_fail": 0,
            "passed": 0,
        },
        "short": {
            "missing_price": 0,
            "direction_inconsistent": 0,
            "net_alpha_only": 0,
            "pi_guard_only": 0,
            "both_fail": 0,
            "passed": 0,
            "blocked_live_short_execution": 0,
        },
        "neutral": {
            "missing_price": 0,
            "direction_inconsistent": 0,
            "net_alpha_only": 0,
            "pi_guard_only": 0,
            "both_fail": 0,
            "passed": 0,
        },
    }
    step1_signal_stats_raw = {
        "long": {"net_alpha": [], "pi_guard_shortfall": [], "step1_score": []},
        "short": {"net_alpha": [], "pi_guard_shortfall": [], "step1_score": []},
        "neutral": {"net_alpha": [], "pi_guard_shortfall": [], "step1_score": []},
    }
    recommendation_csv_path = None
    minrec_applied = False
    minrec_watch_only = False
    final_stage_label = "Final Recommendations"
    
    if not predictions:
        logger.warning("Recommender received no predictions to analyze.")
        min_live = int(getattr(config.Recommender, "MIN_RECOMMENDATIONS_LIVE", 0) or 0)
        if mode == "live" and min_live > 0:
            logger.error("[MinRec] No predictions available; emitting watch-only fallback output.")
            final_recommendations = [_synthesize_watch_only_recommendation([], reason="no_predictions")]
            # Best-effort save, to match normal run artifacts.
            try:
                df = pd.DataFrame(final_recommendations)
                if "pattern" in df.columns:
                    df["pattern"] = df["pattern"].apply(lambda p: ",".join([f"{float(x):+.4f}" for x in p]))
                if emit_artifacts:
                    output_dir = "recommendations"
                    os.makedirs(output_dir, exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    tag = config.General.REC_TAG
                    filename = os.path.join(output_dir, f"recs_{tag}_{timestamp}.csv")
                    df.to_csv(filename, index=False, encoding="utf-8-sig")
                    logger.info(f"[MinRec] Recommendations successfully saved to {filename}")
            except Exception as e:
                logger.error(f"[MinRec] Failed to save recommendations to CSV: {e}")
            logger.info(f"[RecommenderTrace] returning no-predictions fallback n={len(final_recommendations)}")
            return final_recommendations
        return []

    # --- [Step 0] Filter Non-Tradeable Markets ---
    tradeable_markets = get_tradeable_markets()
    if tradeable_markets:
        original_count = len(predictions)
        predictions = [p for p in predictions if p.get('market') in tradeable_markets]
        filtered_count = original_count - len(predictions)
        if filtered_count > 0:
            logger.info(f"[Tradeable Filter] Removed {filtered_count} non-tradeable markets (e.g., delisted coins)")
        logger.info(f"[Tradeable Filter] {len(predictions)} markets remain after validation")
    else:
        logger.warning("[Tradeable Filter] Could not fetch tradeable markets, skipping validation")
    step1_reject_counts["tradeable_predictions_n"] = int(len(predictions or []))

    # Current price fallback: if API is offline, predictor may not have current_price.
    # Use DB latest close so we can still produce at least a watch-only output.
    try:
        need_price = []
        for p in predictions:
            cp = p.get("current_price")
            if cp is None:
                need_price.append(p.get("market"))
                continue
            try:
                if float(cp) <= 0:
                    need_price.append(p.get("market"))
            except Exception:
                need_price.append(p.get("market"))
        need_price = [m for m in need_price if m]
        if need_price:
            fb = _get_latest_close_map_from_db(need_price)
            for p in predictions:
                if p.get("market") in fb and (p.get("current_price") is None or float(p.get("current_price") or 0) <= 0):
                    p["current_price"] = fb[p.get("market")]
    except Exception:
        pass

    # --- [Step 1] Initial Predictions ---
    funnel_data = []
    for pred in predictions:
        current_price = pred.get('current_price')
        if current_price is None or current_price <= 0:
            step1_reject_counts["missing_price"] += 1
            signal_key = str(pred.get("signal", "") or "").strip().lower()
            if signal_key in step1_signal_reject_counts:
                step1_signal_reject_counts[signal_key]["missing_price"] += 1
            continue
        pattern = _coerce_pattern_array(pred.get('predicted_pattern'))
        expected_return = float(np.prod(1 + pattern) - 1)
        signal = 'Long' if expected_return > 0 else ('Short' if expected_return < 0 else 'Neutral')
        signal_key = str(signal).strip().lower()

        # --- [NEW] 방향 일관성 체크 (예측 horizon의 약 2/3 이상이 같은 방향) ---
        n_positive = int(np.sum(pattern > 0))
        n_negative = int(np.sum(pattern < 0))
        direction_consistency = max(n_positive, n_negative) / len(pattern) if len(pattern) > 0 else 0

        if direction_consistency < 0.55:
            logger.info(f"[DIAG] {pred['market']}: direction_inconsistent ({n_positive}+/{n_negative}-), pattern={[round(float(x),5) for x in pattern]}")
            step1_reject_counts["direction_inconsistent"] += 1
            if signal_key in step1_signal_reject_counts:
                step1_signal_reject_counts[signal_key]["direction_inconsistent"] += 1
            continue  # 모순된 신호 스킵

        # --- 이상 예측 하드 필터 (uncertainty bypass 이전에 적용) ---
        _uncertainty_raw = float(pred.get('uncertainty', 0) or 0)
        if abs(expected_return) > 0.5 or _uncertainty_raw > 3.0:
            logger.info(f"[DIAG] {pred['market']}: anomaly_rejected (|return|={abs(expected_return):.4f}, uncertainty={_uncertainty_raw:.3f})")
            step1_reject_counts["direction_inconsistent"] += 1
            continue

        # --- gate_value 필터: Short 신호에서 gate < 0.25이면 예측력 없음 ---
        _gate_val = float(pred.get('gate_value', 0.5) or 0.5)
        if signal_key == 'short' and _gate_val < 0.25:
            logger.info(f"[DIAG] {pred['market']}: low_gate_short_rejected (gate={_gate_val:.3f})")
            step1_reject_counts["direction_inconsistent"] += 1
            continue

        if signal_key in {"long", "short", "neutral"}:
            step1_signal_counts[f"total_{signal_key}"] += 1

        # PI_80 from predictor (10th/90th percentile of MC-Dropout cumulative returns)
        pi_low_80 = float(pred.get('pi_low_80', expected_return))
        pi_high_80 = float(pred.get('pi_high_80', expected_return))

        pi_low_floor = _resolve_pi_low_80_floor(mode=mode, strategy=pred.get("strategy"))
        directional_metrics = _compute_directional_trade_metrics(
            expected_return=expected_return,
            pi_low_80=pi_low_80,
            pi_high_80=pi_high_80,
            pi_low_floor=pi_low_floor,
        )

        directional_return = float(directional_metrics["directional_return"])
        directional_pi_guard = float(directional_metrics["directional_pi_guard"])
        step1_cost_budget = _resolve_step1_cost_budget(mode=mode, strategy=pred.get("strategy"), signal=signal)
        net_alpha = directional_return - step1_cost_budget
        step1_score_metrics = _compute_step1_score(
            net_alpha=net_alpha,
            directional_pi_guard=directional_pi_guard,
            pi_guard_floor=pi_low_floor,
            mode=mode,
            strategy=pred.get("strategy"),
            signal=signal,
        )
        step1_score = float(step1_score_metrics["step1_score"])
        pi_guard_shortfall = float(step1_score_metrics["pi_guard_shortfall"])
        pi_guard_penalty = float(step1_score_metrics["pi_guard_penalty"])

        if signal_key in {"long", "short", "neutral"}:
            step1_signal_stats_raw[signal_key]["net_alpha"].append(float(net_alpha))
            step1_signal_stats_raw[signal_key]["pi_guard_shortfall"].append(float(pi_guard_shortfall))
            step1_signal_stats_raw[signal_key]["step1_score"].append(float(step1_score))

        # Step 1 uses a risk-adjusted score instead of a brittle hard PI veto.
        failed_net_alpha = net_alpha <= 0
        failed_pi_guard = pi_guard_shortfall > 0
        if step1_score <= 0:
            if signal_key in {"long", "short", "neutral"}:
                step1_signal_counts[f"failed_step1_{signal_key}"] += 1
                step1_signal_reject_counts[signal_key]["passed"] += 0
            if failed_net_alpha and failed_pi_guard:
                step1_reject_counts["both_fail"] += 1
                if signal_key in step1_signal_reject_counts:
                    step1_signal_reject_counts[signal_key]["both_fail"] += 1
            elif failed_net_alpha:
                step1_reject_counts["net_alpha_only"] += 1
                if signal_key in step1_signal_reject_counts:
                    step1_signal_reject_counts[signal_key]["net_alpha_only"] += 1
            else:
                step1_reject_counts["pi_guard_only"] += 1
                if signal_key in step1_signal_reject_counts:
                    step1_signal_reject_counts[signal_key]["pi_guard_only"] += 1
            logger.info(
                f"[DIAG] {pred['market']}: step1_fail signal={signal}, score={step1_score:.4f}, "
                f"net_alpha={net_alpha:.4f}, pi_guard={directional_pi_guard:.4f}, "
                f"floor={pi_low_floor:.4f}, shortfall={pi_guard_shortfall:.4f}, pattern={[round(float(x),5) for x in pattern]}"
            )
            continue
        step1_reject_counts["passed"] += 1
        if signal_key in {"long", "short", "neutral"}:
            step1_signal_counts[f"passed_{signal_key}"] += 1
            step1_signal_reject_counts[signal_key]["passed"] += 1

        # --- Composite Position Sizing (Confidence × Volatility) ---
        volatility = np.std(pattern) if len(pattern) > 0 else 0.01
        
        # --- Confidence: prefer calibrated confidence_head output ---
        model_conf = pred.get('model_confidence')
        if model_conf is not None:
            confidence = float(model_conf)
        else:
            # Fallback: heuristic from uncertainty
            MIN_UNCERTAINTY = 0.5
            uncertainty = max(pred['uncertainty'], MIN_UNCERTAINTY)
            confidence = 1 / (1 + uncertainty)
        
        # Composite formula:
        # - Higher confidence = larger position
        # - Higher volatility = smaller position
        base_position = 0.10  # 10% base
        max_position = 0.20   # 20% max
        min_position = 0.03   # 3% min
        
        # confidence_factor: 0.5 ~ 1.0 range normalized
        confidence_factor = max(0.3, min(1.0, confidence))
        
        # volatility_factor: inverse relationship (high vol = smaller factor)
        volatility_factor = 1 / (1 + volatility * 5)
        
        # Composite position size
        position_size = base_position * confidence_factor * volatility_factor
        position_size = max(min_position, min(max_position, position_size))

        funnel_data.append({
            'market': pred['market'],
            'expected_return': expected_return,
            'net_alpha': net_alpha,
            'pi_low_80': pi_low_80,
            'pi_high_80': pi_high_80,
            'pi_low_80_floor': pi_low_floor,
            'directional_return': directional_return,
            'directional_pi_guard': directional_pi_guard,
            'step1_score': step1_score,
            'decision_score': _compute_trade_quality_score({
                'mode': mode,
                'strategy': pred.get('strategy', 'trending'),
                'signal': signal,
                'step1_score': step1_score,
                'expected_return': expected_return,
                'confidence': confidence,
                'consensus_score': pred.get('consensus_score', 0.6),
            }),
            'pi_guard_shortfall': pi_guard_shortfall,
            'pi_guard_penalty': pi_guard_penalty,
            'attention_top3': pred.get('attention_top3', []),
            'prototype_match': pred.get('prototype_match', {}),
            'confidence': confidence,
            'uncertainty': pred['uncertainty'],
            'current_price': current_price,
            'mode': mode,
            'strategy': pred.get('strategy', 'trending'),
            'pattern': pattern,
            'status': 'Initial Candidate',
            'dtw_distance': 999.0,
            'signal': signal,
            'step1_cost_budget': step1_cost_budget,
            'volatility': volatility,
            'position_size': position_size,  # Dynamic position sizing
            # Gate Analysis (passed through for paper)
            'gate_value': pred.get('gate_value', 0.5),
            'consensus_score': pred.get('consensus_score', 0.6)
        })

    _log_recommendation_table("[Funnel Step 1] Initial Predictions", funnel_data)
    diagnostics_steps.append(_capture_funnel_snapshot("step1_initial", funnel_data))

    # --- [Step 1.5] Market Regime & Lead-Lag Filter ---
    # We combine Market Regime (Macro) with Lead-Lag (Micro) analysis
    market_index_df = get_market_index()
    regime_label = "Unknown"
    regime_known = False
    
    # 1. Macro Regime Check
    is_downtrend = False
    if not market_index_df.empty and len(market_index_df) > config.Recommender.REGIME_SMA_LONG:
        regime_known = True
        market_index_df['cumulative_index'] = (1 + market_index_df['market_index_return']).cumprod()
        market_index_df['sma_short'] = market_index_df['cumulative_index'].rolling(window=config.Recommender.REGIME_SMA_SHORT).mean()
        market_index_df['sma_long'] = market_index_df['cumulative_index'].rolling(window=config.Recommender.REGIME_SMA_LONG).mean()
        
        last_row = market_index_df.iloc[-1]
        is_downtrend = last_row['sma_short'] < last_row['sma_long']
        regime_label = "Downtrend" if is_downtrend else "Uptrend"
    
    # 2. Micro Lead-Lag Analysis
    logger.info(f"--- Lead-Lag Analysis (Leader: KRW-BTC) ---")
    
    if historical_data is not None and not historical_data.empty:
        btc_data = historical_data[historical_data['market'] == 'KRW-BTC']
    else:
        btc_data = None 

    for trade in funnel_data:
        if trade['status'] == 'Initial Candidate':
            signal = trade.get('signal', 'Neutral')
            market = trade['market']
            
            # --- Lead-Lag Check ---
            lead_lag_status = "Sync" # Default
            lag_hours = 0
            
            if btc_data is not None and not btc_data.empty and market != 'KRW-BTC':
                try:
                    target_data = historical_data[historical_data['market'] == market]
                    if not target_data.empty:
                        common_idx = btc_data.index.intersection(target_data.index)
                        if len(common_idx) > 48:
                            btc_series = btc_data.loc[common_idx]['close'].pct_change().fillna(0)
                            target_series = target_data.loc[common_idx]['close'].pct_change().fillna(0)
                            
                            corrs = []
                            for lag in range(5):
                                c = target_series.corr(btc_series.shift(lag))
                                corrs.append(c)
                            
                            max_corr_idx = np.argmax(corrs)
                            if max_corr_idx > 0 and corrs[max_corr_idx] > 0.3:
                                lead_lag_status = "Lagging"
                                lag_hours = max_corr_idx
                                trade['lead_lag_info'] = f"Lags BTC by {lag_hours}h (Corr: {corrs[max_corr_idx]:.2f})"
                except Exception as e:
                    logger.debug(f"Lead-lag calc failed for {market}: {e}")

            # --- Score Adjustment ---
            if lead_lag_status == "Lagging":
                trade['confidence'] = min(0.99, trade['confidence'] * 1.15) # 15% Boost
                trade['reason'] = f"Lagging Leader by {lag_hours}h"

            if regime_known and ((is_downtrend and signal == 'Short') or (not is_downtrend and signal == 'Long')):
                original_conf = trade['confidence']
                trade['confidence'] = min(1.0, original_conf * config.Recommender.TREND_CONFIDENCE_BOOST)
                trade['trend_alignment'] = 'Following'
                if 'reason' not in trade: trade['reason'] = "Trend Alignment"
            
            elif regime_known and ((is_downtrend and signal == 'Long') or (not is_downtrend and signal == 'Short')):
                trade['trend_alignment'] = 'Counter'
                if 'reason' not in trade: trade['reason'] = "Counter-Trend Opportunity"
            
            else:
                trade['trend_alignment'] = 'Neutral'

            # Refresh long-only decision score after trend alignment is known.
            trade['decision_score'] = _compute_trade_quality_score({
                'mode': mode,
                'strategy': trade.get('strategy'),
                'signal': trade.get('signal'),
                'step1_score': trade.get('step1_score', 0.0),
                'expected_return': trade.get('expected_return', 0.0),
                'confidence': trade.get('confidence', 0.0),
                'consensus_score': trade.get('consensus_score', 0.0),
                'trend_alignment': trade.get('trend_alignment'),
            })

    _log_recommendation_table(f"[Funnel Step 1.5] After Regime & Lead-Lag Analysis", funnel_data)
    diagnostics_steps.append(_capture_funnel_snapshot("step1_5_regime_leadlag", funnel_data))

    # --- [Step 2] Liquidity Filter ---
    all_markets = [p['market'] for p in funnel_data]
    current_time = reference_time or datetime.now(timezone.utc)
    trading_values = get_trading_values_for_markets(all_markets, end_time=current_time, hours=config.Recommender.LIQUIDITY_LOOKBACK_HOURS)
    threshold = config.Recommender.LIQUIDITY_THRESHOLDS.get(mode, config.Recommender.LIQUIDITY_THRESHOLDS['live'])
    
    for trade in funnel_data:
        if trade['status'] == 'Initial Candidate':
            market_value = trading_values.get(trade['market'], 0)
            if market_value < threshold:
                trade['status'] = f"Failed: Low Liquidity"

    _log_recommendation_table(f"[Funnel Step 2] After Liquidity Filter", [t for t in funnel_data if t['status'] == 'Initial Candidate' or 'Failed: Low Liquidity' in t['status']])
    diagnostics_steps.append(_capture_funnel_snapshot("step2_liquidity", funnel_data))

    # --- [Step 3] Minimum Expected Return Filtering ---
    min_signal_return = config.Recommender.MIN_SIGNAL_RETURN
    for trade in funnel_data:
        if trade['status'] == 'Initial Candidate':
            if abs(trade['expected_return']) < min_signal_return:
                trade['status'] = f"Failed: Low Return"
                continue
    
    _log_recommendation_table(f"[Funnel Step 3] After Expected Return Filter", [t for t in funnel_data if t['status'] == 'Initial Candidate' or 'Failed: Low Return' in t['status']])
    diagnostics_steps.append(_capture_funnel_snapshot("step3_expected_return", funnel_data))

    # --- [Step 3.5] Consensus Filter ---
    for trade in funnel_data:
        if trade['status'] == 'Initial Candidate':
            consensus_probe.append({
                'market': trade.get('market'),
                'strategy': trade.get('strategy'),
                'signal': trade.get('signal'),
                'trend_alignment': trade.get('trend_alignment'),
                'expected_return': trade.get('expected_return'),
                'net_alpha': trade.get('net_alpha'),
                'pi_low_80': trade.get('pi_low_80'),
                'pi_low_80_floor': trade.get('pi_low_80_floor'),
                'consensus_score': trade.get('consensus_score'),
                'gate_value': trade.get('gate_value'),
                'min_consensus_required': None,
                'passed_consensus': None,
            })
    for trade in funnel_data:
        if trade['status'] == 'Initial Candidate':
            consensus = float(trade.get('consensus_score', 0.0))
            min_consensus = _resolve_min_consensus(
                mode=mode,
                strategy=trade.get('strategy'),
                trend_alignment=trade.get('trend_alignment'),
            )
            trade['min_consensus_required'] = min_consensus
            for row in reversed(consensus_probe):
                if row.get('market') == trade.get('market') and row.get('min_consensus_required') is None:
                    row['min_consensus_required'] = min_consensus
                    row['passed_consensus'] = bool(consensus >= min_consensus)
                    break
            logger.info(f"[DIAG-CONS] {trade.get('market')}: consensus={consensus:.3f}, min={min_consensus:.3f}, mode={mode!r}")
            if consensus < min_consensus:
                trade['status'] = f"Failed: Low Consensus"

    _log_recommendation_table(
        f"[Funnel Step 3.5] After Consensus Filter",
        [t for t in funnel_data if t['status'] == 'Initial Candidate' or 'Failed: Low Consensus' in t['status']]
    )
    diagnostics_steps.append(_capture_funnel_snapshot("step3_5_consensus", funnel_data))

    # --- [Step 4] Uncertainty Filtering ---
    base_uncertainty_threshold = float(config.Recommender.UNCERTAINTY_THRESHOLD)
    effective_uncertainty_threshold = _compute_uncertainty_threshold(funnel_data, base_uncertainty_threshold)

    active_before_unc = [t for t in funnel_data if t.get('status') == 'Initial Candidate']
    allow_single_bypass = bool(getattr(config.Recommender, "ALLOW_SINGLE_CANDIDATE_UNCERTAINTY_BYPASS", True))
    if allow_single_bypass and len(active_before_unc) == 1:
        logger.warning(
            "[Uncertainty] Single-candidate bypass enabled: "
            "skipping uncertainty filter to avoid unnecessary MinRec forcing."
        )
    else:
        for trade in funnel_data:
            if trade['status'] == 'Initial Candidate':
                if trade.get('trend_alignment') == 'Counter':
                    adjusted_threshold = effective_uncertainty_threshold * config.Recommender.COUNTER_TREND_UNCERTAINTY_MULTIPLIER
                    if trade['uncertainty'] >= adjusted_threshold:
                        trade['status'] = f"Failed: High Uncertainty (Counter-Trend)"
                else:
                    if trade['uncertainty'] >= effective_uncertainty_threshold:
                        trade['status'] = f"Failed: High Uncertainty"

    _log_recommendation_table(f"[Funnel Step 4] After Uncertainty Filter", [t for t in funnel_data if t['status'] == 'Initial Candidate' or 'Failed: High Uncertainty' in t['status']])
    diagnostics_steps.append(_capture_funnel_snapshot("step4_uncertainty", funnel_data))

    # --- [Step 5] DTW Pattern Filtering ---
    success_patterns = get_historical_success_patterns()
    if success_patterns.any():
        for trade in funnel_data:
            if trade['status'] == 'Initial Candidate':
                min_dist = min([get_pattern_similarity(trade['pattern'], sp) for sp in success_patterns])
                trade['dtw_distance'] = min_dist
                if min_dist >= config.Recommender.DTW_THRESHOLD:
                    trade['status'] = f"Failed: Low Similarity"
    else:
        logger.warning("No success patterns loaded, skipping DTW filter.")

    _log_recommendation_table(f"[Funnel Step 5] After DTW Filter", [t for t in funnel_data if t['status'] == 'Initial Candidate' or 'Failed: Low Similarity' in t['status']])
    diagnostics_steps.append(_capture_funnel_snapshot("step5_dtw", funnel_data))
    
    # --- Final Selection ---
    final_recommendations = []
    live_short_execution_allowed = _live_short_execution_allowed(mode)
    
    # 1. Gather success candidates
    for trade in funnel_data:
        if trade['status'] == 'Initial Candidate':
            if not live_short_execution_allowed and str(trade.get('signal', '')).lower() == 'short':
                step1_signal_counts["blocked_live_short_execution"] += 1
                trade['status'] = 'Watch (Short Bias)'
                trade['position_size'] = 0.0
                prev_reason = str(trade.get('fallback_reason', '') or '')
                note = "spot_live_short_execution_disabled"
                trade['fallback_reason'] = (prev_reason + " | " + note).strip(" |") if prev_reason else note
                continue
            trade['status'] = 'Recommended'
            final_recommendations.append(trade)

    # 2. Optional Forced Top-K Fallback (disabled by default for safety realism)
    forced_enabled = getattr(config.Recommender, "FORCED_TOPK_ENABLED", False)
    if mode == 'backtest':
        forced_enabled = getattr(config.Recommender, "FORCED_TOPK_BACKTEST_ENABLED", forced_enabled)

    if len(final_recommendations) < min_k and forced_enabled:
        logger.info(f"Only {len(final_recommendations)} recommendations found. Activating Forced Top-K Fallback to reach {min_k}...")

        excluded_tokens = tuple(getattr(config.Recommender, "FORCED_TOPK_EXCLUDE_FAILED_REASONS", ()))
        candidates = [
            t for t in funnel_data
            if t['status'] != 'Recommended' and not any(token in t['status'] for token in excluded_tokens)
        ]
        # Sort key: Confidence DESC, then Abs(Return) DESC
        candidates.sort(key=_trade_sort_key, reverse=True)

        needed = min_k - len(final_recommendations)
        for i in range(min(needed, len(candidates))):
            trade = candidates[i]
            original_status = trade['status']
            trade['status'] = f"Forced (Low Conf)"
            trade['fallback_reason'] = original_status
            if 'reason' not in trade:
                trade['reason'] = "Forced Fallback (Top rank)"
            final_recommendations.append(trade)
            logger.info(f"  -> Forced inclusion: {trade['market']} (Conf: {trade['confidence']:.2%}, Orig: {original_status})")
    elif len(final_recommendations) < min_k:
        logger.info(
            f"Only {len(final_recommendations)} recommendations found. "
            f"Forced Top-K Fallback is disabled for mode='{mode}'."
        )

    # Limit to min_k if we have too many? No, user wants Top 3 "at least" or exactly?
    # User said "Top 3". So we should slice to Top 3 if we have more.
    # But wait, if we have 5 valid ones, should we hide 2?
    # User said: "Trending Top 3... Pattern Top 3".
    # So yes, let's limit to min_k (which effectively acts as 'target_k').
    
    final_recommendations.sort(key=_trade_sort_key, reverse=True)
    final_recommendations = final_recommendations[:min_k]

    # Ops override: force watch-only outputs (no position sizing) for safety.
    runtime_watch_only = bool(getattr(config.Recommender, "RUNTIME_WATCH_ONLY", False))
    if mode == "live" and runtime_watch_only and final_recommendations:
        for t in final_recommendations:
            prev = str(t.get("status", "") or "")
            t["position_size"] = 0.0
            if not str(t.get("status", "")).startswith("Watch"):
                t["status"] = "Watch (Runtime)"
            fr = str(t.get("fallback_reason", "") or "")
            note = f"runtime_watch_only=1 (prev_status='{prev}')"
            t["fallback_reason"] = (fr + " | " + note).strip(" |") if fr else note

    if final_recommendations:
        _log_recommendation_table(f"Final Recommendations (Top {min_k})", final_recommendations)
        final_stage_label = f"Final Recommendations (Top {min_k})"
        
        # --- Save to CSV ---
        df = pd.DataFrame(final_recommendations)
        df['pattern'] = df['pattern'].apply(lambda p: ','.join([f'{x:+.4f}' for x in p]))
        
        cols_to_save = [
            'market', 'signal', 'strategy', 'expected_return', 'net_alpha',
            'pi_low_80', 'pi_high_80', 'attention_top3', 'prototype_match',
            'confidence', 'consensus_score', 'trend_alignment', 'decision_score', 'gate_value',
            'position_size', 'volatility', 'dtw_distance',
            'current_price', 'pattern', 'reason', 'status', 'fallback_reason'
        ]
        # Add 'reason' to cols_to_save if present
        cols_present = [c for c in cols_to_save if c in df.columns]
        df = df[cols_present]

        if emit_artifacts:
            output_dir = 'recommendations'
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            tag = rec_tag
            if 'strategy' in df.columns and len(df['strategy'].unique()) == 1:
                # Avoid duplicated tags like "intraday_intraday".
                st = str(df['strategy'].iloc[0])
                tag_parts = set(str(tag).split("_"))
                if st and st not in tag_parts:
                    tag += f"_{st}"  # Append strategy name to file if uniform
            
            filename = os.path.join(output_dir, f"recs_{tag}_{timestamp}.csv")
            try:
                df.to_csv(filename, index=False, encoding='utf-8-sig')
                logger.info(f"Recommendations successfully saved to {filename}")
                recommendation_csv_path = filename
            except Exception as e:
                logger.error(f"Failed to save recommendations to CSV: {e}")
    else:
        logger.warning("No recommendations remained after all filtering stages.")

    # --- Enforce Minimum Live Recommendations (>=1) ---
    min_live = int(getattr(config.Recommender, "MIN_RECOMMENDATIONS_LIVE", 0) or 0)
    if mode == 'live' and min_live > 0 and len(final_recommendations) < min_live:
        allow_low_liq = bool(getattr(config.Recommender, "MIN_REC_FALLBACK_ALLOW_LOW_LIQUIDITY", False))
        minrec_mode = str(getattr(config.Recommender, "MIN_REC_MODE", "trade") or "trade").strip().lower()
        disallow_neutral = bool(getattr(config.Recommender, "MIN_REC_DISALLOW_NEUTRAL", True))
        min_abs_exp = float(getattr(config.Recommender, "MIN_REC_MIN_ABS_EXPECTED_RETURN", 0.0) or 0.0)
        min_cons = float(getattr(config.Recommender, "MIN_REC_MIN_CONSENSUS_SCORE", 0.0) or 0.0)
        max_unc_mult = float(getattr(config.Recommender, "MIN_REC_MAX_UNCERTAINTY_MULTIPLIER", 999.0) or 999.0)
        allow_watch = bool(getattr(config.Recommender, "MIN_REC_ALLOW_WATCH_ONLY_FALLBACK", True))

        # NOTE: uncertainty_threshold in the funnel may be adaptive; if not available, use base threshold.
        # We approximate a safe bound using base threshold * multiplier.
        try:
            base_unc = float(getattr(config.Recommender, "UNCERTAINTY_THRESHOLD", 0.0) or 0.0)
        except Exception:
            base_unc = 0.0
        max_unc = base_unc * max_unc_mult if base_unc > 0 else float("inf")

        # Prefer candidates that passed liquidity; only relax other filters (DTW/uncertainty/consensus).
        def _liquidity_ok(t: dict) -> bool:
            status = str(t.get('status', ''))
            if allow_low_liq:
                return True
            return "Low Liquidity" not in status

        def _minrec_candidate_ok(t: dict) -> bool:
            if disallow_neutral and str(t.get("signal", "")).lower() == "neutral":
                return False
            if not live_short_execution_allowed and str(t.get("signal", "")).lower() == "short":
                return False
            try:
                if abs(float(t.get("expected_return", 0.0))) < float(min_abs_exp):
                    return False
            except Exception:
                return False
            try:
                if float(t.get("consensus_score", 0.0)) < float(min_cons):
                    return False
            except Exception:
                return False
            # Uncertainty: allow if below safe ceiling; if missing, allow.
            try:
                u = float(t.get("uncertainty", float("nan")))
                if np.isfinite(u) and u > max_unc:
                    return False
            except Exception:
                pass
            return True

        fallback_pool_all = list(funnel_data)
        fallback_pool_liq = [t for t in fallback_pool_all if _liquidity_ok(t)]

        # Build a compact reason summary for "why no safe trade was available".
        def _minrec_stats(pool: list) -> dict:
            total = len(pool)
            non_neutral = 0
            abs_ok = 0
            cons_ok = 0
            unc_ok = 0
            safe = 0
            for t in pool:
                sig = str(t.get("signal", "")).lower()
                if (not disallow_neutral) or sig != "neutral":
                    non_neutral += 1
                try:
                    if abs(float(t.get("expected_return", 0.0))) >= float(min_abs_exp):
                        abs_ok += 1
                except Exception:
                    pass
                try:
                    if float(t.get("consensus_score", 0.0)) >= float(min_cons):
                        cons_ok += 1
                except Exception:
                    pass
                try:
                    u = float(t.get("uncertainty", float("nan")))
                    if (not np.isfinite(u)) or (u <= max_unc):
                        unc_ok += 1
                except Exception:
                    unc_ok += 1
                if _minrec_candidate_ok(t):
                    safe += 1
            return {
                "pool": total,
                "non_neutral": non_neutral,
                "abs_ok": abs_ok,
                "cons_ok": cons_ok,
                "unc_ok": unc_ok,
                "safe": safe,
            }

        liq_stats = _minrec_stats(fallback_pool_liq)

        # 1) Try to force a trade only if allowed and safe enough.
        picked = None
        if minrec_mode == "trade":
            safe_pool = [t for t in fallback_pool_liq if _minrec_candidate_ok(t)]
            if safe_pool:
                picked = sorted(safe_pool, key=_trade_sort_key, reverse=True)[0]

        # 2) If we cannot force a safe trade, optionally output a watch-only item to satisfy ">=1 output".
        watch_only = False
        if picked is None and allow_watch:
            # Pick best liquid candidate as watch-only (or best overall if none are liquid).
            pool = fallback_pool_liq if fallback_pool_liq else fallback_pool_all
            if pool:
                # Prefer non-neutral watch items if configured.
                if disallow_neutral:
                    pool_non_neutral = [t for t in pool if str(t.get("signal", "")).lower() != "neutral"]
                    if pool_non_neutral:
                        pool = pool_non_neutral
                picked = sorted(pool, key=_trade_sort_key, reverse=True)[0]
                watch_only = True
            else:
                # Absolute last-resort: even if the funnel produced zero candidates, emit one watch-only item.
                picked = _synthesize_watch_only_recommendation(predictions, reason="minrec_no_candidates")
                watch_only = True

        if picked is not None:
            original_status = picked.get('status', '')
            if watch_only or minrec_mode == "watch":
                minrec_applied = True
                minrec_watch_only = True
                new_reason = (
                    f"MinRec watch-only from status='{original_status}'. "
                    f"safe_trade_candidates={liq_stats.get('safe', 0)}/{liq_stats.get('pool', 0)} "
                    f"(non_neutral={liq_stats.get('non_neutral', 0)}, abs_ok={liq_stats.get('abs_ok', 0)}, "
                    f"cons_ok={liq_stats.get('cons_ok', 0)}, unc_ok={liq_stats.get('unc_ok', 0)})"
                )
                prev_reason = str(picked.get("fallback_reason", "") or "")
                picked["fallback_reason"] = (prev_reason + " | " + new_reason).strip(" |") if prev_reason else new_reason
                picked['status'] = "Watch (MinRec)"
                picked['position_size'] = 0.0
                logger.warning(
                    f"[MinRec] Watch-only fallback: picked {picked.get('market')} (was: {original_status})"
                )
            else:
                minrec_applied = True
                picked['fallback_reason'] = (
                    f"MinRec enforcement from status='{original_status}'. "
                    f"safe_trade_candidates={liq_stats.get('safe', 0)}/{liq_stats.get('pool', 0)}"
                )
                picked['status'] = "Forced (Min 1)"
                picked['position_size'] = min(float(picked.get('position_size', 0.03)), 0.03)
                logger.warning(
                    f"[MinRec] Enforcing at least {min_live} live recommendation(s): "
                    f"picked {picked.get('market')} (was: {original_status})"
                )

            final_recommendations = [picked]

            # Apply ops watch-only override after MinRec, too.
            runtime_watch_only = bool(getattr(config.Recommender, "RUNTIME_WATCH_ONLY", False))
            if runtime_watch_only:
                for t in final_recommendations:
                    prev = str(t.get("status", "") or "")
                    t["position_size"] = 0.0
                    if not str(t.get("status", "")).startswith("Watch"):
                        t["status"] = "Watch (Runtime)"
                    fr = str(t.get("fallback_reason", "") or "")
                    note = f"runtime_watch_only=1 (prev_status='{prev}')"
                    t["fallback_reason"] = (fr + " | " + note).strip(" |") if fr else note

            _log_recommendation_table(
                "Final Recommendations (MinRec Enforced)" if not watch_only else "Final Recommendations (MinRec Watch-Only)",
                final_recommendations
            )
            final_stage_label = "Final Recommendations (MinRec Enforced)" if not watch_only else "Final Recommendations (MinRec Watch-Only)"

            # Save enforced output as well (same pipeline as normal save)
            df = pd.DataFrame(final_recommendations)
            df['pattern'] = df['pattern'].apply(lambda p: ','.join([f'{x:+.4f}' for x in p]))
            cols_to_save = [
                'market', 'signal', 'strategy', 'expected_return', 'net_alpha',
                'pi_low_80', 'pi_high_80', 'attention_top3', 'prototype_match',
                'confidence', 'consensus_score', 'trend_alignment', 'decision_score', 'gate_value',
                'position_size', 'volatility', 'dtw_distance',
                'current_price', 'pattern', 'reason', 'status', 'fallback_reason'
            ]
            cols_present = [c for c in cols_to_save if c in df.columns]
            df = df[cols_present]

            if emit_artifacts:
                output_dir = 'recommendations'
                os.makedirs(output_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                tag = rec_tag
                if 'strategy' in df.columns and len(df['strategy'].unique()) == 1:
                    st = str(df['strategy'].iloc[0])
                    tag_parts = set(str(tag).split("_"))
                    if st and st not in tag_parts:
                        tag += f"_{st}"
                filename = os.path.join(output_dir, f"recs_{tag}_{timestamp}.csv")
                try:
                    df.to_csv(filename, index=False, encoding='utf-8-sig')
                    logger.info(f"[MinRec] Recommendations successfully saved to {filename}")
                    recommendation_csv_path = filename
                except Exception as e:
                    logger.error(f"[MinRec] Failed to save recommendations to CSV: {e}")
        else:
            logger.error(
                f"[MinRec] Could not enforce min recommendations because all candidates failed liquidity "
                f"(allow_low_liquidity={allow_low_liq})."
            )

    diagnostics_steps.append(_capture_funnel_snapshot("final", funnel_data))
    diagnostics_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "rec_tag": rec_tag,
        "min_k": int(min_k),
        "predictions_n": len(predictions or []),
        "step1_reject_counts": step1_reject_counts,
        "step1_signal_counts": step1_signal_counts,
        "step1_signal_reject_counts": step1_signal_reject_counts,
        "step1_signal_stats": {
            signal: {
                metric: _summarize_numeric_list(values)
                for metric, values in metrics.items()
            }
            for signal, metrics in step1_signal_stats_raw.items()
        },
        "funnel_steps": diagnostics_steps,
        "thresholds": {
            "min_signal_return": getattr(config.Recommender, "MIN_SIGNAL_RETURN", None),
            "pi_low_80_min_live": getattr(config.Recommender, "PI_LOW_80_MIN_LIVE", None),
            "pi_low_80_min_intraday": getattr(config.Recommender, "PI_LOW_80_MIN_INTRADAY", None),
            "step1_pi_guard_penalty": getattr(config.Recommender, "STEP1_PI_GUARD_PENALTY", None),
            "step1_pi_guard_penalty_intraday_long": getattr(config.Recommender, "STEP1_PI_GUARD_PENALTY_INTRADAY_LONG", None),
            "trade_fee_per_leg": getattr(config.Recommender, "TRADE_FEE_PER_LEG", None),
            "slippage_per_leg": getattr(config.Recommender, "SLIPPAGE_PER_LEG", None),
            "live_intraday_long_step1_cost_legs": getattr(config.Recommender, "LIVE_INTRADAY_LONG_STEP1_COST_LEGS", None),
            "min_consensus_score": getattr(config.Recommender, "MIN_CONSENSUS_SCORE", None),
            "min_consensus_score_intraday": getattr(config.Recommender, "MIN_CONSENSUS_SCORE_INTRADAY", None),
            "counter_trend_min_consensus_score": getattr(config.Recommender, "COUNTER_TREND_MIN_CONSENSUS_SCORE", None),
            "counter_trend_min_consensus_score_intraday": getattr(config.Recommender, "COUNTER_TREND_MIN_CONSENSUS_SCORE_INTRADAY", None),
            "uncertainty_threshold": getattr(config.Recommender, "UNCERTAINTY_THRESHOLD", None),
            "effective_uncertainty_threshold": effective_uncertainty_threshold,
        },
        "consensus_probe": sorted(
            consensus_probe,
            key=lambda row: float(row.get("consensus_score", 0.0) or 0.0),
            reverse=True,
        )[:20],
        "final": {
            "stage_label": final_stage_label,
            "recommendations_n": len(final_recommendations or []),
            "recommendation_markets": [t.get("market") for t in (final_recommendations or [])],
            "recommendation_statuses": [t.get("status") for t in (final_recommendations or [])],
            "recommendation_path": recommendation_csv_path,
            "minrec_applied": bool(minrec_applied),
            "minrec_watch_only": bool(minrec_watch_only),
        },
    }
    diagnostics_path = None
    if emit_artifacts:
        diagnostics_path = _write_recommender_diagnostics(diagnostics_payload, rec_tag=rec_tag)
        diagnostics_payload["path"] = diagnostics_path
        _set_last_run_diagnostics(diagnostics_payload, diagnostics_path)
    elif collect_diagnostics:
        diagnostics_payload["path"] = None
        _set_last_run_diagnostics(diagnostics_payload, None)

    logger.info("======================================================")

    # --- [NEW] Save Gate Values for Real-time Dashboard ---
    # Backtest runs should not pollute live gate streams.
    try:
        if mode == "backtest" or not emit_artifacts:
            logger.info(f"[RecommenderTrace] returning backtest recommendations n={len(final_recommendations)}")
            return final_recommendations
        # Collect gate values from ALL predictions (not just recommendations)
        gate_data = []
        current_ts = datetime.now().isoformat()
        
        for p in predictions:
            if 'gate_value' in p:
                gate_data.append({
                    'timestamp': current_ts,
                    'market': p.get('market', 'unknown'),
                    'gate_value': p['gate_value']
                })
        
        if gate_data:
            analysis_dir = 'analysis'
            os.makedirs(analysis_dir, exist_ok=True)
            gate_file = os.path.join(analysis_dir, 'gate_values.csv')
            
            gate_df = pd.DataFrame(gate_data)
            
            # Append if exists, else create
            if os.path.exists(gate_file):
                gate_df.to_csv(gate_file, mode='a', header=False, index=False)
            else:
                gate_df.to_csv(gate_file, mode='w', header=True, index=False)
                
            logger.info(f"Saved {len(gate_data)} gate values to {gate_file}")
            
    except Exception as e:
        logger.warning(f"Failed to save gate values: {e}")

    logger.info(f"[RecommenderTrace] returning recommendations n={len(final_recommendations)}")
    return final_recommendations
