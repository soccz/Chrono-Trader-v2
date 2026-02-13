import os
import glob
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Dict, Tuple, Union, Any

import numpy as np
import pandas as pd

from utils.logger import logger
from utils.config import config
from data.database import (
    get_latest_db_timestamp,
    get_top_markets_by_trading_value,
    load_data,
)


def _dedupe_preserve(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if not x:
            continue
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _load_holdings_from_state(path: str = "portfolio_state.json", max_n: int = 5) -> List[str]:
    try:
        import json

        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f) or {}
        positions = d.get("positions") or {}
        markets = list(positions.keys())
        # Deterministic order: newest first if available
        def _pos_ts(m: str) -> float:
            try:
                ts = positions[m].get("last_updated") or positions[m].get("opened_at")
                return datetime.fromisoformat(ts).timestamp() if ts else 0.0
            except Exception:
                return 0.0

        markets.sort(key=_pos_ts, reverse=True)
        return markets[: max(0, int(max_n))]
    except Exception as e:
        logger.debug(f"[Select] Failed to load holdings from {path}: {e}")
        return []


def _load_recent_recs(tag_prefix: str, top_n: int = 3) -> List[str]:
    """
    Load most recent recommendation CSV matching a tag prefix (e.g., 'intraday', 'morning').
    Returns up to top_n markets from that file.
    """
    try:
        pattern = os.path.join("recommendations", f"recs_{tag_prefix}_*.csv")
        files = glob.glob(pattern)
        if not files:
            return []
        files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        latest = files[0]
        df = pd.read_csv(latest)
        if "market" not in df.columns or df.empty:
            return []
        mkts = df["market"].dropna().astype(str).tolist()
        return mkts[: max(0, int(top_n))]
    except Exception as e:
        logger.debug(f"[Select] Failed to load recent recs for tag={tag_prefix}: {e}")
        return []


def _compute_candidate_stats(
    markets: List[str],
    end_time: datetime,
    lookback_hours: int = 168,
) -> pd.DataFrame:
    """
    Compute per-market stats from DB for selection:
    - tv_24h, tv_6h, vol_24h, abs_ret_6h, tv_surge
    - corr_btc over lookback_hours
    - last_ts, lag_h
    """
    if not markets:
        return pd.DataFrame()

    # Ensure BTC exists for corr reference.
    markets = _dedupe_preserve(["KRW-BTC"] + markets)
    start_time = end_time - pd.Timedelta(hours=int(lookback_hours))

    placeholders = ", ".join("?" for _ in markets)
    q = (
        "SELECT timestamp, market, close, volume "
        f"FROM crypto_data WHERE market IN ({placeholders}) AND timestamp BETWEEN ? AND ?"
    )
    params = list(markets) + [
        start_time.strftime("%Y-%m-%dT%H:%M:%S"),
        end_time.strftime("%Y-%m-%dT%H:%M:%S"),
    ]
    df = load_data(q, params=params)
    if df.empty:
        return pd.DataFrame()

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "market", "close", "volume"])
    if df.empty:
        return pd.DataFrame()

    df = df.sort_values(["timestamp", "market"])
    now_utc = datetime.now(timezone.utc)

    last_ts = df.groupby("market")["timestamp"].max()
    lag_h = (pd.Timestamp(now_utc) - last_ts).dt.total_seconds() / 3600.0

    # 24h window stats
    df24 = df[df["timestamp"] >= (pd.Timestamp(end_time) - pd.Timedelta(hours=24))]
    if df24.empty:
        return pd.DataFrame()

    tv_24h = (df24["close"] * df24["volume"]).groupby(df24["market"]).sum()
    # 6h window stats
    df6 = df[df["timestamp"] >= (pd.Timestamp(end_time) - pd.Timedelta(hours=6))]
    tv_6h = (df6["close"] * df6["volume"]).groupby(df6["market"]).sum() if not df6.empty else pd.Series(dtype=float)

    # Pivot for returns/vol/corr.
    pivot24 = df24.pivot(index="timestamp", columns="market", values="close").sort_index()
    ret24 = pivot24.pct_change(fill_method=None)
    vol_24h = ret24.std(skipna=True)

    # abs ret 6h (end close / close at t<=end-6h - 1)
    end_close = (
        df[df["timestamp"] <= pd.Timestamp(end_time)]
        .sort_values(["market", "timestamp"])
        .groupby("market")
        .tail(1)
        .set_index("market")["close"]
    )
    start6_cut = pd.Timestamp(end_time) - pd.Timedelta(hours=6)
    start_close = (
        df[df["timestamp"] <= start6_cut]
        .sort_values(["market", "timestamp"])
        .groupby("market")
        .tail(1)
        .set_index("market")["close"]
    )
    ret_6h = (end_close / start_close) - 1.0
    abs_ret_6h = ret_6h.abs()

    # Correlation with BTC over lookback_hours
    pivot = df.pivot(index="timestamp", columns="market", values="close").sort_index()
    rets = pivot.pct_change(fill_method=None)
    btc = rets.get("KRW-BTC")
    if btc is None or btc.dropna().shape[0] < 24:
        corr_btc = pd.Series(index=rets.columns, data=np.nan, dtype=float)
    else:
        corr_btc = rets.corrwith(btc)

    out = pd.DataFrame(
        {
            "market": list(tv_24h.index),
            "tv_24h": tv_24h.values.astype(float),
        }
    ).set_index("market")
    out["tv_6h"] = tv_6h.reindex(out.index).fillna(0.0).astype(float)
    out["vol_24h"] = vol_24h.reindex(out.index).fillna(np.nan).astype(float)
    out["abs_ret_6h"] = abs_ret_6h.reindex(out.index).fillna(np.nan).astype(float)
    out["last_ts"] = last_ts.reindex(out.index)
    out["lag_h"] = lag_h.reindex(out.index).astype(float)
    out["corr_btc"] = corr_btc.reindex(out.index).astype(float)

    # tv_surge: compare tv_6h to expected per-6h average from tv_24h
    expected_6h = out["tv_24h"] / 4.0
    out["tv_surge"] = np.where(expected_6h > 0, (out["tv_6h"] / expected_6h) - 1.0, 0.0)
    return out.reset_index()


def _load_returns(
    markets: List[str],
    end_time: datetime,
    lookback_hours: int = 168,
) -> pd.DataFrame:
    """
    Returns pct_change returns pivoted by market for correlation checks.
    """
    markets = _dedupe_preserve(markets or [])
    if not markets:
        return pd.DataFrame()

    start_time = end_time - pd.Timedelta(hours=int(lookback_hours))
    placeholders = ", ".join("?" for _ in markets)
    q = (
        "SELECT timestamp, market, close "
        f"FROM crypto_data WHERE market IN ({placeholders}) AND timestamp BETWEEN ? AND ?"
    )
    params = list(markets) + [
        start_time.strftime("%Y-%m-%dT%H:%M:%S"),
        end_time.strftime("%Y-%m-%dT%H:%M:%S"),
    ]
    df = load_data(q, params=params)
    if df.empty:
        return pd.DataFrame()

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "market", "close"])
    if df.empty:
        return pd.DataFrame()

    pivot = df.pivot(index="timestamp", columns="market", values="close").sort_index()
    rets = pivot.pct_change(fill_method=None)
    return rets


def _max_corr_to_selected(
    candidate: str,
    selected: List[str],
    rets: pd.DataFrame,
    min_overlap: int = 24,
) -> float:
    """
    Returns max correlation of `candidate` returns to any `selected` market.
    Only large positive correlations represent "theme duplication". Negative correlation is allowed.
    If insufficient overlap, returns 0.0 (do not penalize).
    """
    if not selected:
        return 0.0
    if rets is None or rets.empty:
        return 0.0
    if candidate not in rets.columns:
        return 0.0

    sel_cols = [c for c in selected if c in rets.columns and c != candidate]
    if not sel_cols:
        return 0.0

    cand = rets[candidate]
    # Require enough overlap points with at least one selected series.
    overlap_ok = False
    for c in sel_cols:
        if int((cand.notna() & rets[c].notna()).sum()) >= int(min_overlap):
            overlap_ok = True
            break
    if not overlap_ok:
        return 0.0

    try:
        corr = rets[sel_cols].corrwith(cand)
        corr = corr.replace([np.inf, -np.inf], np.nan).dropna()
        return float(corr.max()) if not corr.empty else 0.0
    except Exception:
        return 0.0


def _parse_bucket_quotas(env_val: str) -> Optional[Dict[str, float]]:
    """
    Parse bucket quotas from env string.
    Accepted forms:
      - "high=0.5,mid=0.35,low=0.15"
      - "0.5,0.35,0.15" (treated as high,mid,low)
    Returns normalized dict or None on parse failure.
    """
    try:
        s = str(env_val or "").strip()
        if not s:
            return None
        parts = [p.strip() for p in s.split(",") if p.strip()]
        if not parts:
            return None

        q: Dict[str, float] = {}
        if all("=" in p for p in parts):
            for p in parts:
                k, v = p.split("=", 1)
                k = k.strip().lower()
                v = float(v.strip())
                if k not in ("high", "mid", "low"):
                    continue
                q[k] = v
        else:
            if len(parts) < 3:
                return None
            q = {"high": float(parts[0]), "mid": float(parts[1]), "low": float(parts[2])}

        # Validate and normalize
        if any((k not in q) for k in ("high", "mid", "low")):
            return None
        total = float(q["high"] + q["mid"] + q["low"])
        if not np.isfinite(total) or total <= 0:
            return None
        for k in ("high", "mid", "low"):
            q[k] = max(0.0, float(q[k]) / total)
        # Ensure numeric sanity
        if not all(np.isfinite(q[k]) for k in ("high", "mid", "low")):
            return None
        return q
    except Exception:
        return None


def _select_with_corr_dedup(
    candidates: List[str],
    selected_so_far: List[str],
    rets: pd.DataFrame,
    target_n: int,
    corr_max: float,
    min_overlap: int,
    corr_exempt: Optional[List[str]] = None,
    return_meta: bool = False,
) -> Union[List[str], Tuple[List[str], Dict[str, Any]]]:
    """
    Greedy selection from candidates in order, skipping items that are too correlated
    with already selected markets (excluding corr_exempt).
    """
    target_n = int(target_n)
    if target_n <= 0:
        return ([], {"corr_skipped": 0, "considered": 0}) if return_meta else []

    corr_exempt = set(corr_exempt or [])
    out = []
    corr_skipped = 0
    considered = 0
    for m in candidates:
        if len(out) >= target_n:
            break
        if not m:
            continue
        if m in selected_so_far or m in out:
            continue
        considered += 1
        corr_ref = [x for x in selected_so_far + out if x not in corr_exempt]
        mc = _max_corr_to_selected(m, corr_ref, rets, min_overlap=min_overlap)
        if mc >= float(corr_max):
            corr_skipped += 1
            continue
        out.append(m)
    if return_meta:
        return out, {"corr_skipped": int(corr_skipped), "considered": int(considered)}
    return out


def select_markets_for_scheduled_run(
    mode: str,
    seed_markets: List[str],
    budget: int,
    tv_hours: int = 24,
    candidate_top: int = 200,
    lookback_hours: int = 168,
    exploit_target: Optional[int] = None,
    max_holdings: int = 5,
    max_core: int = 10,
    max_lag_h: Optional[float] = None,
    return_meta: bool = False,
) -> Union[List[str], Tuple[List[str], Dict[str, Any]]]:
    """
    Build a diversified, compute-bounded market list for intraday/morning runs.

    mode: 'intraday' or 'morning'
    """
    budget = int(budget)
    if budget <= 0:
        return ([], {"sel_mode": mode, "budget": budget, "empty_budget": True}) if return_meta else []

    seed_markets = seed_markets or []
    index_coins = list(getattr(config.Data, "MARKET_INDEX_COINS", [])) or ["KRW-BTC", "KRW-ETH"]
    holdings = _load_holdings_from_state(max_n=max_holdings)
    recent_recs = _load_recent_recs("intraday" if mode == "intraday" else "morning", top_n=3)

    meta: Dict[str, Any] = {
        "sel_mode": str(mode),
        "budget": int(budget),
        "tv_hours": int(tv_hours),
        "candidate_top": int(candidate_top),
        "lookback_hours": int(lookback_hours),
        "corr_max": float(os.getenv("AETHER_SELECTION_CORR_MAX", "0.85")),
        "corr_min_overlap": int(os.getenv("AETHER_SELECTION_CORR_MIN_OVERLAP", "24")),
        "seed_n": int(len(seed_markets or [])),
        "holdings_n": int(len(holdings or [])),
        "recent_recs_n": int(len(recent_recs or [])),
    }

    core = _dedupe_preserve(index_coins + holdings + recent_recs + seed_markets)
    # Budget is a hard compute bound; never exceed it even if core sources overflow.
    core_cap = min(int(max_core), int(budget))
    core = core[: max(0, core_cap)]
    meta["core_n"] = int(len(core))
    try:
        meta["core_index_n"] = int(sum(1 for m in core if m in set(index_coins)))
        meta["core_holdings_n"] = int(sum(1 for m in core if m in set(holdings)))
        meta["core_recent_recs_n"] = int(sum(1 for m in core if m in set(recent_recs)))
        meta["core_seed_n"] = int(sum(1 for m in core if m in set(seed_markets)))
    except Exception:
        pass
    meta["core_cap"] = int(core_cap)

    remaining = budget - len(core)
    meta["remaining_after_core"] = int(remaining)
    if remaining <= 0:
        return (core, meta) if return_meta else core

    end_time = get_latest_db_timestamp()
    if end_time is None:
        # No DB: just return core (it at least contains index coins).
        meta["db_missing"] = True
        return (core, meta) if return_meta else core

    # Candidate pool by liquidity (DB), then score + diversity quotas.
    # Pull a wider set than we finally need to allow bucketing.
    pool = get_top_markets_by_trading_value(
        limit=max(candidate_top, remaining * 5),
        hours=tv_hours,
        market_prefix="KRW-",
    )
    meta["pool_n_raw"] = int(len(pool or []))
    pool = [m for m in pool if m not in core]
    pool = pool[:candidate_top]
    meta["pool_n"] = int(len(pool or []))

    stats = _compute_candidate_stats(pool, end_time=end_time, lookback_hours=lookback_hours)
    if stats.empty:
        meta["stats_empty"] = True
        return (core, meta) if return_meta else core
    meta["stats_n"] = int(stats.shape[0])

    # Freshness filter for selection stage (separate from the main hard gate).
    if max_lag_h is not None:
        meta["max_lag_h_sel"] = float(max_lag_h)
        stats = stats[stats["lag_h"].notna() & (stats["lag_h"] <= float(max_lag_h))]
        if stats.empty:
            meta["stats_all_stale"] = True
            return (core, meta) if return_meta else core
    meta["stats_n_after_freshness"] = int(stats.shape[0])

    # Score: liquidity + volatility + movement + volume surge.
    # Keep it simple and monotonic; clamp surge to avoid extreme outliers dominating.
    stats = stats.copy()
    stats["tv_surge_clamped"] = stats["tv_surge"].clip(lower=-1.0, upper=5.0)
    stats["score"] = (
        np.log1p(stats["tv_24h"].clip(lower=0.0))
        + 0.7 * stats["vol_24h"].fillna(0.0)
        + 0.5 * stats["abs_ret_6h"].fillna(0.0)
        + 0.7 * stats["tv_surge_clamped"].fillna(0.0)
    )

    # Correlation de-dup settings (already recorded into meta above).
    corr_max = float(meta["corr_max"])
    corr_min_overlap = int(meta["corr_min_overlap"])

    # Preload returns for correlation checks (cheap: lookback_hours x candidate_top).
    # We intentionally exempt BTC/ETH from corr reference because corr_btc bucketing already handles "BTC-ness".
    end_rets = _load_returns(
        _dedupe_preserve(list(core) + stats["market"].astype(str).tolist()),
        end_time=end_time,
        lookback_hours=lookback_hours,
    )

    # Exploit: top liquidity slice (with optional corr de-dup to improve diversity).
    if exploit_target is None:
        exploit_target = 16 if mode == "intraday" else 32
    exploit_target = min(int(exploit_target), remaining)
    meta["exploit_target"] = int(exploit_target)
    exploit_candidates = stats.sort_values("tv_24h", ascending=False)["market"].astype(str).tolist()
    meta["exploit_candidates_n"] = int(len(exploit_candidates or []))
    if exploit_target > 0:
        # First pass: apply corr de-dup.
        exploit, exp_meta = _select_with_corr_dedup(
            exploit_candidates,
            selected_so_far=list(core),
            rets=end_rets,
            target_n=exploit_target,
            corr_max=corr_max,
            min_overlap=corr_min_overlap,
            corr_exempt=index_coins,
            return_meta=True,
        )
        meta["exploit_corr_skipped"] = int((exp_meta or {}).get("corr_skipped", 0))
        # Second pass: top-up without corr constraint to preserve liquidity coverage.
        exploit_topup = 0
        if len(exploit) < exploit_target:
            need = exploit_target - len(exploit)
            for m in exploit_candidates:
                if need <= 0:
                    break
                if m in core or m in exploit:
                    continue
                exploit.append(m)
                exploit_topup += 1
                need -= 1
        meta["exploit_topup_used"] = int(exploit_topup)
    else:
        exploit = []
        meta["exploit_corr_skipped"] = 0
        meta["exploit_topup_used"] = 0
    meta["exploit_selected"] = int(len(exploit))

    remaining2 = remaining - len(exploit)
    meta["explore_target"] = int(max(0, remaining2))
    if remaining2 <= 0:
        final0 = _dedupe_preserve(core + exploit)[:budget]
        meta["final_n"] = int(len(final0))
        return (final0, meta) if return_meta else final0

    picked = set(core + exploit)
    cand = stats[~stats["market"].isin(picked)].copy()
    if cand.empty:
        final0 = _dedupe_preserve(core + exploit)[:budget]
        meta["final_n"] = int(len(final0))
        meta["cand_empty_after_picked"] = True
        return (final0, meta) if return_meta else final0

    # For explore stage, reuse end_rets if available; extend if needed.
    rets = end_rets
    if rets is None or rets.empty:
        rets = _load_returns(
            _dedupe_preserve(list(picked) + cand["market"].astype(str).tolist()),
            end_time=end_time,
            lookback_hours=lookback_hours,
        )

    # Buckets by BTC corr.
    def _bucket(v: float) -> str:
        try:
            if not np.isfinite(v):
                return "mid"
            if v < 0.2:
                return "low"
            if v <= 0.6:
                return "mid"
            return "high"
        except Exception:
            return "mid"

    cand["bucket"] = cand["corr_btc"].apply(_bucket)

    if mode == "intraday":
        quotas = {"high": 0.50, "mid": 0.35, "low": 0.15}
        q_env = os.getenv("AETHER_SELECTION_BUCKET_QUOTAS_INTRADAY", "")
    else:
        quotas = {"high": 0.40, "mid": 0.40, "low": 0.20}
        q_env = os.getenv("AETHER_SELECTION_BUCKET_QUOTAS_MORNING", "")

    q_override = _parse_bucket_quotas(q_env)
    if q_override:
        quotas = q_override
        meta["bucket_quotas_source"] = "env"
        meta["bucket_quotas_env"] = str(q_env)
    else:
        meta["bucket_quotas_source"] = "default"

    # Convert quotas to counts; ensure sum == remaining2
    counts = {k: int(round(remaining2 * quotas[k])) for k in quotas}
    # Fix rounding drift
    drift = remaining2 - sum(counts.values())
    if drift != 0:
        # Add/subtract to mid bucket first.
        counts["mid"] = max(0, counts["mid"] + drift)

    explore = []
    selected_so_far = _dedupe_preserve(list(picked))
    explore_corr_skipped = 0
    bucket_filled: Dict[str, int] = {"high": 0, "mid": 0, "low": 0}
    for b in ("high", "mid", "low"):
        n = max(0, int(counts.get(b, 0)))
        if n <= 0:
            continue
        part = cand[cand["bucket"] == b].sort_values("score", ascending=False)
        chosen_in_bucket = 0
        for m in part["market"].astype(str).tolist():
            if chosen_in_bucket >= n:
                break
            if m in picked or m in explore:
                continue
            mc = _max_corr_to_selected(
                m,
                [x for x in selected_so_far if x not in set(index_coins)],
                rets,
                min_overlap=corr_min_overlap,
            )
            if mc >= corr_max:
                explore_corr_skipped += 1
                continue
            explore.append(m)
            selected_so_far.append(m)
            chosen_in_bucket += 1
        bucket_filled[b] = int(chosen_in_bucket)

    # If any bucket was empty, top-up by score globally.
    explore = _dedupe_preserve(explore)
    explore_topup_used = 0
    if len(explore) < remaining2:
        need = remaining2 - len(explore)
        topup = cand[~cand["market"].isin(explore)].sort_values("score", ascending=False)["market"].astype(str).tolist()
        for m in topup:
            if len(explore) >= remaining2:
                break
            if m in picked or m in explore:
                continue
            mc = _max_corr_to_selected(
                m,
                [x for x in selected_so_far if x not in set(index_coins)],
                rets,
                min_overlap=corr_min_overlap,
            )
            if mc >= corr_max:
                explore_corr_skipped += 1
                continue
            explore.append(m)
            selected_so_far.append(m)
            explore_topup_used += 1

    final = _dedupe_preserve(core + exploit + explore)[:budget]
    meta["explore_selected"] = int(max(0, len(final) - len(core) - len(exploit)))
    meta["explore_corr_skipped"] = int(explore_corr_skipped)
    meta["explore_topup_used"] = int(explore_topup_used)

    # Bucket quotas/filled (helps diagnose diversity constraints).
    meta["bucket_quota_high"] = int(counts.get("high", 0))
    meta["bucket_quota_mid"] = int(counts.get("mid", 0))
    meta["bucket_quota_low"] = int(counts.get("low", 0))
    meta["bucket_filled_high"] = int(bucket_filled.get("high", 0))
    meta["bucket_filled_mid"] = int(bucket_filled.get("mid", 0))
    meta["bucket_filled_low"] = int(bucket_filled.get("low", 0))
    meta["final_n"] = int(len(final))

    logger.info(
        f"[Select] mode={mode} budget={budget} core={len(core)} exploit={len(exploit)} explore={len(final)-len(core)-len(exploit)} "
        f"(holdings={len(holdings)} seed={len(seed_markets)})"
    )
    return (final, meta) if return_meta else final
