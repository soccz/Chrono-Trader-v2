import json
import os
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from utils.logger import logger
from data.database import get_latest_db_timestamps_by_market


def _cache_path(mode: str) -> str:
    os.makedirs("analysis", exist_ok=True)
    safe = mode.replace("/", "_")
    return os.path.join("analysis", f"run_markets_{safe}.json")


def load_previous(mode: str) -> List[str]:
    path = _cache_path(mode)
    try:
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f) or {}
        mkts = d.get("markets") or []
        return [str(x) for x in mkts if x]
    except Exception as e:
        logger.debug(f"[RunMarkets] Failed to load cache: {e}")
        return []


def save_current(mode: str, markets: List[str], meta: Optional[dict] = None) -> None:
    path = _cache_path(mode)
    try:
        payload = {
            "mode": mode,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "markets": list(markets or []),
            "meta": meta or {},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.debug(f"[RunMarkets] Failed to save cache: {e}")


def rotate_markets(
    mode: str,
    new_markets: List[str],
    budget: int,
    keep_ratio: float = 0.7,
    max_lag_h: Optional[float] = None,
) -> Tuple[List[str], dict]:
    """
    Keep a portion of previous run markets (if still fresh), then fill with new markets.
    """
    budget = int(budget)
    if budget <= 0:
        return [], {"kept": 0, "added": 0, "budget": budget}

    prev = load_previous(mode)
    prev = [m for m in prev if m]
    new_markets = [m for m in (new_markets or []) if m]

    # Freshness filter for prev markets.
    kept_prev = []
    if prev and max_lag_h is not None:
        ts_map = get_latest_db_timestamps_by_market(prev)
        now_utc = datetime.now(timezone.utc)
        for m in prev:
            ts = ts_map.get(m)
            if ts is None:
                continue
            lag_h = (now_utc - ts).total_seconds() / 3600.0
            if lag_h <= float(max_lag_h):
                kept_prev.append(m)
    else:
        kept_prev = prev

    keep_n = int(round(budget * float(keep_ratio)))
    keep_n = max(0, min(keep_n, budget))
    kept_prev = kept_prev[:keep_n]

    out = []
    seen = set()
    for m in kept_prev:
        if m not in seen:
            out.append(m)
            seen.add(m)

    for m in new_markets:
        if len(out) >= budget:
            break
        if m in seen:
            continue
        out.append(m)
        seen.add(m)

    meta = {"kept": len(kept_prev), "added": max(0, len(out) - len(kept_prev)), "budget": budget}
    save_current(mode, out, meta=meta)
    return out, meta

