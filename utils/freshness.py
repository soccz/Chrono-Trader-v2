from datetime import datetime, timezone
from typing import Optional, Tuple, List

from data.database import get_latest_db_timestamp, get_latest_db_timestamps_by_market


def get_db_latest_and_lag_hours(
    markets: Optional[List[str]] = None,
    now_utc: Optional[datetime] = None,
) -> Tuple[Optional[datetime], Optional[float]]:
    """
    Returns (db_latest_utc, lag_hours) for given markets (or globally if markets is None).
    If DB has no data, returns (None, None).
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    latest = get_latest_db_timestamp(markets=markets)
    if latest is None:
        return None, None

    lag_h = (now_utc - latest).total_seconds() / 3600.0
    return latest, lag_h


def evaluate_market_freshness(
    markets: List[str],
    max_lag_h: float,
    fallback_markets: Optional[List[str]] = None,
    now_utc: Optional[datetime] = None,
):
    """
    Returns a standardized freshness decision payload for scheduled runs.

    Payload keys:
    - status: ok | fallback | stale_abort
    - kept_markets: list[str]
    - dropped: list[tuple[str, Optional[float]]]
    - kept_lags: list[float]
    - used_fallback: bool
    - fallback_markets: list[str]
    - worst_lag_h: Optional[float]
    """
    markets = list(markets or [])
    fallback_markets = list(fallback_markets or [])
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    ts_map = get_latest_db_timestamps_by_market(markets)
    kept = []
    dropped = []
    kept_lags = []
    for market in markets:
        ts = ts_map.get(market)
        if ts is None:
            dropped.append((market, None))
            continue
        lag_h = (now_utc - ts).total_seconds() / 3600.0
        if lag_h > float(max_lag_h):
            dropped.append((market, lag_h))
            continue
        kept.append(market)
        kept_lags.append(lag_h)

    if kept:
        return {
            "status": "ok",
            "kept_markets": kept,
            "dropped": dropped,
            "kept_lags": kept_lags,
            "used_fallback": False,
            "fallback_markets": [],
            "worst_lag_h": max(kept_lags) if kept_lags else None,
        }

    fallback_ts_map = get_latest_db_timestamps_by_market(fallback_markets)
    fb_kept = []
    fb_lags = []
    for market in fallback_markets:
        ts = fallback_ts_map.get(market)
        if ts is None:
            continue
        lag_h = (now_utc - ts).total_seconds() / 3600.0
        if lag_h <= float(max_lag_h):
            fb_kept.append(market)
            fb_lags.append(lag_h)

    if fb_kept:
        return {
            "status": "fallback",
            "kept_markets": fb_kept,
            "dropped": dropped,
            "kept_lags": fb_lags,
            "used_fallback": True,
            "fallback_markets": fb_kept,
            "worst_lag_h": max(fb_lags) if fb_lags else None,
        }

    return {
        "status": "stale_abort",
        "kept_markets": [],
        "dropped": dropped,
        "kept_lags": [],
        "used_fallback": False,
        "fallback_markets": [],
        "worst_lag_h": None,
    }
