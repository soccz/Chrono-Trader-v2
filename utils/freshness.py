from datetime import datetime, timezone
from typing import Optional, Tuple, List

from data.database import get_latest_db_timestamp


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

