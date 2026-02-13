from datetime import datetime, timezone, timedelta


def test_get_db_latest_and_lag_hours_returns_none_when_no_data(monkeypatch):
    from utils import freshness

    monkeypatch.setattr(freshness, "get_latest_db_timestamp", lambda markets=None: None)
    latest, lag_h = freshness.get_db_latest_and_lag_hours(markets=["KRW-BTC"])
    assert latest is None
    assert lag_h is None


def test_get_db_latest_and_lag_hours_computes_lag(monkeypatch):
    from utils import freshness

    base = datetime(2026, 2, 13, 0, 0, 0, tzinfo=timezone.utc)
    now = base + timedelta(hours=5, minutes=30)

    monkeypatch.setattr(freshness, "get_latest_db_timestamp", lambda markets=None: base)
    latest, lag_h = freshness.get_db_latest_and_lag_hours(markets=["KRW-BTC"], now_utc=now)
    assert latest == base
    assert abs(lag_h - 5.5) < 1e-9

