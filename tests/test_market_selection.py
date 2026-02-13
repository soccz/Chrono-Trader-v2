import pandas as pd


def test_dedupe_preserve():
    from utils.market_selection import _dedupe_preserve

    assert _dedupe_preserve(["A", "B", "A", "", None, "C"]) == ["A", "B", "C"]


def test_select_markets_budget_respected(monkeypatch):
    from utils import market_selection as ms

    # Fake DB timestamp exists
    monkeypatch.setattr(ms, "get_latest_db_timestamp", lambda: pd.Timestamp("2026-02-13T04:00:00Z").to_pydatetime())

    # Candidates from DB by trading value
    monkeypatch.setattr(ms, "get_top_markets_by_trading_value", lambda limit, hours, market_prefix: [f"KRW-X{i}" for i in range(300)])

    # Fake stats: 200 rows
    def _fake_stats(markets, end_time, lookback_hours):
        rows = []
        for i, m in enumerate(markets):
            rows.append(
                {
                    "market": m,
                    "tv_24h": 1000.0 + i,
                    "tv_6h": 200.0 + i,
                    "vol_24h": 0.01 + (i % 10) * 0.001,
                    "abs_ret_6h": 0.02,
                    "tv_surge": 0.0,
                    "last_ts": pd.Timestamp("2026-02-13T04:00:00Z"),
                    "lag_h": 1.0,
                    "corr_btc": 0.7 if i % 3 == 0 else (0.4 if i % 3 == 1 else 0.1),
                }
            )
        return pd.DataFrame(rows)

    monkeypatch.setattr(ms, "_compute_candidate_stats", _fake_stats)

    out, meta = ms.select_markets_for_scheduled_run(
        mode="intraday",
        seed_markets=["KRW-SEED"],
        budget=24,
        tv_hours=24,
        candidate_top=200,
        lookback_hours=168,
        max_holdings=0,
        max_core=10,
        max_lag_h=6.0,
        return_meta=True,
    )
    assert len(out) <= 24
    assert "KRW-BTC" in out  # core contains index coin
    assert isinstance(meta, dict)
    assert meta.get("budget") == 24


def test_corr_dedup_skips_highly_correlated(monkeypatch):
    from utils import market_selection as ms
    import numpy as np
    import pandas as pd

    monkeypatch.setenv("AETHER_SELECTION_CORR_MAX", "0.85")
    monkeypatch.setenv("AETHER_SELECTION_CORR_MIN_OVERLAP", "5")

    monkeypatch.setattr(ms, "get_latest_db_timestamp", lambda: pd.Timestamp("2026-02-13T04:00:00Z").to_pydatetime())
    monkeypatch.setattr(ms, "get_top_markets_by_trading_value", lambda limit, hours, market_prefix: ["KRW-A", "KRW-B", "KRW-C"])

    # Make A and B perfectly correlated, C uncorrelated.
    idx = pd.date_range("2026-02-12", periods=10, freq="h", tz="UTC")
    rets = pd.DataFrame(
        {
            "KRW-BTC": np.linspace(0, 0.09, 10),
            "KRW-A": np.linspace(0, 0.09, 10),
            "KRW-B": np.linspace(0, 0.09, 10),
            "KRW-C": np.linspace(0, 0.18, 10)[::-1],
        },
        index=idx,
    )

    def _fake_stats(markets, end_time, lookback_hours):
        rows = []
        for m in markets:
            rows.append(
                {
                    "market": m,
                    "tv_24h": 1000.0,
                    "tv_6h": 200.0,
                    "vol_24h": 0.02,
                    "abs_ret_6h": 0.02,
                    "tv_surge": 0.0,
                    "last_ts": pd.Timestamp("2026-02-13T04:00:00Z"),
                    "lag_h": 1.0,
                    "corr_btc": 0.7,
                }
            )
        return pd.DataFrame(rows)

    monkeypatch.setattr(ms, "_compute_candidate_stats", _fake_stats)
    monkeypatch.setattr(ms, "_load_returns", lambda markets, end_time, lookback_hours: rets)

    out = ms.select_markets_for_scheduled_run(
        mode="intraday",
        seed_markets=[],
        budget=6,
        tv_hours=24,
        candidate_top=3,
        lookback_hours=168,
        exploit_target=0,  # force selection into explore stage
        max_holdings=0,
        max_core=2,  # core = BTC/ETH
        max_lag_h=6.0,
    )
    # core has BTC/ETH, explore should not contain both A and B due to corr dedup.
    assert not ("KRW-A" in out and "KRW-B" in out)


def test_exploit_stage_corr_dedup(monkeypatch):
    from utils import market_selection as ms
    import numpy as np
    import pandas as pd

    monkeypatch.setenv("AETHER_SELECTION_CORR_MAX", "0.85")
    monkeypatch.setenv("AETHER_SELECTION_CORR_MIN_OVERLAP", "5")

    monkeypatch.setattr(ms, "get_latest_db_timestamp", lambda: pd.Timestamp("2026-02-13T04:00:00Z").to_pydatetime())
    monkeypatch.setattr(ms, "get_top_markets_by_trading_value", lambda limit, hours, market_prefix: ["KRW-A", "KRW-B", "KRW-C"])

    def _fake_stats(markets, end_time, lookback_hours):
        # Make A/B high trading value, C lower.
        rows = []
        tvs = {"KRW-A": 3000.0, "KRW-B": 2900.0, "KRW-C": 1000.0}
        for m in markets:
            rows.append(
                {
                    "market": m,
                    "tv_24h": tvs.get(m, 1000.0),
                    "tv_6h": 200.0,
                    "vol_24h": 0.02,
                    "abs_ret_6h": 0.02,
                    "tv_surge": 0.0,
                    "last_ts": pd.Timestamp("2026-02-13T04:00:00Z"),
                    "lag_h": 1.0,
                    "corr_btc": 0.7,
                }
            )
        return pd.DataFrame(rows)

    idx = pd.date_range("2026-02-12", periods=10, freq="h", tz="UTC")
    # A and B identical, so exploit corr de-dup should pick only one of them (and then top-up by C if needed).
    rets = pd.DataFrame(
        {
            "KRW-BTC": np.linspace(0, 0.09, 10),
            "KRW-ETH": np.linspace(0, 0.07, 10),
            "KRW-A": np.linspace(0, 0.09, 10),
            "KRW-B": np.linspace(0, 0.09, 10),
            "KRW-C": np.linspace(0, 0.18, 10)[::-1],
        },
        index=idx,
    )

    monkeypatch.setattr(ms, "_compute_candidate_stats", _fake_stats)
    monkeypatch.setattr(ms, "_load_returns", lambda markets, end_time, lookback_hours: rets)

    out = ms.select_markets_for_scheduled_run(
        mode="intraday",
        seed_markets=[],
        budget=6,
        tv_hours=24,
        candidate_top=3,
        lookback_hours=168,
        exploit_target=2,  # exploit tries to take 2
        max_holdings=0,
        max_core=2,  # core BTC/ETH
        max_lag_h=6.0,
    )

    # Ensure exploit doesn't include both A and B simultaneously.
    assert not ("KRW-A" in out and "KRW-B" in out)
