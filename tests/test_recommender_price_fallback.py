import numpy as np
import pandas as pd


def test_recommender_db_price_fallback(monkeypatch):
    from inference import recommender as rec
    from utils.config import config

    # Ensure we don't call any external endpoints in this test.
    monkeypatch.setattr(rec, "get_tradeable_markets", lambda: {"KRW-BTC"})
    monkeypatch.setattr(rec, "get_market_index", lambda: pd.DataFrame())
    monkeypatch.setattr(rec, "get_historical_success_patterns", lambda: np.array([]))
    monkeypatch.setattr(rec, "get_trading_values_for_markets", lambda markets, end_time, hours: {"KRW-BTC": 10_000_000_000})

    # Provide DB last close for KRW-BTC
    monkeypatch.setattr(rec, "load_data", lambda q, params=None: pd.DataFrame([{"market": "KRW-BTC", "close": 100.0}]))

    preds = [
        {
            "market": "KRW-BTC",
            "predicted_pattern": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
            "uncertainty": 10.0,
            "consensus_score": 1.0,
            "current_price": None,
        }
    ]

    # Relax DTW threshold for safety (though DTW is skipped anyway due to empty success patterns).
    monkeypatch.setattr(config.Recommender, "DTW_THRESHOLD", 999.0, raising=False)

    out = rec.run(predictions=preds, mode="live", min_k=1)
    assert out and out[0].get("market") == "KRW-BTC"
    assert float(out[0].get("current_price") or 0) > 0


def test_recommender_runtime_watch_only(monkeypatch):
    from inference import recommender as rec
    from utils.config import config

    monkeypatch.setattr(rec, "get_tradeable_markets", lambda: {"KRW-BTC"})
    monkeypatch.setattr(rec, "get_market_index", lambda: pd.DataFrame())
    monkeypatch.setattr(rec, "get_historical_success_patterns", lambda: np.array([]))
    monkeypatch.setattr(rec, "get_trading_values_for_markets", lambda markets, end_time, hours: {"KRW-BTC": 10_000_000_000})
    monkeypatch.setattr(rec, "load_data", lambda q, params=None: pd.DataFrame([{"market": "KRW-BTC", "close": 100.0}]))

    monkeypatch.setattr(config.Recommender, "RUNTIME_WATCH_ONLY", True, raising=False)

    preds = [
        {
            "market": "KRW-BTC",
            "predicted_pattern": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
            "uncertainty": 10.0,
            "consensus_score": 1.0,
            "current_price": None,
        }
    ]

    out = rec.run(predictions=preds, mode="live", min_k=1)
    assert out and out[0].get("status", "").startswith("Watch")
    assert float(out[0].get("position_size", 1.0)) == 0.0
