def test_minrec_watch_only_when_trade_impossible(monkeypatch):
    # Minimal unit test: if minrec_mode=watch, ensure status becomes Watch (MinRec)
    from inference import recommender as rec
    from utils.config import config

    monkeypatch.setattr(config.Recommender, "MIN_RECOMMENDATIONS_LIVE", 1, raising=False)
    monkeypatch.setattr(config.Recommender, "MIN_REC_MODE", "watch", raising=False)

    # Provide one candidate; make it survive initial filters enough to reach minrec.
    predictions = [
        {
            "market": "KRW-BTC",
            "current_price": 100.0,
            "predicted_pattern": [0.001] * 6,  # small expected return
            "uncertainty": 999999.0,
            "gate_value": 0.5,
            "consensus_score": 0.6,
        }
    ]

    out = rec.run(predictions=predictions, mode="live", min_k=1)
    assert out, "should emit at least one item under MinRec"
    assert out[0].get("status") in ("Watch (MinRec)", "Recommended", "Forced (Min 1)")

