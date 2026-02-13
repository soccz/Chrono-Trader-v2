def test_minrec_synth_when_no_predictions(monkeypatch):
    from inference import recommender as rec
    from utils.config import config

    monkeypatch.setattr(config.Recommender, "MIN_RECOMMENDATIONS_LIVE", 1, raising=False)
    monkeypatch.setattr(config.Recommender, "MIN_REC_MODE", "watch", raising=False)
    monkeypatch.setattr(config.Recommender, "MIN_REC_ALLOW_WATCH_ONLY_FALLBACK", True, raising=False)

    out = rec.run(predictions=[], mode="live", min_k=1)
    assert out and len(out) >= 1
    assert str(out[0].get("status", "")).startswith("Watch")
    assert float(out[0].get("position_size", 0.0)) == 0.0


def test_minrec_synth_when_all_filtered(monkeypatch):
    from inference import recommender as rec
    from utils.config import config

    monkeypatch.setattr(config.Recommender, "MIN_RECOMMENDATIONS_LIVE", 1, raising=False)
    monkeypatch.setattr(config.Recommender, "MIN_REC_MODE", "watch", raising=False)
    monkeypatch.setattr(config.Recommender, "MIN_REC_ALLOW_WATCH_ONLY_FALLBACK", True, raising=False)

    # Make direction-consistency < 0.66 (e.g., 3 positive / 3 negative) so Step-1 drops it.
    predictions = [
        {
            "market": "KRW-BTC",
            "current_price": 100.0,
            "predicted_pattern": [0.01, 0.01, 0.01, -0.01, -0.01, -0.01],
            "uncertainty": 1000.0,
            "gate_value": 0.5,
            "consensus_score": 0.6,
        }
    ]

    out = rec.run(predictions=predictions, mode="live", min_k=1)
    assert out and len(out) >= 1
    assert str(out[0].get("status", "")).startswith("Watch")
    assert float(out[0].get("position_size", 0.0)) == 0.0

