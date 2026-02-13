def test_parse_bucket_quotas():
    from utils.market_selection import _parse_bucket_quotas

    q = _parse_bucket_quotas("high=0.5,mid=0.3,low=0.2")
    assert q is not None
    assert abs((q["high"] + q["mid"] + q["low"]) - 1.0) < 1e-9

    q2 = _parse_bucket_quotas("0.5,0.3,0.2")
    assert q2 is not None
    assert abs(q2["high"] - q["high"]) < 1e-9

    assert _parse_bucket_quotas("") is None
    assert _parse_bucket_quotas("high=1") is None

