from pathlib import Path


def test_build_and_read_output_manifest(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rec_dir = tmp_path / "recommendations"
    pred_dir = tmp_path / "predictions"
    analysis_dir = tmp_path / "analysis"
    rec_dir.mkdir()
    pred_dir.mkdir()
    analysis_dir.mkdir()

    rec_path = rec_dir / "recs_intraday_20260319_120000.csv"
    pred_path = pred_dir / "preds_latest.csv"
    pump_path = pred_dir / "pump_preds_20260319_120000.csv"
    metrics_path = analysis_dir / "run_markets_metrics_intraday.json"

    rec_path.write_text("market,expected_return\nKRW-BTC,0.01\n", encoding="utf-8")
    pred_path.write_text("market,pred\nKRW-BTC,0.01\n", encoding="utf-8")
    pump_path.write_text("market,total_pump_probability\nKRW-XRP,0.8\n", encoding="utf-8")
    metrics_path.write_text('{"ts":"2026-03-19T12:00:00+00:00"}', encoding="utf-8")

    from utils.output_contract import build_output_manifest, write_output_manifest, read_output_manifest

    manifest = build_output_manifest(
        mode="intraday",
        rec_tag="intraday",
        run_metrics_path=str(metrics_path),
        include_pump_preds=True,
    )
    manifest_path = write_output_manifest("intraday", manifest)
    loaded = read_output_manifest("intraday")

    assert Path(manifest_path).exists()
    assert loaded is not None
    assert loaded["mode"] == "intraday"
    assert loaded["recommendation"]["basename"] == rec_path.name
    assert loaded["prediction"]["basename"] == pred_path.name
    assert loaded["pump_prediction"]["basename"] == pump_path.name
    assert loaded["run_metrics"]["basename"] == metrics_path.name


def test_read_output_manifests_skips_missing_modes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()

    path = analysis_dir / "output_contract_morning.json"
    path.write_text('{"mode":"morning","ts":"2026-03-19T00:00:00+00:00"}', encoding="utf-8")

    from utils.output_contract import read_output_manifests

    manifests = read_output_manifests(["morning", "intraday"])
    assert set(manifests.keys()) == {"morning"}
    assert manifests["morning"]["mode"] == "morning"
