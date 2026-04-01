import importlib.util
from pathlib import Path


def _load_backfill_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "backfill_output_contracts.py"
    spec = importlib.util.spec_from_file_location("test_backfill_output_contracts_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_output_contract_module():
    module_path = Path(__file__).resolve().parents[1] / "utils" / "output_contract.py"
    spec = importlib.util.spec_from_file_location("test_output_contract_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_backfill_output_manifest_creates_intraday_manifest(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "analysis").mkdir()
    (tmp_path / "recommendations").mkdir()
    (tmp_path / "predictions").mkdir()

    (tmp_path / "analysis" / "run_markets_metrics_intraday.json").write_text(
        '{"mode":"intraday","ts":"2026-03-19T12:00:00+00:00","recs":{"n":1},"meta":{}}',
        encoding="utf-8",
    )
    rec_path = tmp_path / "recommendations" / "recs_intraday_20260319_120000.csv"
    rec_path.write_text("market,expected_return\nKRW-BTC,0.01\n", encoding="utf-8")

    output_contract = _load_output_contract_module()
    manifest_path = output_contract.backfill_output_manifest("intraday")
    manifest = output_contract.read_output_manifest("intraday")

    assert manifest_path
    assert manifest is not None
    assert manifest["mode"] == "intraday"
    assert manifest["recommendation"]["basename"] == rec_path.name
    assert manifest["run_metrics"]["basename"] == "run_markets_metrics_intraday.json"


def test_backfill_script_runs_without_error(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "analysis").mkdir()
    (tmp_path / "analysis" / "run_markets_metrics_refresh-db.json").write_text(
        '{"mode":"refresh-db","ts":"2026-03-19T12:00:00+00:00","recs":{"n":0},"meta":{}}',
        encoding="utf-8",
    )

    module = _load_backfill_module()
    monkeypatch.setattr("sys.argv", ["backfill_output_contracts.py", "--modes", "refresh-db"])
    module.main()
    out = capsys.readouterr().out
    assert "refresh-db" in out
