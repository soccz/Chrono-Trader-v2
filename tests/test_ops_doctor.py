import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path


def _load_ops_doctor():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "ops_doctor.py"
    spec = importlib.util.spec_from_file_location("test_ops_doctor_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_report_marks_core_ok_with_local_runtime_and_assets(monkeypatch, tmp_path):
    repo_root = tmp_path
    analysis_dir = repo_root / "analysis"
    data_dir = repo_root / "data"
    models_dir = repo_root / "models"
    venv_bin = repo_root / ".venv_local" / "bin"
    venv_bin.mkdir(parents=True)
    analysis_dir.mkdir()
    data_dir.mkdir()
    models_dir.mkdir()

    (venv_bin / "python").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    (venv_bin / "python").chmod(0o755)
    (data_dir / "crypto_data.db").write_text("db", encoding="utf-8")
    (models_dir / "model_1.pth").write_text("main", encoding="utf-8")
    (models_dir / "model_short.pth").write_text("short", encoding="utf-8")
    now = datetime.now(timezone.utc)
    (analysis_dir / "run_markets_metrics_intraday.json").write_text(
        json.dumps({"ts": now.isoformat(), "recs": {"n": 1, "has_watch": False}, "meta": {}}),
        encoding="utf-8",
    )
    (analysis_dir / "run_markets_metrics_morning.json").write_text(
        json.dumps({"ts": (now - timedelta(hours=1)).isoformat(), "recs": {"n": 1, "has_watch": True}, "meta": {}}),
        encoding="utf-8",
    )
    (analysis_dir / "output_contract_intraday.json").write_text('{"mode":"intraday"}', encoding="utf-8")
    (analysis_dir / "output_contract_morning.json").write_text('{"mode":"morning"}', encoding="utf-8")
    (analysis_dir / "output_contract_refresh-db.json").write_text('{"mode":"refresh-db"}', encoding="utf-8")

    monkeypatch.chdir(repo_root)
    ops_doctor = _load_ops_doctor()

    report = ops_doctor.build_report(
        repo_root=repo_root,
        max_age_intraday_h=5.0,
        max_age_morning_h=30.0,
    )

    assert report["core_ok"] is True
    assert report["outputs_ok"] is True
    assert report["python"]["ok"] is True
    assert report["db"]["exists"] is True
    assert report["models"]["main"]["exists"] is True
    assert report["models"]["audit"]["total_count"] == 1
    assert report["models"]["audit"]["compatible_count"] == 0
    assert report["manifests"]["refresh-db"] is True


def test_build_report_detects_transplanted_venv(monkeypatch, tmp_path):
    repo_root = tmp_path
    analysis_dir = repo_root / "analysis"
    data_dir = repo_root / "data"
    models_dir = repo_root / "models"
    venv_dir = repo_root / ".venv"
    analysis_dir.mkdir()
    data_dir.mkdir()
    models_dir.mkdir()
    venv_dir.mkdir()

    (data_dir / "crypto_data.db").write_text("db", encoding="utf-8")
    (models_dir / "model_1.pth").write_text("main", encoding="utf-8")
    (venv_dir / "pyvenv.cfg").write_text(
        "home = /Users/someone/.pyenv/versions/3.13.0/bin\nexecutable = /Users/someone/project/.venv/bin/python\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(repo_root)
    ops_doctor = _load_ops_doctor()
    report = ops_doctor.build_report(repo_root=repo_root)

    assert report["venv"]["transplanted"] is True
    assert report["venv"]["reasons"]


def test_main_strict_outputs_fails_when_outputs_are_missing(monkeypatch, tmp_path, capsys):
    repo_root = tmp_path
    data_dir = repo_root / "data"
    models_dir = repo_root / "models"
    venv_bin = repo_root / ".venv_local" / "bin"
    venv_bin.mkdir(parents=True)
    data_dir.mkdir()
    models_dir.mkdir()

    (venv_bin / "python").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    (venv_bin / "python").chmod(0o755)
    (data_dir / "crypto_data.db").write_text("db", encoding="utf-8")
    (models_dir / "model_1.pth").write_text("main", encoding="utf-8")

    monkeypatch.chdir(repo_root)
    monkeypatch.setattr("sys.argv", ["ops_doctor.py", "--json", "--strict_outputs"])
    ops_doctor = _load_ops_doctor()

    try:
        ops_doctor.main()
    except SystemExit as exc:
        assert int(exc.code) == 2

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["core_ok"] is True
    assert payload["outputs_ok"] is False
