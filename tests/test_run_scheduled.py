from contextlib import nullcontext
import importlib.util
from pathlib import Path


def _load_run_scheduled():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "run_scheduled.py"
    spec = importlib.util.spec_from_file_location("test_run_scheduled_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_run_scheduled_intraday_orchestrates_refresh_and_infer(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    run_scheduled = _load_run_scheduled()

    calls = []
    tuning = {
        "intraday": {
            "recommend": {
                "selection_corr_max": 0.42,
                "bucket_quotas_env": "high:2,mid:2,low:1",
                "rotation_keep": 0.55,
                "market_budget": 19,
            }
        }
    }

    def fake_run(cmd, env=None):
        calls.append((list(cmd), dict(env or {})))
        return 0

    monkeypatch.setattr(run_scheduled, "run_lock", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(run_scheduled, "_load_tuning", lambda path=None: tuning)
    monkeypatch.setattr(run_scheduled, "_run", fake_run)
    monkeypatch.setattr(
        run_scheduled,
        "_get_cli_with_tuning",
        lambda job_mode, tuning: (19, 0.55, 11),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_scheduled.py",
            "--job",
            "intraday",
            "--python",
            "py",
            "--no_telegram",
            "--limit",
            "5",
            "--lookback_days",
            "2",
            "--min_k",
            "2",
        ],
    )

    try:
        run_scheduled.main()
    except SystemExit as exc:
        assert int(exc.code) == 0

    assert len(calls) == 2

    refresh_cmd, refresh_env = calls[0]
    infer_cmd, infer_env = calls[1]

    assert refresh_cmd[:4] == ["py", "main.py", "--mode", "refresh-db"]
    assert infer_cmd[:4] == ["py", "main.py", "--mode", "intraday"]
    assert "--no_auto_refresh" in infer_cmd
    assert "--allow_stale_data" not in infer_cmd
    assert "--no_telegram" in refresh_cmd
    assert "--no_telegram" in infer_cmd
    assert "19" in refresh_cmd
    assert "11" in refresh_cmd
    assert "0.55" in refresh_cmd
    assert refresh_env["AETHER_SELECTION_CORR_MAX"] == "0.42"
    assert infer_env["AETHER_SELECTION_CORR_MAX"] == "0.42"
    assert infer_env["AETHER_SELECTION_BUCKET_QUOTAS_INTRADAY"] == "high:2,mid:2,low:1"


def test_run_scheduled_reruns_watch_only_on_stale_abort(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    run_scheduled = _load_run_scheduled()

    calls = []
    responses = iter([0, 2, 0])

    def fake_run(cmd, env=None):
        calls.append((list(cmd), dict(env or {})))
        return next(responses)

    monkeypatch.setattr(run_scheduled, "run_lock", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(run_scheduled, "_load_tuning", lambda path=None: {})
    monkeypatch.setattr(run_scheduled, "_run", fake_run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_scheduled.py",
            "--job",
            "morning-report",
            "--python",
            "py",
            "--skip_aux",
            "--skip_pattern_followers",
            "--skip_pump_radar",
        ],
    )

    try:
        run_scheduled.main()
    except SystemExit as exc:
        assert int(exc.code) == 0

    assert len(calls) == 3

    first_infer_cmd, first_infer_env = calls[1]
    rerun_cmd, rerun_env = calls[2]

    assert first_infer_cmd[:4] == ["py", "main.py", "--mode", "morning-report"]
    assert "--skip_aux" in first_infer_cmd
    assert "--skip_pattern_followers" in first_infer_cmd
    assert "--skip_pump_radar" in first_infer_cmd
    assert "--allow_stale_data" not in first_infer_cmd
    assert "AETHER_RUNTIME_WATCH_ONLY" not in first_infer_env

    assert rerun_cmd[:4] == ["py", "main.py", "--mode", "morning-report"]
    assert "--allow_stale_data" in rerun_cmd
    assert rerun_env["AETHER_RUNTIME_WATCH_ONLY"] == "1"
    assert rerun_env["AETHER_MIN_REC_MODE"] == "watch"


def test_run_scheduled_can_allow_stale_after_refresh_failure(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    run_scheduled = _load_run_scheduled()

    calls = []
    responses = iter([1, 0])

    def fake_run(cmd, env=None):
        calls.append((list(cmd), dict(env or {})))
        return next(responses)

    monkeypatch.setattr(run_scheduled, "run_lock", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(run_scheduled, "_load_tuning", lambda path=None: {})
    monkeypatch.setattr(run_scheduled, "_run", fake_run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_scheduled.py",
            "--job",
            "intraday",
            "--python",
            "py",
            "--allow_stale_on_refresh_fail",
        ],
    )

    try:
        run_scheduled.main()
    except SystemExit as exc:
        assert int(exc.code) == 0

    assert len(calls) == 2
    infer_cmd, infer_env = calls[1]
    assert infer_cmd[:4] == ["py", "main.py", "--mode", "intraday"]
    assert "--allow_stale_data" in infer_cmd
    assert "AETHER_RUNTIME_WATCH_ONLY" not in infer_env


def test_run_scheduled_dry_run_prints_commands_without_execution(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)

    run_scheduled = _load_run_scheduled()

    def fail_run(cmd, env=None):
        raise AssertionError("dry_run should not execute subprocesses")

    monkeypatch.setattr(run_scheduled, "run_lock", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(run_scheduled, "_load_tuning", lambda path=None: {})
    monkeypatch.setattr(run_scheduled, "_run", fail_run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_scheduled.py",
            "--job",
            "intraday",
            "--python",
            "py",
            "--dry_run",
        ],
    )

    try:
        run_scheduled.main()
    except SystemExit as exc:
        assert int(exc.code) == 0

    payload = capsys.readouterr().out
    assert '"refresh_cmd"' in payload
    assert '"infer_cmd"' in payload
    assert '"stale_rerun_cmd"' in payload
    assert '"job": "intraday"' in payload


def test_refresh_db_offline_persists_metrics_before_exit(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    run_scheduled = _load_run_scheduled()
    import utils.scheduled_modes as scheduled_modes

    captured = {}

    class Args:
        no_telegram = True
        send_telegram = False
        timeout_sec = 0
        offline_ok = False
        limit = 8
        lookback_days = 1
        market_budget = 0
        refresh_tv_hours = 24
        refresh_top_n = 16
        rotation_keep = 0.7
        refresh_days = 3

    def fake_persist(mode, markets, run_meta, recs):
        captured["mode"] = mode
        captured["markets"] = list(markets)
        captured["run_meta"] = dict(run_meta)
        return {}, "analysis/output_contract_refresh-db.json"

    monkeypatch.setattr("utils.netcheck.resolution_status", lambda *args, **kwargs: {
        "ok": False,
        "attempts": 3,
        "error": "name or service not known",
        "ips": [],
    })
    monkeypatch.setattr(run_scheduled, "run_lock", lambda *args, **kwargs: nullcontext())

    try:
        scheduled_modes.run_refresh_db_mode(
            Args(),
            config=type("Cfg", (), {"Recommender": type("Rec", (), {"DNS_RESOLVE_RETRIES": 3, "DNS_RESOLVE_RETRY_DELAY_SEC": 0.0})(), "Device": type("Dev", (), {"DEVICE": "cpu"})()}),
            logger=type("Log", (), {"info": lambda *a, **k: None, "warning": lambda *a, **k: None, "error": lambda *a, **k: None, "debug": lambda *a, **k: None})(),
            collect_refresh_batch=lambda *a, **k: [],
            summarize_refresh_results=lambda *a, **k: {},
            alert_index_refresh_failures=lambda *a, **k: None,
            persist_run_outputs=fake_persist,
            send_refresh_done_alert=lambda *a, **k: None,
        )
    except SystemExit as exc:
        assert int(exc.code) == 2

    assert captured["mode"] == "refresh-db"
    assert captured["markets"] == []
    assert captured["run_meta"]["network_status"] == "offline"
    assert captured["run_meta"]["auto_refresh_skipped_offline"] is True
    assert captured["run_meta"]["auto_refresh_dns_error"] == "name or service not known"
