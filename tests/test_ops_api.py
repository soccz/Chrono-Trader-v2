import importlib.util
from pathlib import Path


def _load_app_module():
    module_path = Path(__file__).resolve().parents[1] / "app.py"
    spec = importlib.util.spec_from_file_location("test_app_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_api_health_data_pipeline_includes_refresh_db(monkeypatch):
    app_module = _load_app_module()

    def fake_mode_health(mode, max_age_h, now_utc=None):
        if mode == "refresh-db":
            return {
                "ok": False,
                "mode": mode,
                "status": "offline",
                "age_hours": 0.4,
                "recs_n": 0,
                "has_watch": False,
                "kept": 0,
                "dropped": 0,
                "used_fallback": False,
                "auto_refresh_skipped_offline": True,
                "auto_refresh_dns_error": "name or service not known",
                "message": "refresh-db: offline",
            }
        return {
            "ok": True,
            "mode": mode,
            "status": "ok",
            "age_hours": 0.2,
            "recs_n": 1,
            "has_watch": False,
            "kept": 8,
            "dropped": 0,
            "used_fallback": False,
            "message": f"{mode}: ok",
        }

    import utils.ops_health as ops_health
    monkeypatch.setattr(ops_health, "mode_health", fake_mode_health)

    client = app_module.app.test_client()
    response = client.get("/api/health/data-pipeline")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["success"] is True
    assert payload["status"] == "degraded"
    assert payload["refresh_db"]["status"] == "offline"
    assert payload["refresh_db"]["auto_refresh_skipped_offline"] is True


def test_api_ops_plan_returns_parsed_plan(monkeypatch):
    app_module = _load_app_module()

    class DummyResult:
        returncode = 0
        stderr = ""
        stdout = '[ts] Running refresh-db\n{"job":"intraday","refresh_cmd":["py","main.py"],"infer_cmd":["py","main.py","--mode","intraday"]}'

    monkeypatch.setattr(app_module.subprocess, "run", lambda *args, **kwargs: DummyResult())

    client = app_module.app.test_client()
    response = client.post("/api/ops/plan", json={"job": "intraday"})
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["success"] is True
    assert payload["plan"]["job"] == "intraday"
    assert payload["plan"]["infer_cmd"][-1] == "intraday"


def test_api_ops_run_dispatches_task_runner(monkeypatch):
    app_module = _load_app_module()
    import utils.ops_preflight as ops_preflight

    captured = {}

    def fake_start_task(command, task_name="task", cwd=None, env=None, **kwargs):
        captured["command"] = list(command)
        captured["task_name"] = task_name
        captured["cwd"] = cwd
        captured["env"] = env
        captured["kwargs"] = kwargs
        return {"success": True, "message": f"{task_name} started"}

    monkeypatch.setattr(app_module.task_runner, "start_task", fake_start_task)
    monkeypatch.setattr(
        ops_preflight,
        "build_ops_preflight",
        lambda job, extra_args=None, repo_root=None: {
            "success": True,
            "status": "ready_to_run",
            "job": job,
            "target": "morning",
            "ready_to_run": True,
            "doctor": {"core_ok": True},
            "dry_run": {"success": True},
            "issues": [],
            "checked_at": "2026-03-19T00:00:00+00:00",
        },
    )

    client = app_module.app.test_client()
    response = client.post("/api/ops/run", json={"job": "morning-report"})
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["success"] is True
    assert captured["task_name"] == "Ops morning-report"
    assert captured["cwd"]
    assert captured["command"][-2:] == ["--job", "morning-report"]
    assert payload["preflight_status"] == "ready_to_run"
    assert payload["forced"] is False


def test_api_ops_run_blocks_when_preflight_is_not_ready(monkeypatch):
    app_module = _load_app_module()
    import utils.ops_preflight as ops_preflight

    monkeypatch.setattr(
        ops_preflight,
        "build_ops_preflight",
        lambda job, extra_args=None, repo_root=None: {
            "success": True,
            "status": "blocked",
            "job": job,
            "target": "intraday",
            "ready_to_run": False,
            "doctor": {"core_ok": False},
            "dry_run": {"success": False},
            "issues": [{"level": "error", "code": "core_not_ready", "message": "runtime bad"}],
            "checked_at": "2026-03-19T00:00:00+00:00",
        },
    )

    client = app_module.app.test_client()
    response = client.post("/api/ops/run", json={"job": "intraday"})
    payload = response.get_json()

    assert response.status_code == 409
    assert payload["success"] is False
    assert payload["blocked"] is True
    assert payload["preflight"]["status"] == "blocked"


def test_api_ops_run_force_bypasses_preflight_block(monkeypatch):
    app_module = _load_app_module()
    import utils.ops_preflight as ops_preflight

    captured = {}

    def fake_start_task(command, task_name="task", cwd=None, env=None, **kwargs):
        captured["command"] = list(command)
        captured["task_name"] = task_name
        return {"success": True, "message": "started"}

    monkeypatch.setattr(app_module.task_runner, "start_task", fake_start_task)
    monkeypatch.setattr(
        ops_preflight,
        "build_ops_preflight",
        lambda job, extra_args=None, repo_root=None: {
            "success": True,
            "status": "blocked",
            "job": job,
            "target": "intraday",
            "ready_to_run": False,
            "doctor": {"core_ok": False},
            "dry_run": {"success": False},
            "issues": [{"level": "error", "code": "core_not_ready", "message": "runtime bad"}],
            "checked_at": "2026-03-19T00:00:00+00:00",
        },
    )

    client = app_module.app.test_client()
    response = client.post("/api/ops/run", json={"job": "intraday", "force": True})
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["success"] is True
    assert payload["forced"] is True
    assert captured["task_name"] == "Ops intraday"


def test_api_status_exposes_task_name_and_command(monkeypatch):
    app_module = _load_app_module()

    monkeypatch.setattr(
        app_module.task_runner,
        "get_status",
        lambda: {
            "status": "running",
            "task_name": "Ops intraday",
            "task_key": "ops",
            "command": ["python", "scripts/run_ops_job.py", "--job", "intraday"],
            "start_time": "2026-03-19T00:00:00",
            "end_time": None,
            "returncode": None,
            "running": True,
        },
    )

    client = app_module.app.test_client()
    response = client.get("/api/status")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["task_name"] == "Ops intraday"
    assert payload["task_key"] == "ops"
    assert payload["command"][-1] == "intraday"


def test_api_trigger_prediction_uses_intraday_ops(monkeypatch):
    app_module = _load_app_module()
    import utils.ops_preflight as ops_preflight

    captured = {}

    def fake_start_task(command, task_name="task", cwd=None, env=None, **kwargs):
        captured["command"] = list(command)
        captured["task_name"] = task_name
        captured["cwd"] = cwd
        captured["kwargs"] = kwargs
        return {"success": True, "message": f"{task_name} started"}

    monkeypatch.setattr(app_module.task_runner, "start_task", fake_start_task)
    monkeypatch.setattr(
        ops_preflight,
        "build_ops_preflight",
        lambda job, extra_args=None, repo_root=None: {
            "success": True,
            "status": "ready_but_stale",
            "job": job,
            "target": "intraday",
            "ready_to_run": True,
            "doctor": {"core_ok": True},
            "dry_run": {"success": True},
            "issues": [],
            "checked_at": "2026-03-19T00:00:00+00:00",
        },
    )

    client = app_module.app.test_client()
    response = client.post("/api/trigger/prediction")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["success"] is True
    assert captured["task_name"] == "Ops intraday"
    assert captured["command"][-2:] == ["--job", "intraday"]
    assert payload["preflight_status"] == "ready_but_stale"


def test_api_continuous_aliases_intraday_ops(monkeypatch):
    app_module = _load_app_module()
    import utils.ops_preflight as ops_preflight

    captured = {}

    def fake_start_task(command, task_name="task", cwd=None, env=None, **kwargs):
        captured["command"] = list(command)
        captured["task_name"] = task_name
        captured["cwd"] = cwd
        captured["kwargs"] = kwargs
        return {"success": True, "message": f"{task_name} started"}

    monkeypatch.setattr(app_module.task_runner, "start_task", fake_start_task)
    monkeypatch.setattr(
        ops_preflight,
        "build_ops_preflight",
        lambda job, extra_args=None, repo_root=None: {
            "success": True,
            "status": "ready_to_run",
            "job": job,
            "target": "intraday",
            "ready_to_run": True,
            "doctor": {"core_ok": True},
            "dry_run": {"success": True},
            "issues": [],
            "checked_at": "2026-03-19T00:00:00+00:00",
        },
    )

    client = app_module.app.test_client()
    response = client.post("/api/continuous", headers={"X-API-Key": "chrono-trader-api-key-2024"})
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["success"] is True
    assert captured["task_name"] == "Ops intraday"
    assert captured["command"][-3:] == ["--job", "intraday", "--no_telegram"]


def test_api_trigger_prediction_blocks_when_preflight_is_not_ready(monkeypatch):
    app_module = _load_app_module()
    import utils.ops_preflight as ops_preflight

    monkeypatch.setattr(
        ops_preflight,
        "build_ops_preflight",
        lambda job, extra_args=None, repo_root=None: {
            "success": True,
            "status": "blocked",
            "job": job,
            "target": "intraday",
            "ready_to_run": False,
            "doctor": {"core_ok": False},
            "dry_run": {"success": False},
            "issues": [{"level": "error", "code": "core_not_ready", "message": "runtime bad"}],
            "checked_at": "2026-03-19T00:00:00+00:00",
        },
    )

    client = app_module.app.test_client()
    response = client.post("/api/trigger/prediction")
    payload = response.get_json()

    assert response.status_code == 409
    assert payload["success"] is False
    assert payload["blocked"] is True


def test_api_train_uses_current_python_and_repo_cwd(monkeypatch):
    app_module = _load_app_module()

    captured = {}

    def fake_start_task(command, task_name="task", cwd=None, env=None, **kwargs):
        captured["command"] = list(command)
        captured["task_name"] = task_name
        captured["cwd"] = cwd
        captured["kwargs"] = kwargs
        return {"success": True, "message": f"{task_name} started"}

    monkeypatch.setattr(app_module.task_runner, "start_task", fake_start_task)

    client = app_module.app.test_client()
    response = client.post("/api/train", json={"epochs": 7, "tune": True}, headers={"X-API-Key": "chrono-trader-api-key-2024"})
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["success"] is True
    assert captured["task_name"] == "Training"
    assert captured["cwd"]
    assert captured["command"][1:4] == ["-u", "main.py", "--mode"]
    assert captured["command"][4] == "train"
    assert "--tune" in captured["command"]
    assert captured["command"][-2:] == ["--epochs", "7"]


def test_api_backtest_uses_current_python_and_repo_cwd(monkeypatch):
    app_module = _load_app_module()

    captured = {}

    def fake_start_task(command, task_name="task", cwd=None, env=None, **kwargs):
        captured["command"] = list(command)
        captured["task_name"] = task_name
        captured["cwd"] = cwd
        captured["kwargs"] = kwargs
        return {"success": True, "message": f"{task_name} started"}

    monkeypatch.setattr(app_module.task_runner, "start_task", fake_start_task)

    client = app_module.app.test_client()
    response = client.post("/api/backtest", json={"days": 14}, headers={"X-API-Key": "chrono-trader-api-key-2024"})
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["success"] is True
    assert captured["task_name"] == "Backtest"
    assert captured["cwd"]
    assert captured["command"][1:4] == ["-u", "main.py", "--mode"]
    assert captured["command"][4] == "backtest"
    assert captured["command"][-2:] == ["--days", "14"]


def test_api_ops_history_returns_rows(monkeypatch):
    app_module = _load_app_module()

    import utils.ops_history as ops_history

    monkeypatch.setattr(
        ops_history,
        "read_recent_ops_runs",
        lambda limit_per_mode=6: [
            {
                "mode": "intraday",
                "ts": "2026-03-19T11:00:00+00:00",
                "recs_n": 1,
                "kept": 8,
                "dropped": 1,
                "has_watch": False,
                "has_forced": False,
                "used_fallback": False,
                "elapsed_sec": 12.3,
                "markets_n": 8,
            }
        ],
    )

    client = app_module.app.test_client()
    response = client.get("/api/ops/history?limit=4")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["success"] is True
    assert payload["runs"][0]["mode"] == "intraday"


def test_api_system_resources_returns_snapshot(monkeypatch):
    app_module = _load_app_module()

    import utils.system_resources as system_resources

    monkeypatch.setattr(
        system_resources,
        "snapshot",
        lambda path=".": {"cpu_percent": 11.1, "memory_percent": 22.2, "disk_percent": 33.3},
    )

    client = app_module.app.test_client()
    response = client.get("/api/system/resources")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["success"] is True
    assert payload["data"]["cpu_percent"] == 11.1
    assert payload["data"]["memory_percent"] == 22.2
    assert payload["data"]["disk_percent"] == 33.3


def test_api_ops_preflight_returns_readiness_summary(monkeypatch):
    app_module = _load_app_module()

    import utils.ops_preflight as ops_preflight

    monkeypatch.setattr(
        ops_preflight,
        "build_ops_preflight",
        lambda job, extra_args=None, repo_root=None: {
            "success": True,
            "status": "ready_but_stale",
            "job": job,
            "target": "intraday",
            "ready_to_run": True,
            "doctor": {
                "core_ok": True,
                "health": {"intraday": {"status": "stale"}},
                "manifests": {"intraday": True},
                "python": {"path": "/tmp/.venv_local/bin/python"},
            },
            "dry_run": {
                "success": True,
                "plan": {"job": job},
            },
            "issues": [{"level": "warn", "code": "target_stale", "message": "needs run"}],
            "checked_at": "2026-03-19T00:00:00+00:00",
        },
    )

    client = app_module.app.test_client()
    response = client.post("/api/ops/preflight", json={"job": "intraday"})
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["success"] is True
    assert payload["status"] == "ready_but_stale"
    assert payload["doctor"]["core_ok"] is True
    assert payload["issues"][0]["code"] == "target_stale"


def test_api_ops_readiness_returns_job_rows(monkeypatch):
    app_module = _load_app_module()

    import utils.ops_preflight as ops_preflight

    monkeypatch.setattr(
        ops_preflight,
        "build_ops_readiness",
        lambda repo_root=None: {
            "success": True,
            "jobs": [
                {"job": "intraday", "status": "ready_but_stale", "next_action": {"label": "Run ops now"}, "blocking_errors": 0, "warning_count": 1},
                {"job": "morning-report", "status": "ready_to_run", "next_action": {"label": "Monitor only"}, "blocking_errors": 0, "warning_count": 0},
            ],
            "checked_at": "2026-03-19T00:00:00+00:00",
        },
    )

    client = app_module.app.test_client()
    response = client.get("/api/ops/readiness")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["success"] is True
    assert len(payload["jobs"]) == 2
    assert payload["jobs"][0]["job"] == "intraday"


def test_api_ops_next_action_runs_job_when_recommended(monkeypatch):
    app_module = _load_app_module()
    import utils.ops_preflight as ops_preflight

    captured = {}

    def fake_start_task(command, task_name="task", cwd=None, env=None, **kwargs):
        captured["command"] = list(command)
        captured["task_name"] = task_name
        return {"success": True, "message": "started"}

    monkeypatch.setattr(app_module.task_runner, "start_task", fake_start_task)
    monkeypatch.setattr(
        ops_preflight,
        "build_ops_preflight",
        lambda job, extra_args=None, repo_root=None: {
            "success": True,
            "status": "ready_but_stale",
            "job": job,
            "target": "intraday",
            "ready_to_run": True,
            "next_action": {"action": "run_now", "label": "Run ops now"},
            "doctor": {"core_ok": True},
            "dry_run": {"success": True},
            "issues": [],
            "checked_at": "2026-03-19T00:00:00+00:00",
        },
    )

    client = app_module.app.test_client()
    response = client.post("/api/ops/next-action", json={"job": "intraday"})
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["success"] is True
    assert payload["action_taken"] == "run"
    assert captured["task_name"] == "Ops intraday"
    assert captured["command"][-2:] == ["--job", "intraday"]


def test_api_ops_next_action_returns_dry_run_for_inspection(monkeypatch):
    app_module = _load_app_module()
    import utils.ops_preflight as ops_preflight

    monkeypatch.setattr(
        ops_preflight,
        "build_ops_preflight",
        lambda job, extra_args=None, repo_root=None: {
            "success": True,
            "status": "blocked",
            "job": job,
            "target": "intraday",
            "ready_to_run": False,
            "next_action": {"action": "inspect_dry_run", "label": "Inspect dry-run failure"},
            "doctor": {"core_ok": True},
            "dry_run": {"success": False},
            "issues": [{"level": "error", "code": "dry_run_failed", "message": "bad dry run"}],
            "checked_at": "2026-03-19T00:00:00+00:00",
        },
    )
    monkeypatch.setattr(
        ops_preflight,
        "run_ops_dry_plan",
        lambda job, extra_args=None, repo_root=None: {
            "success": True,
            "plan": {"job": job, "infer_cmd": ["py", "main.py", "--mode", job]},
            "stdout": "",
            "stderr": "",
            "returncode": 0,
        },
    )

    client = app_module.app.test_client()
    response = client.post("/api/ops/next-action", json={"job": "intraday"})
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["success"] is True
    assert payload["action_taken"] == "inspect_dry_run"
    assert payload["dry_run"]["plan"]["job"] == "intraday"


def test_api_ops_audit_returns_recent_events(monkeypatch):
    app_module = _load_app_module()

    import utils.ops_audit as ops_audit

    monkeypatch.setattr(
        ops_audit,
        "read_recent_ops_audit",
        lambda limit=8: [
            {"event": "ops_preflight", "payload": {"job": "intraday", "preflight_status": "ready_but_stale"}},
            {"event": "ops_run_started", "payload": {"job": "intraday", "preflight_status": "ready_but_stale"}},
        ],
    )

    client = app_module.app.test_client()
    response = client.get("/api/ops/audit?limit=2")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["success"] is True
    assert len(payload["events"]) == 2
    assert payload["events"][0]["event"] == "ops_preflight"


def test_api_ops_overview_returns_combined_payload(monkeypatch):
    app_module = _load_app_module()

    import utils.ops_preflight as ops_preflight
    import utils.ops_history as ops_history
    import utils.ops_audit as ops_audit
    import scripts.ops_doctor as ops_doctor

    monkeypatch.setattr(
        ops_doctor,
        "build_report",
        lambda max_age_intraday_h=5.0, max_age_morning_h=30.0: {
            "core_ok": True,
            "outputs_ok": False,
            "python": {"path": "/tmp/.venv_local/bin/python"},
        },
    )
    monkeypatch.setattr(
        ops_preflight,
        "build_ops_readiness",
        lambda repo_root=None: {
            "success": True,
            "jobs": [{"job": "intraday", "status": "ready_but_stale", "age_hours": 6.5}],
            "summary": {"jobs_total": 1, "stale_n": 1, "primary_job": "intraday", "primary_age_hours": 6.5, "primary_action": {"label": "Run ops now"}},
        },
    )
    monkeypatch.setattr(
        ops_history,
        "read_recent_ops_runs",
        lambda limit_per_mode=6: [{"mode": "intraday", "recs_n": 1}],
    )
    monkeypatch.setattr(
        ops_audit,
        "read_recent_ops_audit",
        lambda limit=6: [{"event": "ops_preflight", "payload": {"job": "intraday"}}],
    )

    client = app_module.app.test_client()
    response = client.get("/api/ops/overview?runs_limit=3&audit_limit=2")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["success"] is True
    assert payload["runtime"]["status"] == "degraded"
    assert payload["readiness"]["summary"]["primary_job"] == "intraday"
    assert payload["readiness"]["summary"]["primary_age_hours"] == 6.5
    assert payload["readiness"]["jobs"][0]["job"] == "intraday"
    assert payload["runs"][0]["mode"] == "intraday"
    assert payload["audit"][0]["event"] == "ops_preflight"
