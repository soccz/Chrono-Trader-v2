import importlib.util
from pathlib import Path


def _load_ops_preflight():
    module_path = Path(__file__).resolve().parents[1] / "utils" / "ops_preflight.py"
    spec = importlib.util.spec_from_file_location("test_ops_preflight_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_run_ops_dry_plan_parses_json_payload(monkeypatch, tmp_path):
    ops_preflight = _load_ops_preflight()

    class DummyRunOpsJob:
        @staticmethod
        def build_ops_command(argv, repo_root=None, env=None):
            return ["py", "scripts/run_scheduled.py", *list(argv)]

    class DummyResult:
        returncode = 0
        stdout = '[ts] Running refresh-db\n{"job":"intraday","refresh_cmd":["py","main.py"],"infer_cmd":["py","main.py","--mode","intraday"]}'
        stderr = ""

    monkeypatch.setattr(ops_preflight, "_load_module", lambda *args, **kwargs: DummyRunOpsJob)

    payload = ops_preflight.run_ops_dry_plan(
        "intraday",
        repo_root=tmp_path,
        runner=lambda *args, **kwargs: DummyResult(),
    )

    assert payload["success"] is True
    assert payload["plan"]["job"] == "intraday"
    assert payload["command"][-1] == "--dry_run"


def test_build_ops_preflight_marks_ready_but_stale(monkeypatch, tmp_path):
    ops_preflight = _load_ops_preflight()

    class DummyOpsDoctor:
        @staticmethod
        def build_report(repo_root=None, max_age_intraday_h=5.0, max_age_morning_h=30.0):
            return {
                "ts": "2026-03-19T00:00:00+00:00",
                "core_ok": True,
                "outputs_ok": False,
                "health": {
                    "intraday": {"status": "stale"},
                    "morning": {"status": "ok"},
                },
                "manifests": {
                    "intraday": True,
                    "morning": True,
                    "refresh-db": True,
                },
                "venv": {"transplanted": True},
                "python": {"path": "/tmp/.venv_local/bin/python"},
            }

    monkeypatch.setattr(
        ops_preflight,
        "_load_module",
        lambda module_name, relative_path: DummyOpsDoctor if relative_path == "scripts/ops_doctor.py" else None,
    )
    monkeypatch.setattr(
        ops_preflight,
        "run_ops_dry_plan",
        lambda *args, **kwargs: {
            "success": True,
            "plan": {"job": "intraday"},
            "command": ["py", "scripts/run_scheduled.py", "--job", "intraday", "--dry_run"],
            "stdout": "",
            "stderr": "",
            "returncode": 0,
        },
    )

    payload = ops_preflight.build_ops_preflight("intraday", repo_root=tmp_path)

    assert payload["success"] is True
    assert payload["status"] == "ready_but_stale"
    assert payload["ready_to_run"] is True
    assert payload["next_action"]["action"] == "run_now"
    codes = {issue["code"] for issue in payload["issues"]}
    assert "target_stale" in codes
    assert "legacy_venv_transplanted" in codes


def test_build_ops_readiness_collects_both_jobs(monkeypatch, tmp_path):
    ops_preflight = _load_ops_preflight()

    monkeypatch.setattr(
        ops_preflight,
        "build_ops_preflight",
        lambda job, **kwargs: {
            "success": True,
            "status": "ready_to_run" if job == "morning-report" else "ready_but_stale",
            "job": job,
            "target": "morning" if job == "morning-report" else "intraday",
            "age_hours": 1.5 if job == "morning-report" else 7.25,
            "ready_to_run": True,
            "next_action": {"action": "monitor_only" if job == "morning-report" else "run_now", "label": "x"},
            "blocking_errors": 0,
            "warning_count": 0 if job == "morning-report" else 1,
            "checked_at": "2026-03-19T00:00:00+00:00",
        },
    )

    payload = ops_preflight.build_ops_readiness(repo_root=tmp_path)

    assert payload["success"] is True
    assert len(payload["jobs"]) == 2
    assert payload["summary"]["jobs_total"] == 2
    assert payload["summary"]["stale_n"] == 1
    assert payload["summary"]["primary_age_hours"] == 7.25
    assert payload["jobs"][0]["job"] == "intraday"
    assert payload["jobs"][1]["job"] == "morning-report"
