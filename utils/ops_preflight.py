import importlib.util
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name: str, relative_path: str):
    module_path = _REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _extract_json_payload(stdout: str) -> Optional[Dict[str, Any]]:
    text = str(stdout or "").strip()
    if not text:
        return None
    start = text.find("{")
    if start < 0:
        return None
    try:
        payload = json.loads(text[start:])
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _target_health_key(job: str) -> str:
    return "morning" if str(job) == "morning-report" else "intraday"


def _recommended_action(status: str, issues: list[Dict[str, str]]) -> Dict[str, str]:
    codes = {str(issue.get("code") or "") for issue in issues}
    if status == "blocked":
        if "core_not_ready" in codes:
            return {
                "action": "fix_core",
                "label": "Fix runtime/db/model first",
                "reason": "core runtime assets are not ready",
            }
        if "dry_run_failed" in codes:
            return {
                "action": "inspect_dry_run",
                "label": "Inspect dry-run failure",
                "reason": "scheduled ops dry-run could not be assembled cleanly",
            }
        return {
            "action": "inspect_preflight",
            "label": "Inspect preflight issues",
            "reason": "blocking issues remain before ops can run",
        }
    if status == "ready_but_stale":
        return {
            "action": "run_now",
            "label": "Run ops now",
            "reason": "core is ready; outputs are stale and need a fresh run",
        }
    if status == "ready_but_empty":
        return {
            "action": "run_bootstrap",
            "label": "Run bootstrap ops",
            "reason": "core is ready but target outputs are still missing or empty",
        }
    return {
        "action": "monitor_only",
        "label": "Monitor only",
        "reason": "runtime and outputs look healthy",
    }


def _collect_issues(job: str, report: Dict[str, Any], plan_result: Dict[str, Any]) -> list[Dict[str, str]]:
    issues: list[Dict[str, str]] = []
    target_key = _target_health_key(job)
    target_health = ((report or {}).get("health") or {}).get(target_key) or {}
    manifests = (report or {}).get("manifests") or {}
    venv = (report or {}).get("venv") or {}

    if not report.get("core_ok"):
        issues.append({"level": "error", "code": "core_not_ready", "message": "runtime, db, or model assets are not ready"})
    if not plan_result.get("success"):
        issues.append({"level": "error", "code": "dry_run_failed", "message": "scheduled ops dry-run did not complete successfully"})
    if target_health.get("status") == "stale":
        issues.append({"level": "warn", "code": "target_stale", "message": f"{target_key} output is stale and needs a fresh run"})
    elif target_health.get("status") in {"missing", "empty_recs", "critical"}:
        issues.append({"level": "warn", "code": "target_missing", "message": f"{target_key} output is not healthy yet"})
    if not manifests.get("refresh-db"):
        issues.append({"level": "warn", "code": "refresh_manifest_missing", "message": "refresh-db manifest is missing"})
    if not manifests.get(target_key):
        issues.append({"level": "warn", "code": "target_manifest_missing", "message": f"{target_key} manifest is missing"})
    if venv.get("transplanted"):
        issues.append({"level": "warn", "code": "legacy_venv_transplanted", "message": "legacy .venv points to another machine"})
    return issues


def run_ops_dry_plan(
    job: str,
    *,
    extra_args: Optional[Iterable[str]] = None,
    repo_root: Path = _REPO_ROOT,
    timeout_sec: int = 30,
    runner=subprocess.run,
) -> Dict[str, Any]:
    run_ops_job = _load_module("ops_preflight_run_ops_job", "scripts/run_ops_job.py")
    command = run_ops_job.build_ops_command(
        ["--job", str(job), *(str(arg) for arg in (extra_args or [])), "--dry_run"],
        repo_root=repo_root,
        env=os.environ,
    )
    result = runner(
        command,
        capture_output=True,
        text=True,
        timeout=int(timeout_sec),
        cwd=str(repo_root),
    )
    payload = _extract_json_payload(result.stdout or "")
    return {
        "success": result.returncode == 0 and payload is not None,
        "command": command,
        "plan": payload,
        "stdout": result.stdout or "",
        "stderr": result.stderr or "",
        "returncode": int(result.returncode),
    }


def build_ops_preflight(
    job: str,
    *,
    extra_args: Optional[Iterable[str]] = None,
    repo_root: Path = _REPO_ROOT,
    timeout_sec: int = 30,
    max_age_intraday_h: float = 5.0,
    max_age_morning_h: float = 30.0,
    runner=subprocess.run,
) -> Dict[str, Any]:
    ops_doctor = _load_module("ops_preflight_ops_doctor", "scripts/ops_doctor.py")
    report = ops_doctor.build_report(
        repo_root=repo_root,
        max_age_intraday_h=float(max_age_intraday_h),
        max_age_morning_h=float(max_age_morning_h),
    )
    plan_result = run_ops_dry_plan(
        job,
        extra_args=extra_args,
        repo_root=repo_root,
        timeout_sec=timeout_sec,
        runner=runner,
    )
    issues = _collect_issues(job, report, plan_result)
    ready = bool(report.get("core_ok")) and bool(plan_result.get("success"))
    target_key = _target_health_key(job)
    target_health_obj = (report.get("health") or {}).get(target_key) or {}
    target_health = target_health_obj.get("status")
    target_age_hours = target_health_obj.get("age_hours")

    status = "ready_to_run" if ready else "blocked"
    if ready and target_health == "stale":
        status = "ready_but_stale"
    elif ready and target_health in {"missing", "empty_recs", "critical"}:
        status = "ready_but_empty"
    next_action = _recommended_action(status, issues)

    return {
        "success": True,
        "status": status,
        "job": str(job),
        "target": target_key,
        "age_hours": target_age_hours,
        "ready_to_run": ready,
        "next_action": next_action,
        "blocking_errors": len([issue for issue in issues if issue.get("level") == "error"]),
        "warning_count": len([issue for issue in issues if issue.get("level") == "warn"]),
        "doctor": report,
        "dry_run": plan_result,
        "issues": issues,
        "checked_at": report.get("ts"),
    }


def build_ops_readiness(
    jobs: Optional[Iterable[str]] = None,
    *,
    repo_root: Path = _REPO_ROOT,
    timeout_sec: int = 30,
    max_age_intraday_h: float = 5.0,
    max_age_morning_h: float = 30.0,
    runner=subprocess.run,
) -> Dict[str, Any]:
    jobs = list(jobs or ("intraday", "morning-report"))
    rows = []
    for job in jobs:
        row = build_ops_preflight(
            str(job),
            repo_root=repo_root,
            timeout_sec=timeout_sec,
            max_age_intraday_h=max_age_intraday_h,
            max_age_morning_h=max_age_morning_h,
            runner=runner,
        )
        rows.append(row)

    return {
        "success": True,
        "jobs": rows,
        "summary": summarize_ops_readiness(rows),
        "checked_at": rows[0].get("checked_at") if rows else None,
    }


def summarize_ops_readiness(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    items = list(rows or [])
    blocked = [row for row in items if row.get("status") == "blocked"]
    stale = [row for row in items if row.get("status") == "ready_but_stale"]
    empty = [row for row in items if row.get("status") == "ready_but_empty"]
    ready = [row for row in items if row.get("status") == "ready_to_run"]

    primary = None
    for group in (blocked, stale, empty, ready):
        if group:
            primary = group[0]
            break

    next_action = (primary or {}).get("next_action") or {}
    return {
        "jobs_total": len(items),
        "blocked_n": len(blocked),
        "stale_n": len(stale),
        "empty_n": len(empty),
        "ready_n": len(ready),
        "primary_job": (primary or {}).get("job"),
        "primary_status": (primary or {}).get("status"),
        "primary_age_hours": (primary or {}).get("age_hours"),
        "primary_action": next_action,
    }
