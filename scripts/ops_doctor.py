import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure repo root is importable when invoked as `python scripts/...`.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils.config import config
from utils.model_artifacts import build_model_artifact_audit
from utils.ops_health import mode_health
from utils.output_contract import read_output_manifest


def _repo_root() -> Path:
    return Path(_REPO_ROOT)


def _file_report(path_str: str) -> Dict[str, Any]:
    path = Path(path_str)
    report: Dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return report
    try:
        stat = path.stat()
        report["size_bytes"] = int(stat.st_size)
        report["mtime"] = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
    except Exception as exc:
        report["error"] = str(exc)
    return report


def _parse_pyvenv_cfg(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not path.exists():
        return out
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            out[key.strip()] = value.strip()
    except Exception:
        return {}
    return out


def _venv_report(repo_root: Path) -> Dict[str, Any]:
    venv_root = repo_root / ".venv"
    cfg = _parse_pyvenv_cfg(venv_root / "pyvenv.cfg")
    home = cfg.get("home", "")
    executable = cfg.get("executable", "")
    transplanted = False
    reasons = []

    if home and not Path(home).exists():
        transplanted = True
        reasons.append(f"missing_home:{home}")
    if executable and not Path(executable).exists():
        transplanted = True
        reasons.append(f"missing_executable:{executable}")

    return {
        "path": str(venv_root),
        "exists": venv_root.exists(),
        "cfg": cfg,
        "transplanted": transplanted,
        "reasons": reasons,
    }


def _resolve_python(repo_root: Path) -> Dict[str, Any]:
    try:
        module_path = _repo_root() / "scripts" / "run_ops_job.py"
        spec = importlib.util.spec_from_file_location("ops_doctor_run_ops_job", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"failed to load {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        python_bin = module.resolve_repo_python(repo_root=repo_root)
        return {"ok": True, "path": str(python_bin)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def build_report(
    *,
    repo_root: Optional[Path] = None,
    max_age_intraday_h: float = 5.0,
    max_age_morning_h: float = 30.0,
) -> Dict[str, Any]:
    repo_root = Path(repo_root) if repo_root else _repo_root()
    python_report = _resolve_python(repo_root)
    db_report = _file_report(str(repo_root / config.General.DB_PATH))
    model_main = _file_report(str(repo_root / config.General.MODEL_PATH))
    model_short = _file_report(str(repo_root / config.General.MODEL_PATH_SHORT))
    model_audit = build_model_artifact_audit(str(repo_root))
    venv_report = _venv_report(repo_root)

    intraday = mode_health("intraday", max_age_intraday_h)
    morning = mode_health("morning", max_age_morning_h)

    manifests = {
        "intraday": read_output_manifest("intraday"),
        "morning": read_output_manifest("morning"),
        "refresh-db": read_output_manifest("refresh-db"),
    }

    core_ok = bool(python_report.get("ok")) and bool(db_report.get("exists")) and bool(model_main.get("exists"))
    outputs_ok = bool(intraday.get("ok")) and bool(morning.get("ok"))

    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "core_ok": core_ok,
        "outputs_ok": outputs_ok,
        "python": python_report,
        "venv": venv_report,
        "db": db_report,
        "models": {
            "main": model_main,
            "short": model_short,
            "audit": model_audit,
        },
        "health": {
            "intraday": intraday,
            "morning": morning,
        },
        "manifests": {
            name: bool(value) for name, value in manifests.items()
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Local ops doctor: inspect runtime, DB, models, outputs, and venv health.")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--strict_outputs", action="store_true", help="Treat stale/missing intraday or morning outputs as a failing condition.")
    ap.add_argument("--max_age_intraday_h", type=float, default=5.0)
    ap.add_argument("--max_age_morning_h", type=float, default=30.0)
    args = ap.parse_args()

    report = build_report(
        max_age_intraday_h=float(args.max_age_intraday_h),
        max_age_morning_h=float(args.max_age_morning_h),
    )
    overall_ok = bool(report.get("core_ok"))
    if args.strict_outputs:
        overall_ok = overall_ok and bool(report.get("outputs_ok"))

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"[ops-doctor] core_ok={report['core_ok']} outputs_ok={report['outputs_ok']}")
        print(f"[ops-doctor] python={report['python']}")
        print(f"[ops-doctor] db={report['db']}")
        print(f"[ops-doctor] models={report['models']}")
        audit = report["models"].get("audit", {})
        print(
            f"[ops-doctor] model_compat={audit.get('compatible_count', 0)}/"
            f"{audit.get('total_count', 0)} compatible"
        )
        print(f"[ops-doctor] venv_transplanted={report['venv'].get('transplanted')} reasons={report['venv'].get('reasons')}")
        print(f"[ops-doctor] intraday={report['health']['intraday'].get('message')}")
        print(f"[ops-doctor] morning={report['health']['morning'].get('message')}")
        print(f"[ops-doctor] manifests={report['manifests']}")

    raise SystemExit(0 if overall_ok else 2)


if __name__ == "__main__":
    main()
