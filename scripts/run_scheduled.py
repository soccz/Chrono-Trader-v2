import os
import sys

# Ensure repo root is importable when invoked as `python scripts/...`.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import argparse
import json
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from utils.run_lock import run_lock


def _load_tuning(path: str = os.path.join("analysis", "ops_tuning.json")) -> Dict[str, Any]:
    try:
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _env_with_tuning(job_mode: str, tuning: Dict[str, Any], watch_only: bool) -> Dict[str, str]:
    env = dict(os.environ)

    # Ops safe-mode: outputs are watch-only (no position sizing). Intended for stale DB fallback runs.
    if watch_only:
        env["AETHER_RUNTIME_WATCH_ONLY"] = "1"
        env["AETHER_MIN_REC_MODE"] = "watch"

    key = "intraday" if job_mode == "intraday" else "morning"
    rec = (((tuning or {}).get(key) or {}).get("recommend") or {})

    # Selection tuning via env vars.
    if "selection_corr_max" in rec:
        try:
            env["AETHER_SELECTION_CORR_MAX"] = str(float(rec["selection_corr_max"]))
        except Exception:
            pass

    if "bucket_quotas_env" in rec:
        q = str(rec["bucket_quotas_env"] or "").strip()
        if q:
            if key == "intraday":
                env["AETHER_SELECTION_BUCKET_QUOTAS_INTRADAY"] = q
            else:
                env["AETHER_SELECTION_BUCKET_QUOTAS_MORNING"] = q

    return env


def _get_cli_with_tuning(job_mode: str, tuning: Dict[str, Any]) -> Tuple[Optional[int], Optional[float], Optional[int]]:
    """
    Returns (market_budget, rotation_keep, refresh_top_n) overrides if available.
    """
    key = "intraday" if job_mode == "intraday" else "morning"
    rec = (((tuning or {}).get(key) or {}).get("recommend") or {})

    market_budget = None
    rotation_keep = None
    refresh_top_n = None

    # rotation_keep is tuned
    if "rotation_keep" in rec:
        try:
            rotation_keep = float(rec["rotation_keep"])
        except Exception:
            rotation_keep = None

    if "market_budget" in rec:
        try:
            market_budget = int(rec["market_budget"])
        except Exception:
            market_budget = None

    # If you want to extend this later: infer refresh_top_n or market_budget based on forced rates, etc.
    return market_budget, rotation_keep, refresh_top_n


def _run(cmd: list, env: Optional[Dict[str, str]] = None) -> int:
    p = subprocess.run(cmd, env=env)
    return int(p.returncode)


def _env_delta(env: Optional[Dict[str, str]]) -> Dict[str, str]:
    if not env:
        return {}
    base = dict(os.environ)
    delta = {}
    for key, value in env.items():
        if base.get(key) != value and (key.startswith("AETHER_") or key in {"PYTHONUNBUFFERED", "PYTHONPATH"}):
            delta[key] = value
    return delta


def main():
    ap = argparse.ArgumentParser(description="Scheduled ops runner: refresh-db then intraday/morning-report.")
    ap.add_argument("--job", choices=["intraday", "morning-report"], required=True)
    ap.add_argument("--python", default=os.getenv("PYTHON", "python"), help="Python executable to use.")
    ap.add_argument(
        "--lock_timeout_sec",
        type=float,
        default=0.0,
        help="Prevent overlapping runs. 0 disables wait and exits immediately on lock contention.",
    )
    ap.add_argument("--timeout_sec_refresh", type=int, default=20 * 60)
    ap.add_argument("--timeout_sec_infer", type=int, default=0, help="0 = mode default inside main.py")
    ap.add_argument("--market_budget", type=int, default=0, help="0 = auto (main.py default)")
    ap.add_argument("--refresh_days", type=int, default=3)
    ap.add_argument("--refresh_top_n", type=int, default=16)
    ap.add_argument("--rotation_keep", type=float, default=0.70)
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--lookback_days", type=int, default=1)
    ap.add_argument("--min_k", type=int, default=1)
    ap.add_argument("--no_telegram", action="store_true")
    ap.add_argument("--tune_from_logs", action="store_true", help="Run auto-tune first (writes ops_tuning.json).")
    ap.add_argument("--allow_stale_on_refresh_fail", action="store_true", help="If refresh-db fails, still run inference with --allow_stale_data.")
    ap.add_argument("--skip_aux", action="store_true", help="Pass --skip_aux to morning-report inference.")
    ap.add_argument("--skip_pattern_followers", action="store_true", help="Pass --skip_pattern_followers to morning-report inference.")
    ap.add_argument("--skip_pump_radar", action="store_true", help="Pass --skip_pump_radar to morning-report inference.")
    ap.add_argument("--dry_run", action="store_true", help="Print the assembled refresh/inference commands and exit without running them.")
    args = ap.parse_args()

    with run_lock(f"aether_{args.job}", timeout_sec=float(args.lock_timeout_sec or 0.0), exit_code=0):
        os.makedirs("analysis", exist_ok=True)
        tuning_path = os.path.join("analysis", "ops_tuning.json")

        if args.tune_from_logs:
            # Best-effort: generate tuning file from recent history.
            _run([args.python, "scripts/auto_tune_ops.py", "--window_runs", "42", "--write"])

        tuning = _load_tuning(tuning_path)
        tuned_budget, tuned_rotation_keep, tuned_refresh_top_n = _get_cli_with_tuning(args.job, tuning)

        # Only let tuning override budget if the caller opted into auto budget (0).
        if int(args.market_budget) == 0 and tuned_budget is not None:
            market_budget = int(tuned_budget)
        else:
            market_budget = int(args.market_budget)
        rotation_keep = tuned_rotation_keep if tuned_rotation_keep is not None else float(args.rotation_keep)
        refresh_top_n = tuned_refresh_top_n if tuned_refresh_top_n is not None else int(args.refresh_top_n)

        common = [
            "--limit",
            str(int(args.limit)),
            "--lookback_days",
            str(int(args.lookback_days)),
            "--market_budget",
            str(int(market_budget)),
            "--refresh_top_n",
            str(int(refresh_top_n)),
            "--rotation_keep",
            str(float(rotation_keep)),
        ]
        if args.no_telegram:
            common.append("--no_telegram")

        # 1) refresh-db first
        refresh_cmd = [
            args.python,
            "main.py",
            "--mode",
            "refresh-db",
            "--timeout_sec",
            str(int(args.timeout_sec_refresh)),
            "--refresh_days",
            str(int(args.refresh_days)),
        ] + common

        ts = datetime.now(timezone.utc).isoformat()
        print(f"[{ts}] Running refresh-db: {' '.join(refresh_cmd)}", flush=True)
        refresh_env = _env_with_tuning(args.job, tuning, watch_only=False)
        refresh_failed = False

        # 2) inference
        infer_cmd = [
            args.python,
            "main.py",
            "--mode",
            args.job,
            "--timeout_sec",
            str(int(args.timeout_sec_infer)),
            "--min_k",
            str(int(args.min_k)),
            "--no_auto_refresh",  # refresh-db handled the refresh step; keeps inference more stable/offline-friendly.
        ] + common

        # Morning-report optional knobs
        if args.job == "morning-report":
            if args.skip_aux:
                infer_cmd.append("--skip_aux")
            if args.skip_pattern_followers:
                infer_cmd.append("--skip_pattern_followers")
            if args.skip_pump_radar:
                infer_cmd.append("--skip_pump_radar")

        if refresh_failed and args.allow_stale_on_refresh_fail:
            infer_cmd.append("--allow_stale_data")

        ts = datetime.now(timezone.utc).isoformat()
        print(f"[{ts}] Running inference: {' '.join(infer_cmd)}", flush=True)
        if args.dry_run:
            rerun_cmd = infer_cmd + ["--allow_stale_data"] if "--allow_stale_data" not in infer_cmd else list(infer_cmd)
            payload = {
                "job": args.job,
                "python": args.python,
                "refresh_cmd": refresh_cmd,
                "refresh_env": _env_delta(refresh_env),
                "infer_cmd": infer_cmd,
                "infer_env": _env_delta(_env_with_tuning(args.job, tuning, watch_only=False)),
                "stale_rerun_cmd": rerun_cmd,
                "stale_rerun_env": _env_delta(_env_with_tuning(args.job, tuning, watch_only=True)),
                "tuning_path": tuning_path,
                "market_budget": market_budget,
                "rotation_keep": rotation_keep,
                "refresh_top_n": refresh_top_n,
            }
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            raise SystemExit(0)

        rc_refresh = _run(refresh_cmd, env=refresh_env)
        refresh_failed = rc_refresh != 0
        if refresh_failed and args.allow_stale_on_refresh_fail and "--allow_stale_data" not in infer_cmd:
            infer_cmd.append("--allow_stale_data")

        # Do NOT force watch-only just because refresh-db failed; the DB may still be fresh enough.
        # We rely on the main.py freshness gate. Only if it aborts (exit=2) do we rerun in watch-only mode.
        infer_env = _env_with_tuning(args.job, tuning, watch_only=False)
        rc_infer = _run(infer_cmd, env=infer_env)

        # If inference aborted on stale data, rerun once in watch-only allow-stale mode (explicit).
        if rc_infer == 2 and not ("--allow_stale_data" in infer_cmd):
            ts = datetime.now(timezone.utc).isoformat()
            print(f"[{ts}] Inference exited 2; rerunning once with watch-only + allow_stale_data", flush=True)
            infer_cmd2 = infer_cmd + ["--allow_stale_data"]
            env2 = _env_with_tuning(args.job, tuning, watch_only=True)
            rc_infer = _run(infer_cmd2, env=env2)

        # Prefer surfacing refresh failures only if inference also failed.
        if rc_infer != 0:
            raise SystemExit(int(rc_infer))
        if refresh_failed:
            # inference succeeded, but refresh failed: still mark success to keep pipelines green.
            raise SystemExit(0)
        raise SystemExit(0)


if __name__ == "__main__":
    main()
