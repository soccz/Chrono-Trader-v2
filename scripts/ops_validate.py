import argparse
import json
import os
import subprocess
from typing import Dict, Any, Optional


def _run(cmd: list, env: Optional[Dict[str, str]] = None) -> int:
    p = subprocess.run(cmd, env=env)
    return int(p.returncode)


def _read_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f) or {}


def _assert(cond: bool, msg: str):
    if not cond:
        raise AssertionError(msg)


def main():
    ap = argparse.ArgumentParser(description="Ops acceptance validation (manual, non-CI).")
    ap.add_argument("--python", default=os.getenv("PYTHON", "python"))
    ap.add_argument("--no_telegram", action="store_true")
    ap.add_argument("--market_budget", type=int, default=4)
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--lookback_days", type=int, default=1)
    ap.add_argument("--min_k", type=int, default=1)
    ap.add_argument("--timeout_sec_refresh", type=int, default=120)
    ap.add_argument("--timeout_sec_infer", type=int, default=300)
    args = ap.parse_args()

    base_env = dict(os.environ)
    if args.no_telegram:
        base_env["AETHER_RUNTIME_NO_TELEGRAM"] = "1"

    # 0) Generate tuning file (best-effort)
    _run([args.python, "scripts/auto_tune_ops.py", "--window_runs", "8", "--write"], env=base_env)

    # 1) Intraday: allow stale by widening freshness gate to ensure we can run even offline.
    env1 = dict(base_env)
    env1["AETHER_FRESHNESS_MAX_LAG_HOURS_INTRADAY"] = "1000"
    cmd1 = [
        args.python,
        "scripts/run_scheduled.py",
        "--job",
        "intraday",
        "--limit",
        str(args.limit),
        "--lookback_days",
        str(args.lookback_days),
        "--min_k",
        str(args.min_k),
        "--market_budget",
        str(args.market_budget),
        "--refresh_top_n",
        "8",
        "--refresh_days",
        "1",
        "--timeout_sec_refresh",
        str(args.timeout_sec_refresh),
        "--timeout_sec_infer",
        str(args.timeout_sec_infer),
    ]
    if args.no_telegram:
        cmd1.append("--no_telegram")

    rc1 = _run(cmd1, env=env1)
    _assert(rc1 == 0, f"intraday pipeline failed rc={rc1}")

    m1 = _read_json(os.path.join("analysis", "run_markets_metrics_intraday.json"))
    _assert(m1.get("mode") == "intraday", "intraday metrics mode mismatch")
    _assert(int((m1.get("recs") or {}).get("n") or 0) >= 1, "intraday must produce >=1 rec")
    _assert((m1.get("meta") or {}).get("elapsed_sec") is not None, "intraday must record elapsed_sec")

    # 2) Morning: force freshness gate abort, ensure scheduler reruns once in watch-only safe-mode.
    env2 = dict(base_env)
    env2["AETHER_FRESHNESS_MAX_LAG_HOURS_MORNING"] = "0.1"
    cmd2 = [
        args.python,
        "scripts/run_scheduled.py",
        "--job",
        "morning-report",
        "--limit",
        str(args.limit),
        "--lookback_days",
        str(args.lookback_days),
        "--min_k",
        str(args.min_k),
        "--market_budget",
        str(args.market_budget),
        "--refresh_top_n",
        "8",
        "--refresh_days",
        "1",
        "--timeout_sec_refresh",
        str(args.timeout_sec_refresh),
        "--timeout_sec_infer",
        str(args.timeout_sec_infer),
        "--tune_from_logs",
        "--skip_aux",
    ]
    if args.no_telegram:
        cmd2.append("--no_telegram")

    rc2 = _run(cmd2, env=env2)
    _assert(rc2 == 0, f"morning pipeline failed rc={rc2}")

    m2 = _read_json(os.path.join("analysis", "run_markets_metrics_morning.json"))
    _assert(m2.get("mode") == "morning", "morning metrics mode mismatch")
    _assert(int((m2.get("recs") or {}).get("n") or 0) >= 1, "morning must produce >=1 rec")
    _assert(bool((m2.get("recs") or {}).get("has_watch")) is True, "morning stale fallback must be watch-only")
    _assert((m2.get("meta") or {}).get("elapsed_sec") is not None, "morning must record elapsed_sec")

    print("OPS_VALIDATE_OK")


if __name__ == "__main__":
    main()
