import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, List


def _run(cmd: List[str], env: Dict[str, str]) -> int:
    p = subprocess.run(cmd, env=env)
    return int(p.returncode)


def _read_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f) or {}


def main():
    ap = argparse.ArgumentParser(description="Ablation suite (v1): baseline vs no-factors (optional no-context).")
    ap.add_argument("--python", default=os.getenv("PYTHON", "python"))
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--stride_hours", type=int, default=4)
    ap.add_argument("--tag", default="abl")
    ap.add_argument("--no_telegram", action="store_true")
    ap.add_argument("--include_no_context", action="store_true")
    args = ap.parse_args()

    base_cmd = [args.python, "main.py", "--mode", "backtest", "--days", str(int(args.days))]
    if args.no_telegram:
        base_cmd.append("--no_telegram")

    # Pin a single end_time across all cases so ablation comparisons are apples-to-apples.
    # (Also helps when DB is stale/offline: evaluator will clamp to DB latest if needed.)
    pinned_end_time = datetime.now(timezone.utc).isoformat()

    runs = []
    cases = [
        ("baseline", {}),
        ("no_factors", {"AETHER_ABLATE_FACTORS": "1"}),
    ]
    if args.include_no_context:
        cases.append(("no_context", {"AETHER_ABLATE_CONTEXT": "1"}))
        cases.append(("no_factors_no_context", {"AETHER_ABLATE_FACTORS": "1", "AETHER_ABLATE_CONTEXT": "1"}))

    for name, overrides in cases:
        env = dict(os.environ)
        env["AETHER_BACKTEST_STRIDE_HOURS"] = str(int(args.stride_hours))
        env["AETHER_BACKTEST_SUMMARY_TAG"] = f"{args.tag}_{name}"
        env["AETHER_BACKTEST_END_TIME_ISO"] = pinned_end_time
        for k, v in overrides.items():
            env[k] = str(v)

        ts = datetime.now(timezone.utc).isoformat()
        print(f"[{ts}] Case={name} overrides={overrides}", flush=True)
        rc = _run(base_cmd, env=env)
        summary = _read_json(os.path.join("analysis", "backtest_summary_latest.json"))
        runs.append({"case": name, "rc": rc, "overrides": overrides, "summary": summary})

    out = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "days": int(args.days),
        "stride_hours": int(args.stride_hours),
        "tag": str(args.tag),
        "runs": runs,
    }
    os.makedirs("analysis", exist_ok=True)
    path = os.path.join("analysis", f"ablation_suite_{args.tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Wrote {path}", flush=True)

    # Non-zero if any case failed.
    raise SystemExit(0 if all(r["rc"] == 0 for r in runs) else 2)


if __name__ == "__main__":
    main()
