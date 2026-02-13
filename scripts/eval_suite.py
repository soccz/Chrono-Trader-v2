import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, List


def _run(cmd: List[str]) -> int:
    p = subprocess.run(cmd)
    return int(p.returncode)


def _read_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f) or {}


def main():
    ap = argparse.ArgumentParser(description="Evaluation suite (minimal) - runs backtest and captures JSON summary.")
    ap.add_argument("--python", default=os.getenv("PYTHON", "python"))
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--stride_hours", type=int, default=4, help="Backtest stride in hours (4 = faster, 1 = full).")
    ap.add_argument("--tag", default="wf")
    ap.add_argument("--no_telegram", action="store_true")
    args = ap.parse_args()

    os.makedirs("analysis", exist_ok=True)
    env = dict(os.environ)
    env["AETHER_BACKTEST_STRIDE_HOURS"] = str(int(args.stride_hours))
    env["AETHER_BACKTEST_SUMMARY_TAG"] = str(args.tag)

    cmd = [args.python, "main.py", "--mode", "backtest", "--days", str(int(args.days))]
    if args.no_telegram:
        cmd.append("--no_telegram")

    ts = datetime.now(timezone.utc).isoformat()
    print(f"[{ts}] Running: {' '.join(cmd)}", flush=True)
    rc = subprocess.run(cmd, env=env).returncode

    summary = _read_json(os.path.join("analysis", "backtest_summary_latest.json"))
    out = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "rc": int(rc),
        "git_head": (subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout or "").strip() or None,
        "args": {
            "days": int(args.days),
            "stride_hours": int(args.stride_hours),
            "tag": str(args.tag),
        },
        "summary": summary,
    }

    fname = os.path.join("analysis", f"eval_suite_{args.tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Wrote {fname}", flush=True)

    raise SystemExit(int(rc))


if __name__ == "__main__":
    main()
