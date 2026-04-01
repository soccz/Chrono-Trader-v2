import os
import sys

# Ensure repo root is importable when invoked as `python scripts/...`.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import argparse
from datetime import datetime, timezone

from utils.telegram_bot import send_alert
from utils.ops_health import mode_health


def main():
    ap = argparse.ArgumentParser(description="Ops healthcheck: verifies recent successful runs and outputs.")
    ap.add_argument("--max_age_intraday_h", type=float, default=5.0)
    ap.add_argument("--max_age_morning_h", type=float, default=30.0)
    ap.add_argument("--send_telegram", action="store_true")
    args = ap.parse_args()

    intraday = mode_health("intraday", args.max_age_intraday_h)
    morning = mode_health("morning", args.max_age_morning_h)

    overall_ok = bool(intraday.get("ok")) and bool(morning.get("ok"))
    status = "OK" if overall_ok else "ALERT"
    msg = (
        f"[AETHER Healthcheck] {status}\n"
        f"- {intraday.get('message')}\n"
        f"- {morning.get('message')}\n"
        f"(ts={datetime.now(timezone.utc).isoformat()})"
    )
    print(msg, flush=True)

    if (not overall_ok) and args.send_telegram:
        # bypass dedup so repeated failures still alert (but send_alert itself has dedup; keep it simple).
        send_alert(msg, bypass_dedup=True)

    raise SystemExit(0 if overall_ok else 2)


if __name__ == "__main__":
    main()
