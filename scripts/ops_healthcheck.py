import os
import sys

# Ensure repo root is importable when invoked as `python scripts/...`.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import argparse
import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from utils.telegram_bot import send_alert


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return None


def _parse_ts(s: str) -> Optional[datetime]:
    try:
        if not s:
            return None
        # record_run uses ISO with timezone; keep robust.
        return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _age_hours(ts: Optional[datetime]) -> Optional[float]:
    if ts is None:
        return None
    now = datetime.now(timezone.utc)
    return (now - ts).total_seconds() / 3600.0


def _mode_health(mode: str, max_age_h: float) -> Tuple[bool, str]:
    p = os.path.join("analysis", f"run_markets_metrics_{mode}.json")
    obj = _read_json(p)
    if not obj:
        return False, f"{mode}: metrics missing ({p})"
    ts = _parse_ts(str(obj.get("ts", "") or ""))
    age = _age_hours(ts)
    if age is None:
        return False, f"{mode}: invalid ts in metrics"
    recs_n = int(((obj.get("recs") or {}).get("n") or 0))
    has_watch = bool(((obj.get("recs") or {}).get("has_watch") or False))
    if age > float(max_age_h):
        return False, f"{mode}: stale age={age:.2f}h > {float(max_age_h):.2f}h (recs={recs_n}, watch={has_watch})"
    if recs_n < 1:
        return False, f"{mode}: recs=0 (age={age:.2f}h)"
    return True, f"{mode}: ok age={age:.2f}h recs={recs_n} watch={has_watch}"


def main():
    ap = argparse.ArgumentParser(description="Ops healthcheck: verifies recent successful runs and outputs.")
    ap.add_argument("--max_age_intraday_h", type=float, default=5.0)
    ap.add_argument("--max_age_morning_h", type=float, default=30.0)
    ap.add_argument("--send_telegram", action="store_true")
    args = ap.parse_args()

    ok_i, msg_i = _mode_health("intraday", args.max_age_intraday_h)
    ok_m, msg_m = _mode_health("morning", args.max_age_morning_h)

    overall_ok = ok_i and ok_m
    status = "OK" if overall_ok else "ALERT"
    msg = f"[AETHER Healthcheck] {status}\n- {msg_i}\n- {msg_m}\n(ts={datetime.now(timezone.utc).isoformat()})"
    print(msg, flush=True)

    if (not overall_ok) and args.send_telegram:
        # bypass dedup so repeated failures still alert (but send_alert itself has dedup; keep it simple).
        send_alert(msg, bypass_dedup=True)

    raise SystemExit(0 if overall_ok else 2)


if __name__ == "__main__":
    main()
