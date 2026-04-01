import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return None


def parse_ts(s: str) -> Optional[datetime]:
    try:
        if not s:
            return None
        return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def age_hours(ts: Optional[datetime], now_utc: Optional[datetime] = None) -> Optional[float]:
    if ts is None:
        return None
    now_utc = now_utc or datetime.now(timezone.utc)
    return (now_utc - ts).total_seconds() / 3600.0


def mode_health(mode: str, max_age_h: float, now_utc: Optional[datetime] = None) -> Dict[str, Any]:
    now_utc = now_utc or datetime.now(timezone.utc)
    path = os.path.join("analysis", f"run_markets_metrics_{mode}.json")
    obj = read_json(path)
    if not obj:
        return {
            "ok": False,
            "mode": mode,
            "status": "missing",
            "message": f"{mode}: metrics missing ({path})",
            "metrics_path": path,
        }

    ts = parse_ts(str(obj.get("ts", "") or ""))
    age_h = age_hours(ts, now_utc=now_utc)
    if age_h is None:
        return {
            "ok": False,
            "mode": mode,
            "status": "invalid_ts",
            "message": f"{mode}: invalid ts in metrics",
            "metrics_path": path,
        }

    recs = obj.get("recs") or {}
    recs_n = int(recs.get("n") or 0)
    has_watch = bool(recs.get("has_watch") or False)
    meta = obj.get("meta") or {}
    kept = int(meta.get("run_markets_kept") or 0)
    dropped = int(meta.get("freshness_dropped") or 0)
    used_fallback = bool(meta.get("freshness_used_fallback") or False)
    auto_refresh_skipped_offline = bool(meta.get("auto_refresh_skipped_offline") or False)
    auto_refresh_dns_error = str(meta.get("auto_refresh_dns_error") or "").strip()
    network_status = str(meta.get("network_status") or "").strip().lower()
    network_note = ""
    if auto_refresh_skipped_offline:
        network_note = " offline_auto_refresh"
        if auto_refresh_dns_error:
            network_note += f" dns={auto_refresh_dns_error}"

    if mode == "refresh-db" and network_status == "offline":
        return {
            "ok": False,
            "mode": mode,
            "status": "offline",
            "age_hours": age_h,
            "recs_n": recs_n,
            "has_watch": has_watch,
            "kept": kept,
            "dropped": dropped,
            "used_fallback": used_fallback,
            "auto_refresh_skipped_offline": auto_refresh_skipped_offline,
            "auto_refresh_dns_error": auto_refresh_dns_error,
            "metrics_path": path,
            "message": f"{mode}: offline age={age_h:.2f}h{network_note}",
        }

    if age_h > float(max_age_h):
        return {
            "ok": False,
            "mode": mode,
            "status": "stale",
            "age_hours": age_h,
            "recs_n": recs_n,
            "has_watch": has_watch,
            "kept": kept,
            "dropped": dropped,
            "used_fallback": used_fallback,
            "auto_refresh_skipped_offline": auto_refresh_skipped_offline,
            "auto_refresh_dns_error": auto_refresh_dns_error,
            "metrics_path": path,
            "message": f"{mode}: stale age={age_h:.2f}h > {float(max_age_h):.2f}h (recs={recs_n}, watch={has_watch}){network_note}",
        }

    if recs_n < 1:
        return {
            "ok": False,
            "mode": mode,
            "status": "empty_recs",
            "age_hours": age_h,
            "recs_n": recs_n,
            "has_watch": has_watch,
            "kept": kept,
            "dropped": dropped,
            "used_fallback": used_fallback,
            "auto_refresh_skipped_offline": auto_refresh_skipped_offline,
            "auto_refresh_dns_error": auto_refresh_dns_error,
            "metrics_path": path,
            "message": f"{mode}: recs=0 (age={age_h:.2f}h){network_note}",
        }

    return {
        "ok": True,
        "mode": mode,
        "status": "ok",
        "age_hours": age_h,
        "recs_n": recs_n,
        "has_watch": has_watch,
        "kept": kept,
        "dropped": dropped,
        "used_fallback": used_fallback,
        "auto_refresh_skipped_offline": auto_refresh_skipped_offline,
        "auto_refresh_dns_error": auto_refresh_dns_error,
        "metrics_path": path,
        "message": f"{mode}: ok age={age_h:.2f}h recs={recs_n} watch={has_watch}{network_note}",
    }
