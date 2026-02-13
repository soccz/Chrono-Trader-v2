import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = (line or "").strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def _parse_ts(s: str) -> Optional[datetime]:
    try:
        if not s:
            return None
        return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _mean(xs: List[float]) -> Optional[float]:
    xs = [float(x) for x in xs if x is not None]
    if not xs:
        return None
    return sum(xs) / len(xs)


def _summarize(mode: str, window_h: float) -> Dict[str, Any]:
    path = os.path.join("analysis", f"run_markets_metrics_{mode}.jsonl")
    rows = _read_jsonl(path)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=float(window_h))
    rows = [r for r in rows if (_parse_ts(str(r.get("ts", "") or "")) or datetime.min.replace(tzinfo=timezone.utc)) >= cutoff]
    rows.sort(key=lambda r: str(r.get("ts", "")))

    recs_n = [int(((r.get("recs") or {}).get("n") or 0)) for r in rows]
    has_watch = [bool(((r.get("recs") or {}).get("has_watch") or False)) for r in rows]
    has_forced = [bool(((r.get("recs") or {}).get("has_forced") or False)) for r in rows]
    elapsed = []
    mean_corr = []
    max_corr = []
    for r in rows:
        meta = r.get("meta") or {}
        if meta.get("elapsed_sec") is not None:
            try:
                elapsed.append(float(meta.get("elapsed_sec")))
            except Exception:
                pass
        cm = (r.get("corr_metrics") or {}).get("pairwise") or {}
        try:
            if cm.get("mean_pos_corr") is not None:
                mean_corr.append(float(cm.get("mean_pos_corr")))
        except Exception:
            pass
        try:
            if cm.get("max_pos_corr") is not None:
                max_corr.append(float(cm.get("max_pos_corr")))
        except Exception:
            pass

    last_ts = _parse_ts(str(rows[-1].get("ts", "") or "")) if rows else None
    last_age_h = None
    if last_ts is not None:
        last_age_h = (datetime.now(timezone.utc) - last_ts).total_seconds() / 3600.0

    n = len(rows)
    return {
        "mode": mode,
        "window_hours": float(window_h),
        "runs": n,
        "last_ts": last_ts.isoformat() if last_ts else None,
        "last_age_h": last_age_h,
        "recs_avg": _mean([float(x) for x in recs_n]) if recs_n else None,
        "recs_min": min(recs_n) if recs_n else None,
        "watch_rate": (sum(1 for x in has_watch if x) / n) if n else None,
        "forced_rate": (sum(1 for x in has_forced if x) / n) if n else None,
        "elapsed_avg_sec": _mean(elapsed),
        "corr_mean_pos_avg": _mean(mean_corr),
        "corr_max_pos_avg": _mean(max_corr),
    }


def main():
    ap = argparse.ArgumentParser(description="Soak report from analysis/run_markets_metrics_*.jsonl")
    ap.add_argument("--window_hours", type=float, default=48.0)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "window_hours": float(args.window_hours),
        "intraday": _summarize("intraday", args.window_hours),
        "morning": _summarize("morning", args.window_hours),
        "refresh_db": _summarize("refresh-db", args.window_hours),
    }

    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)

    if args.write:
        os.makedirs("analysis", exist_ok=True)
        with open(os.path.join("analysis", "soak_report_latest.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    raise SystemExit(0)


if __name__ == "__main__":
    main()

