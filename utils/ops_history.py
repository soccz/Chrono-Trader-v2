import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional


def _parse_ts(value: str) -> Optional[datetime]:
    try:
        if not value:
            return None
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    rows: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    rows.append(obj)
    except Exception:
        return []
    return rows


def _normalize_run(row: Dict[str, Any]) -> Dict[str, Any]:
    meta = row.get("meta") or {}
    recs = row.get("recs") or {}
    markets = row.get("markets") or []
    return {
        "mode": str(row.get("mode", "") or ""),
        "ts": row.get("ts"),
        "markets_n": len(markets),
        "recs_n": int(recs.get("n") or 0),
        "has_watch": bool(recs.get("has_watch") or False),
        "has_forced": bool(recs.get("has_forced") or False),
        "elapsed_sec": meta.get("elapsed_sec"),
        "kept": int(meta.get("run_markets_kept") or 0),
        "dropped": int(meta.get("freshness_dropped") or 0),
        "used_fallback": bool(meta.get("freshness_used_fallback") or False),
    }


def read_recent_ops_runs(
    modes: Iterable[str] = ("intraday", "morning", "refresh-db"),
    limit_per_mode: int = 5,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for mode in modes:
        path = os.path.join("analysis", f"run_markets_metrics_{mode}.jsonl")
        rows = _read_jsonl(path)
        for row in rows[-max(1, int(limit_per_mode)):]:
            item = _normalize_run(row)
            item["_sort_ts"] = _parse_ts(str(item.get("ts") or ""))
            out.append(item)

    out.sort(key=lambda x: x.get("_sort_ts") or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    for item in out:
        item.pop("_sort_ts", None)
    return out
