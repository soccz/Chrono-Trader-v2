import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


_AUDIT_PATH = os.path.join("analysis", "ops_control_audit.jsonl")


def append_ops_audit(event_type: str, payload: Optional[Dict[str, Any]] = None, path: str = _AUDIT_PATH) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": str(event_type or "").strip(),
        "payload": payload or {},
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def read_recent_ops_audit(limit: int = 10, path: str = _AUDIT_PATH) -> List[Dict[str, Any]]:
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
                    row = json.loads(line)
                except Exception:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    except Exception:
        return []
    return rows[-max(1, int(limit)) :][::-1]
