import socket
import time
from typing import Any, Dict


def resolution_status(host: str, port: int = 443, attempts: int = 1, delay_sec: float = 0.0) -> Dict[str, Any]:
    """
    Best-effort DNS resolution check with small retry budget for transient resolver hiccups.

    Returns:
    - ok: bool
    - attempts: int
    - error: str | None
    - ips: list[str]
    """
    attempts = max(1, int(attempts or 1))
    delay_sec = max(0.0, float(delay_sec or 0.0))
    last_error = None
    ips = []

    for idx in range(attempts):
        try:
            infos = socket.getaddrinfo(host, port)
            ips = sorted({str(item[4][0]) for item in infos if item and len(item) >= 5 and item[4]})
            return {
                "ok": True,
                "attempts": idx + 1,
                "error": None,
                "ips": ips,
            }
        except Exception as e:
            last_error = str(e)
            if idx + 1 < attempts and delay_sec > 0:
                time.sleep(delay_sec)

    return {
        "ok": False,
        "attempts": attempts,
        "error": last_error,
        "ips": ips,
    }


def can_resolve(host: str, attempts: int = 1, delay_sec: float = 0.0) -> bool:
    """
    Compatibility wrapper for callers that only need a boolean.
    """
    return bool(resolution_status(host, attempts=attempts, delay_sec=delay_sec).get("ok"))
