import os
from typing import Dict, Optional


def cpu_percent() -> float:
    try:
        load1 = float(os.getloadavg()[0])
        cpu_count = max(1, int(os.cpu_count() or 1))
        return max(0.0, min(100.0, (load1 / cpu_count) * 100.0))
    except Exception:
        return 0.0


def memory_percent(meminfo_path: str = "/proc/meminfo") -> float:
    try:
        fields = {}
        with open(meminfo_path, "r", encoding="utf-8") as f:
            for line in f:
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                raw = value.strip().split()[0]
                fields[key.strip()] = float(raw)

        total = fields.get("MemTotal", 0.0)
        available = fields.get("MemAvailable")
        if available is None:
            free = fields.get("MemFree", 0.0)
            buffers = fields.get("Buffers", 0.0)
            cached = fields.get("Cached", 0.0)
            available = free + buffers + cached
        if total <= 0:
            return 0.0
        used = max(0.0, total - float(available))
        return max(0.0, min(100.0, (used / total) * 100.0))
    except Exception:
        return 0.0


def disk_percent(path: str = ".") -> float:
    try:
        stat = os.statvfs(path)
        total = float(stat.f_blocks) * float(stat.f_frsize)
        free = float(stat.f_bavail) * float(stat.f_frsize)
        if total <= 0:
            return 0.0
        used = max(0.0, total - free)
        return max(0.0, min(100.0, (used / total) * 100.0))
    except Exception:
        return 0.0


def snapshot(path: str = ".") -> Dict[str, float]:
    return {
        "cpu_percent": round(cpu_percent(), 1),
        "memory_percent": round(memory_percent(), 1),
        "disk_percent": round(disk_percent(path=path), 1),
    }
