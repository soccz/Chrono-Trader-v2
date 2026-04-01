import importlib.util
from pathlib import Path


def _load_system_resources_module():
    module_path = Path(__file__).resolve().parents[1] / "utils" / "system_resources.py"
    spec = importlib.util.spec_from_file_location("test_system_resources_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_memory_percent_uses_memavailable(tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "\n".join(
            [
                "MemTotal:       1000 kB",
                "MemAvailable:    250 kB",
            ]
        ),
        encoding="utf-8",
    )

    sr = _load_system_resources_module()
    value = sr.memory_percent(meminfo_path=str(meminfo))
    assert round(value, 1) == 75.0


def test_memory_percent_falls_back_to_free_buffers_cached(tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "\n".join(
            [
                "MemTotal:       1000 kB",
                "MemFree:         100 kB",
                "Buffers:          50 kB",
                "Cached:          150 kB",
            ]
        ),
        encoding="utf-8",
    )

    sr = _load_system_resources_module()
    value = sr.memory_percent(meminfo_path=str(meminfo))
    assert round(value, 1) == 70.0


def test_snapshot_contains_expected_keys(monkeypatch):
    sr = _load_system_resources_module()

    monkeypatch.setattr(sr, "cpu_percent", lambda: 12.3)
    monkeypatch.setattr(sr, "memory_percent", lambda meminfo_path="/proc/meminfo": 45.6)
    monkeypatch.setattr(sr, "disk_percent", lambda path=".": 78.9)

    snap = sr.snapshot(path=".")
    assert snap == {
        "cpu_percent": 12.3,
        "memory_percent": 45.6,
        "disk_percent": 78.9,
    }
