import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.netcheck import can_resolve, resolution_status


def test_resolution_status_retries_until_success(monkeypatch):
    calls = {"n": 0}

    def fake_getaddrinfo(host, port):
        calls["n"] += 1
        if calls["n"] < 3:
            raise socket.gaierror("temporary failure")
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("1.2.3.4", port)),
        ]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    status = resolution_status("api.upbit.com", attempts=3, delay_sec=0.0)

    assert status["ok"] is True
    assert status["attempts"] == 3
    assert status["error"] is None
    assert status["ips"] == ["1.2.3.4"]


def test_can_resolve_returns_false_after_retry_budget(monkeypatch):
    calls = {"n": 0}

    def fake_getaddrinfo(host, port):
        calls["n"] += 1
        raise socket.gaierror("name or service not known")

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    assert can_resolve("api.upbit.com", attempts=2, delay_sec=0.0) is False
    assert calls["n"] == 2
