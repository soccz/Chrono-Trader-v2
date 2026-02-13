import socket


def can_resolve(host: str) -> bool:
    """
    Best-effort DNS resolution check to avoid spamming collector calls when network is down.
    """
    try:
        socket.getaddrinfo(host, 443)
        return True
    except Exception:
        return False

