import signal
from contextlib import contextmanager

from utils.logger import logger


@contextmanager
def time_limit(seconds: int, name: str = "run"):
    """
    Best-effort watchdog using SIGALRM (Unix).
    If not supported, it becomes a no-op.
    """
    try:
        seconds = int(seconds)
    except Exception:
        seconds = 0

    if seconds <= 0:
        yield
        return

    if not hasattr(signal, "SIGALRM"):
        logger.warning(f"[Watchdog] SIGALRM not available; cannot enforce timeout for {name}.")
        yield
        return

    def _handler(signum, frame):
        raise TimeoutError(f"Watchdog timeout after {seconds}s: {name}")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    try:
        signal.alarm(seconds)
        yield
    finally:
        try:
            signal.alarm(0)
        except Exception:
            pass
        try:
            signal.signal(signal.SIGALRM, old_handler)
        except Exception:
            pass

