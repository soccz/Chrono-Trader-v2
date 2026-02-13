import sys
import time
import subprocess

import pytest


def test_run_lock_blocks_other_process(tmp_path):
    # Use a temp lock dir to avoid interfering with real cron locks.
    lock_dir = tmp_path / "locks"
    lock_name = "pytest-lock"
    lock_path = lock_dir / f"{lock_name}.lock"

    # Hold the lock in a child process.
    code = (
        "import time\n"
        "from utils.run_lock import run_lock\n"
        f"with run_lock('{lock_name}', lock_dir=r'{str(lock_dir)}'):\n"
        "    time.sleep(2.0)\n"
    )
    p = subprocess.Popen(
        [sys.executable, "-c", code],
        cwd=".",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        # Wait until the child actually creates the lockfile (acquired).
        t0 = time.time()
        while time.time() - t0 < 10.0:
            if p.poll() is not None:
                out, err = p.communicate(timeout=1)
                raise AssertionError(f"child exited early (rc={p.returncode})\nstdout:\n{out}\nstderr:\n{err}")
            if lock_path.exists():
                break
            time.sleep(0.05)
        if not lock_path.exists():
            out, err = p.communicate(timeout=1)
            raise AssertionError(f"child did not acquire lock in time\nstdout:\n{out}\nstderr:\n{err}")

        from utils.run_lock import run_lock

        with pytest.raises(SystemExit) as e:
            with run_lock(lock_name, lock_dir=str(lock_dir), timeout_sec=0.0, exit_code=0):
                pass
        assert e.value.code == 0
    finally:
        p.terminate()
        try:
            p.wait(timeout=5)
        except Exception:
            p.kill()
