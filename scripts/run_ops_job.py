import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional


_REPO_ROOT = Path(__file__).resolve().parents[1]
_RUN_SCHEDULED = _REPO_ROOT / "scripts" / "run_scheduled.py"


def _is_executable_file(path: Optional[str]) -> bool:
    if not path:
        return False
    try:
        return Path(path).is_file() and os.access(path, os.X_OK)
    except Exception:
        return False


def resolve_repo_python(repo_root: Path = _REPO_ROOT, env: Optional[dict] = None) -> str:
    repo_root = Path(repo_root) if not isinstance(repo_root, Path) else repo_root
    env = env or os.environ
    candidates = [
        env.get("AETHER_PYTHON_BIN"),
        str(repo_root / ".venv_local" / "bin" / "python"),
        shutil.which("python3"),
        shutil.which("python"),
    ]
    for candidate in candidates:
        if _is_executable_file(candidate):
            return str(candidate)
    raise FileNotFoundError("No usable Python runtime found for ops job.")


def sanitize_forward_args(argv: Iterable[str]) -> List[str]:
    args = list(argv)
    out: List[str] = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--python":
            i += 2
            continue
        out.append(arg)
        i += 1
    return out


def build_ops_command(argv: Iterable[str], repo_root: Path = _REPO_ROOT, env: Optional[dict] = None) -> List[str]:
    python_bin = resolve_repo_python(repo_root=repo_root, env=env)
    return [python_bin, str(_RUN_SCHEDULED), "--python", python_bin] + sanitize_forward_args(argv)


def main() -> None:
    cmd = build_ops_command(sys.argv[1:])
    raise SystemExit(subprocess.run(cmd, cwd=str(_REPO_ROOT)).returncode)


if __name__ == "__main__":
    main()
