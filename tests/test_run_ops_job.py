import importlib.util
from pathlib import Path


def _load_run_ops_job():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "run_ops_job.py"
    spec = importlib.util.spec_from_file_location("test_run_ops_job_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_resolve_repo_python_prefers_env_override(monkeypatch, tmp_path):
    run_ops_job = _load_run_ops_job()

    custom_python = tmp_path / "custom-python"
    custom_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    custom_python.chmod(0o755)

    monkeypatch.setattr(run_ops_job.shutil, "which", lambda name: None)
    resolved = run_ops_job.resolve_repo_python(
        repo_root=tmp_path,
        env={"AETHER_PYTHON_BIN": str(custom_python)},
    )
    assert resolved == str(custom_python)


def test_resolve_repo_python_prefers_venv_local(monkeypatch, tmp_path):
    run_ops_job = _load_run_ops_job()

    local_python = tmp_path / ".venv_local" / "bin" / "python"
    local_python.parent.mkdir(parents=True)
    local_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    local_python.chmod(0o755)

    monkeypatch.setattr(run_ops_job.shutil, "which", lambda name: None)
    resolved = run_ops_job.resolve_repo_python(repo_root=tmp_path, env={})
    assert resolved == str(local_python)


def test_build_ops_command_strips_forwarded_python(monkeypatch, tmp_path):
    run_ops_job = _load_run_ops_job()

    resolved_python = tmp_path / ".venv_local" / "bin" / "python"
    resolved_python.parent.mkdir(parents=True)
    resolved_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    resolved_python.chmod(0o755)

    monkeypatch.setattr(run_ops_job.shutil, "which", lambda name: None)
    command = run_ops_job.build_ops_command(
        [
            "--job",
            "intraday",
            "--python",
            "/usr/bin/python3",
            "--limit",
            "8",
        ],
        repo_root=tmp_path,
        env={},
    )

    assert command[0] == str(resolved_python)
    assert command[1].endswith("scripts/run_scheduled.py")
    assert command[2:4] == ["--python", str(resolved_python)]
    assert "--job" in command
    assert "intraday" in command
    assert "/usr/bin/python3" not in command
