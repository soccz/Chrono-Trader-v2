import io
import importlib.util
from pathlib import Path


class _DummyProcess:
    def __init__(self, returncode=0):
        self.stdout = io.StringIO("")
        self._returncode = returncode

    def poll(self):
        return None

    def wait(self):
        return self._returncode


class _DummyThread:
    def __init__(self, target=None, daemon=None):
        self.target = target
        self.daemon = daemon

    def start(self):
        return None


def _load_task_runner_module():
    module_path = Path(__file__).resolve().parents[1] / "web_utils" / "task_runner.py"
    spec = importlib.util.spec_from_file_location("test_task_runner_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_task_runner_status_includes_task_key_and_command(monkeypatch):
    task_runner_module = _load_task_runner_module()
    TaskRunner = task_runner_module.TaskRunner

    monkeypatch.setattr(task_runner_module.subprocess, "Popen", lambda *args, **kwargs: _DummyProcess(returncode=0))
    monkeypatch.setattr(task_runner_module.threading, "Thread", _DummyThread)

    runner = TaskRunner()
    result = runner.start_task(["python", "main.py", "--mode", "train"], "Training", task_key="train")
    status = runner.get_status()

    assert result["success"] is True
    assert result["task_key"] == "train"
    assert status["task_name"] == "Training"
    assert status["task_key"] == "train"
    assert status["command"][-1] == "train"


def test_task_runner_monitor_sets_returncode(monkeypatch):
    task_runner_module = _load_task_runner_module()
    TaskRunner = task_runner_module.TaskRunner

    runner = TaskRunner()
    runner.current_process = _DummyProcess(returncode=3)
    runner._monitor_process()
    status = runner.get_status()

    assert status["status"] == "failed"
    assert status["returncode"] == 3
