import importlib.util
from pathlib import Path


def _load_ops_audit_module():
    module_path = Path(__file__).resolve().parents[1] / "utils" / "ops_audit.py"
    spec = importlib.util.spec_from_file_location("test_ops_audit_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_append_and_read_recent_ops_audit(tmp_path):
    ops_audit = _load_ops_audit_module()
    audit_path = tmp_path / "analysis" / "ops_control_audit.jsonl"

    ops_audit.append_ops_audit("ops_preflight", {"job": "intraday"}, path=str(audit_path))
    ops_audit.append_ops_audit("ops_run_started", {"job": "morning-report"}, path=str(audit_path))

    rows = ops_audit.read_recent_ops_audit(limit=2, path=str(audit_path))
    assert len(rows) == 2
    assert rows[0]["event"] == "ops_run_started"
    assert rows[1]["payload"]["job"] == "intraday"
