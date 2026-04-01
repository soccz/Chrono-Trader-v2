import glob
import json
import os
import re
from typing import Any, Dict, List, Optional

from utils.config import config


def current_runtime_model_signature() -> Dict[str, Any]:
    feature_columns = list(getattr(config.Data, "FEATURE_COLUMNS", []) or [])
    return {
        "input_dim": len(feature_columns),
        "output_dim": int(getattr(config.Data, "FUTURE_WINDOW_SIZE", 0) or 0),
        "future_window_size": int(getattr(config.Data, "FUTURE_WINDOW_SIZE", 0) or 0),
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "context_dim": int(getattr(config.Data, "CONTEXT_DIM", 0) or 0),
        "decoder_mode": str(getattr(config.Gan, "DECODER_MODE", "") or ""),
        "pe_mode": str(getattr(config.Gan, "PE_MODE", "") or ""),
        "strip_context_from_main_input": bool(
            getattr(config.Gan, "STRIP_CONTEXT_FROM_MAIN_INPUT", False)
        ),
    }


def read_model_metadata(repo_root: str = ".") -> Dict[str, Any]:
    path = os.path.join(repo_root, "models", "model_metadata.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def list_numbered_model_paths(repo_root: str = ".") -> List[str]:
    pattern = os.path.join(repo_root, "models", "model_*.pth")
    numbered = []
    for path in sorted(glob.glob(pattern)):
        base = os.path.basename(path)
        if re.fullmatch(r"model_\d+\.pth", base):
            numbered.append(path if repo_root != "." else os.path.relpath(path, repo_root))
    return numbered


def model_metadata_entry_issues(
    entry: Optional[Dict[str, Any]],
    signature: Optional[Dict[str, Any]] = None,
) -> List[str]:
    if not entry:
        return ["metadata missing"]

    signature = signature or current_runtime_model_signature()
    issues: List[str] = []

    required_fields = (
        "input_dim",
        "output_dim",
        "decoder_mode",
        "pe_mode",
        "context_dim",
        "feature_columns",
    )
    for field in required_fields:
        if field not in entry:
            issues.append(f"{field} missing")

    comparisons = (
        ("input_dim", signature["input_dim"]),
        ("output_dim", signature["output_dim"]),
        ("future_window_size", signature["future_window_size"]),
        ("decoder_mode", signature["decoder_mode"]),
        ("pe_mode", signature["pe_mode"]),
        ("context_dim", signature["context_dim"]),
        (
            "strip_context_from_main_input",
            signature["strip_context_from_main_input"],
        ),
    )
    for field, expected in comparisons:
        actual = entry.get(field)
        if actual is None:
            continue
        if actual != expected:
            issues.append(f"{field}={actual} (expected {expected})")

    actual_features = entry.get("feature_columns")
    if isinstance(actual_features, list) and actual_features != signature["feature_columns"]:
        issues.append("feature_columns mismatch")

    return issues


def build_model_artifact_audit(repo_root: str = ".") -> Dict[str, Any]:
    signature = current_runtime_model_signature()
    metadata = read_model_metadata(repo_root=repo_root)
    model_entries = (metadata.get("models") or {}) if isinstance(metadata, dict) else {}
    model_paths = list_numbered_model_paths(repo_root=repo_root)

    models: List[Dict[str, Any]] = []
    compatible_count = 0
    for path in model_paths:
        model_key = os.path.splitext(os.path.basename(path))[0]
        entry = model_entries.get(model_key)
        issues = model_metadata_entry_issues(entry, signature=signature)
        runtime_compatible = len(issues) == 0
        if runtime_compatible:
            compatible_count += 1
        models.append(
            {
                "path": path,
                "model_key": model_key,
                "metadata_present": bool(entry),
                "runtime_compatible": runtime_compatible,
                "issues": issues,
                "created_at": entry.get("created_at") if isinstance(entry, dict) else None,
                "val_loss": entry.get("val_loss") if isinstance(entry, dict) else None,
                "epochs": entry.get("epochs") if isinstance(entry, dict) else None,
            }
        )

    return {
        "signature": signature,
        "metadata_version": metadata.get("version") if isinstance(metadata, dict) else None,
        "models": models,
        "compatible_count": compatible_count,
        "total_count": len(models),
    }
