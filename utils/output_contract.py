import glob
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional


def _file_info(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return None
    try:
        stat = os.stat(path)
        return {
            "path": path,
            "basename": os.path.basename(path),
            "size_bytes": int(stat.st_size),
            "mtime": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        }
    except Exception:
        return None


def latest_matching_file(pattern: str) -> Optional[str]:
    matches = glob.glob(pattern)
    if not matches:
        return None
    try:
        return max(matches, key=os.path.getmtime)
    except Exception:
        return None


def latest_prediction_file() -> Optional[str]:
    matches = [
        path for path in glob.glob(os.path.join("predictions", "*.csv"))
        if not os.path.basename(path).startswith("pump_preds_")
    ]
    if not matches:
        return None
    try:
        return max(matches, key=os.path.getmtime)
    except Exception:
        return None


def build_output_manifest(
    mode: str,
    rec_tag: str,
    run_metrics_path: Optional[str] = None,
    include_pump_preds: bool = False,
    include_recommendation: bool = True,
    include_prediction: bool = True,
) -> Dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    rec_pattern = os.path.join("recommendations", f"recs_{rec_tag}_*.csv")
    pump_pattern = os.path.join("predictions", "pump_preds_*.csv")

    recommendation_file = latest_matching_file(rec_pattern) if include_recommendation else None
    prediction_file = latest_prediction_file() if include_prediction else None
    pump_file = latest_matching_file(pump_pattern) if include_pump_preds else None

    return {
        "mode": str(mode),
        "ts": now,
        "recommendation": _file_info(recommendation_file) if include_recommendation else None,
        "prediction": _file_info(prediction_file) if include_prediction else None,
        "pump_prediction": _file_info(pump_file) if include_pump_preds else None,
        "run_metrics": _file_info(run_metrics_path) if run_metrics_path else None,
    }


def write_output_manifest(mode: str, manifest: Dict[str, Any]) -> str:
    os.makedirs("analysis", exist_ok=True)
    path = os.path.join("analysis", f"output_contract_{mode}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return path


def read_output_manifest(mode: str) -> Optional[Dict[str, Any]]:
    path = os.path.join("analysis", f"output_contract_{mode}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def read_output_manifests(modes: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    manifests: Dict[str, Dict[str, Any]] = {}
    for mode in modes:
        manifest = read_output_manifest(mode)
        if manifest:
            manifests[str(mode)] = manifest
    return manifests


def backfill_output_manifest(mode: str, overwrite: bool = False) -> Optional[str]:
    mode = str(mode)
    existing = read_output_manifest(mode)
    if existing and not overwrite:
        return os.path.join("analysis", f"output_contract_{mode}.json")

    rec_tag_map = {
        "intraday": "intraday",
        "morning": "morning",
        "refresh-db": "refresh-db",
    }
    include_pump_map = {
        "intraday": False,
        "morning": True,
        "refresh-db": False,
    }
    include_recommendation = mode != "refresh-db"
    include_prediction = mode != "refresh-db"
    metrics_path = os.path.join("analysis", f"run_markets_metrics_{mode}.json")
    if not os.path.exists(metrics_path) and not include_recommendation:
        return None

    manifest = build_output_manifest(
        mode=mode,
        rec_tag=rec_tag_map.get(mode, mode),
        run_metrics_path=metrics_path if os.path.exists(metrics_path) else None,
        include_pump_preds=include_pump_map.get(mode, False),
        include_recommendation=include_recommendation,
        include_prediction=include_prediction,
    )
    return write_output_manifest(mode, manifest)


def backfill_output_manifests(modes: Iterable[str], overwrite: bool = False) -> Dict[str, str]:
    written: Dict[str, str] = {}
    for mode in modes:
        path = backfill_output_manifest(str(mode), overwrite=overwrite)
        if path:
            written[str(mode)] = path
    return written
