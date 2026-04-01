#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import Dict, List

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.ops_health import mode_health, parse_ts, read_json
from utils.output_contract import read_output_manifest


def _assert(cond: bool, msg: str):
    if not cond:
        raise AssertionError(msg)


def _manifest_path(mode: str) -> str:
    return os.path.join("analysis", f"output_contract_{mode}.json")


def _normalize_legacy_refresh_manifest(mode: str, manifest: Dict[str, object], autofix_legacy: bool) -> Dict[str, object]:
    if mode != "refresh-db" or not autofix_legacy:
        return manifest
    if manifest.get("recommendation") is None and manifest.get("prediction") is None:
        return manifest

    manifest["recommendation"] = None
    manifest["prediction"] = None
    path = _manifest_path(mode)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


def _validate_manifest(mode: str, strict_missing: bool, autofix_legacy: bool) -> Dict[str, object]:
    manifest = read_output_manifest(mode)
    if not manifest:
        if strict_missing:
            raise FileNotFoundError(f"missing output manifest for mode={mode}")
        return {"mode": mode, "manifest": "missing"}
    manifest = _normalize_legacy_refresh_manifest(mode, manifest, autofix_legacy=autofix_legacy)

    _assert(str(manifest.get("mode")) == str(mode), f"{mode}: manifest mode mismatch")
    _assert(parse_ts(str(manifest.get("ts") or "")) is not None, f"{mode}: invalid manifest ts")

    rec = manifest.get("recommendation")
    pred = manifest.get("prediction")
    pump = manifest.get("pump_prediction")
    metrics = manifest.get("run_metrics")

    if rec:
        _assert(os.path.exists(str(rec.get("path") or "")), f"{mode}: missing recommendation file")
    if pred:
        _assert(os.path.exists(str(pred.get("path") or "")), f"{mode}: missing prediction file")
    if pump:
        _assert(os.path.exists(str(pump.get("path") or "")), f"{mode}: missing pump prediction file")
    if metrics:
        _assert(os.path.exists(str(metrics.get("path") or "")), f"{mode}: missing metrics file")
    if mode == "refresh-db":
        _assert(rec is None, "refresh-db: recommendation entry must be null")
        _assert(pred is None, "refresh-db: prediction entry must be null")

    return {
        "mode": mode,
        "manifest": "ok",
        "has_recommendation": bool(rec),
        "has_prediction": bool(pred),
        "has_pump": bool(pump),
        "has_metrics_ref": bool(metrics),
    }


def _validate_metrics(mode: str, max_age_h: float, strict_missing: bool) -> Dict[str, object]:
    path = os.path.join("analysis", f"run_markets_metrics_{mode}.json")
    obj = read_json(path)
    if not obj:
        if strict_missing:
            raise FileNotFoundError(f"missing metrics file for mode={mode}")
        return {"mode": mode, "metrics": "missing"}

    health = mode_health(mode, max_age_h=max_age_h)
    _assert(health.get("status") in {"ok", "stale", "empty_recs", "missing", "invalid_ts"}, f"{mode}: unknown health status")
    _assert(str(obj.get("mode") or mode) == mode, f"{mode}: metrics mode mismatch")
    _assert(parse_ts(str(obj.get("ts") or "")) is not None, f"{mode}: invalid metrics ts")

    recs = obj.get("recs") or {}
    meta = obj.get("meta") or {}
    _assert(isinstance(recs, dict), f"{mode}: recs payload must be object")
    _assert(isinstance(meta, dict), f"{mode}: meta payload must be object")

    if health.get("status") == "ok":
        _assert(int(recs.get("n") or 0) >= 1, f"{mode}: healthy metrics must have recs >= 1")

    return {
        "mode": mode,
        "metrics": "ok",
        "health_status": health.get("status"),
        "recs_n": int(recs.get("n") or 0),
    }


def main():
    ap = argparse.ArgumentParser(description="Validate ops output contracts and run metrics.")
    ap.add_argument("--modes", nargs="*", default=["intraday", "morning", "refresh-db"])
    ap.add_argument("--strict-missing", action="store_true", help="Fail if manifest/metrics files are missing.")
    ap.add_argument("--autofix-legacy", action="store_true", help="Normalize legacy refresh-db manifests in-place.")
    ap.add_argument("--max-age-intraday", type=float, default=5.0)
    ap.add_argument("--max-age-morning", type=float, default=30.0)
    ap.add_argument("--max-age-refresh-db", type=float, default=30.0)
    args = ap.parse_args()

    max_age_map = {
        "intraday": float(args.max_age_intraday),
        "morning": float(args.max_age_morning),
        "refresh-db": float(args.max_age_refresh_db),
    }

    summaries: List[Dict[str, object]] = []
    validated_any = False
    for mode in args.modes:
        manifest_info = _validate_manifest(mode, strict_missing=bool(args.strict_missing), autofix_legacy=bool(args.autofix_legacy))
        metrics_info = _validate_metrics(mode, max_age_h=max_age_map.get(mode, 30.0), strict_missing=bool(args.strict_missing))
        if manifest_info.get("manifest") != "missing" or metrics_info.get("metrics") != "missing":
            validated_any = True
        summaries.append({**manifest_info, **metrics_info})

    if not validated_any and not args.strict_missing:
        print("OPS_CONTRACTS_SKIPPED no manifests/metrics found")
        return

    for item in summaries:
        print(item)
    print("OPS_CONTRACTS_OK")


if __name__ == "__main__":
    main()
