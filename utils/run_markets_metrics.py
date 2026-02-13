import json
import os
from datetime import datetime, timezone
from typing import List, Optional, Dict

import numpy as np
import pandas as pd

from utils.logger import logger
from data.database import load_data, get_latest_db_timestamp


def _dedupe(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items or []:
        if not x:
            continue
        if x in seen:
            continue
        seen.add(x)
        out.append(str(x))
    return out


def _bucket_corr_btc(v: float) -> str:
    try:
        if not np.isfinite(v):
            return "mid"
        if v < 0.2:
            return "low"
        if v <= 0.6:
            return "mid"
        return "high"
    except Exception:
        return "mid"


def compute_corr_metrics(markets: List[str], lookback_hours: int = 168) -> Dict:
    markets = _dedupe(markets)
    if not markets:
        return {}

    end_time = get_latest_db_timestamp(markets=markets) or get_latest_db_timestamp()
    if end_time is None:
        return {}

    start_time = end_time - pd.Timedelta(hours=int(lookback_hours))
    placeholders = ", ".join("?" for _ in markets)
    q = (
        "SELECT timestamp, market, close "
        f"FROM crypto_data WHERE market IN ({placeholders}) AND timestamp BETWEEN ? AND ?"
    )
    params = list(markets) + [
        start_time.strftime("%Y-%m-%dT%H:%M:%S"),
        end_time.strftime("%Y-%m-%dT%H:%M:%S"),
    ]
    df = load_data(q, params=params)
    if df.empty:
        return {}

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "market", "close"])
    if df.empty:
        return {}

    pivot = df.pivot(index="timestamp", columns="market", values="close").sort_index()
    rets = pivot.pct_change(fill_method=None)
    if rets.empty:
        return {}

    # Pairwise correlations (use positive corr for "theme duplication" signal)
    corr = rets.corr(min_periods=24)
    if corr.empty:
        return {}

    cols = [c for c in markets if c in corr.columns]
    corr = corr.loc[cols, cols]
    n = len(cols)
    if n < 2:
        return {"n": n}

    iu = np.triu_indices(n, k=1)
    vals = corr.values[iu]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        pair = {}
    else:
        pos = vals[vals > 0]
        pair = {
            "mean_corr": float(np.mean(vals)),
            "mean_pos_corr": float(np.mean(pos)) if pos.size else float(np.nan),
            "max_corr": float(np.max(vals)),
            "max_pos_corr": float(np.max(pos)) if pos.size else float(np.nan),
        }

    # Corr to BTC for bucket distribution if available
    corr_btc = {}
    if "KRW-BTC" in rets.columns:
        btc = rets["KRW-BTC"]
        c = rets[cols].corrwith(btc)
        buckets = {"low": 0, "mid": 0, "high": 0}
        for m, v in c.items():
            b = _bucket_corr_btc(float(v) if pd.notna(v) else np.nan)
            buckets[b] += 1
        corr_btc = {"buckets": buckets}

    return {
        "n": n,
        "pairwise": pair,
        "corr_btc": corr_btc,
        "end_time": end_time.isoformat(),
    }


def record_run(mode: str, markets: List[str], meta: Optional[dict] = None, recs: Optional[list] = None) -> Dict:
    os.makedirs("analysis", exist_ok=True)
    mode = str(mode)
    meta = meta or {}
    markets = _dedupe(markets)

    statuses = []
    if recs:
        for r in recs:
            statuses.append(str(r.get("status", "") or ""))

    payload = {
        "mode": mode,
        "ts": datetime.now(timezone.utc).isoformat(),
        "markets": markets,
        "meta": meta,
        "recs": {
            "n": len(recs or []),
            "has_watch": any(s.startswith("Watch") for s in statuses),
            "has_forced": any(s.startswith("Forced") for s in statuses),
        },
        "corr_metrics": compute_corr_metrics(markets),
    }

    latest_path = os.path.join("analysis", f"run_markets_metrics_{mode}.json")
    hist_path = os.path.join("analysis", f"run_markets_metrics_{mode}.jsonl")
    try:
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.debug(f"[Metrics] Failed to write {latest_path}: {e}")

    try:
        with open(hist_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.debug(f"[Metrics] Failed to append {hist_path}: {e}")

    return payload

