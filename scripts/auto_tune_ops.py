import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


def _read_jsonl(path: str) -> List[dict]:
    out = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a or []), set(b or [])
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return float(len(sa & sb) / max(1, len(sa | sb)))


def _clip(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def _mean(xs: List[float]) -> Optional[float]:
    xs = [float(x) for x in xs if x is not None]
    return float(sum(xs) / len(xs)) if xs else None


def _get_corr_stat(r: dict, key: str) -> Optional[float]:
    try:
        return float((((r or {}).get("corr_metrics") or {}).get("pairwise") or {}).get(key))
    except Exception:
        return None


def _get_meta(r: dict, key: str) -> Any:
    try:
        return ((r or {}).get("meta") or {}).get(key)
    except Exception:
        return None


def tune_from_runs(runs: List[dict], mode_name: str) -> Dict[str, Any]:
    """
    Diversity-first, compute-safe tuning.
    Produces suggested:
      - selection corr_max (AETHER_SELECTION_CORR_MAX)
      - bucket quotas env (AETHER_SELECTION_BUCKET_QUOTAS_*)
      - rotation_keep (cron arg)
    """
    runs = [r for r in runs if isinstance(r, dict)]
    if not runs:
        return {"mode": mode_name, "error": "no_runs"}

    # Keep only runs with markets list
    runs = [r for r in runs if isinstance(r.get("markets"), list) and r.get("markets")]
    if not runs:
        return {"mode": mode_name, "error": "no_markets"}

    mean_pos = [_get_corr_stat(r, "mean_pos_corr") for r in runs]
    mean_pos = [x for x in mean_pos if x is not None and x == x]
    avg_mean_pos = _mean(mean_pos)

    # Consecutive repetition (stability vs diversity)
    jacc = []
    for i in range(1, len(runs)):
        jacc.append(_jaccard(runs[i - 1].get("markets"), runs[i].get("markets")))
    avg_jacc = _mean(jacc)

    # Bucket distribution (from corr_metrics.corr_btc.buckets if present)
    highs = []
    lows = []
    for r in runs:
        try:
            b = (((r.get("corr_metrics") or {}).get("corr_btc") or {}).get("buckets") or {})
            highs.append(float(b.get("high", 0)))
            lows.append(float(b.get("low", 0)))
        except Exception:
            pass
    avg_high = _mean(highs)
    avg_low = _mean(lows)

    # Current parameters observed (take latest non-null)
    corr_max_cur = None
    rot_keep_cur = None
    budget_cur = None
    for r in reversed(runs):
        cm = _get_meta(r, "corr_max")
        if corr_max_cur is None and cm is not None:
            try:
                corr_max_cur = float(cm)
            except Exception:
                pass
        rk = _get_meta(r, "rotation_keep")
        if rot_keep_cur is None and rk is not None:
            try:
                rot_keep_cur = float(rk)
            except Exception:
                pass
        bd = _get_meta(r, "budget")
        if budget_cur is None and bd is not None:
            try:
                budget_cur = int(bd)
            except Exception:
                pass
        if corr_max_cur is not None and rot_keep_cur is not None:
            break

    # Defaults if not found
    if corr_max_cur is None:
        corr_max_cur = 0.85
    if rot_keep_cur is None:
        rot_keep_cur = 0.70
    if budget_cur is None:
        # Safe fallback: don't try to guess budget if logs don't carry it.
        budget_cur = None

    # Targets (slightly stricter for intraday)
    target_corr = 0.65 if mode_name == "intraday" else 0.70

    corr_max_new = float(corr_max_cur)
    # If too correlated, tighten corr_max; if too uncorrelated (and fills are hard), loosen.
    if avg_mean_pos is not None:
        if avg_mean_pos > target_corr + 0.05:
            corr_max_new -= 0.02
        elif avg_mean_pos < target_corr - 0.10:
            corr_max_new += 0.02
    corr_max_new = _clip(corr_max_new, 0.75, 0.92)

    # Rotation tuning: if repetition too high and corr too high, reduce keep ratio.
    rot_keep_new = float(rot_keep_cur)
    if avg_jacc is not None and avg_mean_pos is not None:
        if avg_jacc > 0.85 and avg_mean_pos > target_corr:
            rot_keep_new -= 0.05
        elif avg_jacc < 0.55:
            rot_keep_new += 0.05
    rot_keep_new = _clip(rot_keep_new, 0.55, 0.85)

    # Budget tuning (optional): only if we have elapsed_sec metrics and current budget.
    elapsed = []
    forced_runs = 0
    watch_runs = 0
    for r in runs:
        try:
            es = _get_meta(r, "elapsed_sec")
            if es is not None:
                elapsed.append(float(es))
        except Exception:
            pass
        try:
            rr = (r.get("recs") or {})
            if rr.get("has_forced") is True:
                forced_runs += 1
            if rr.get("has_watch") is True:
                watch_runs += 1
        except Exception:
            pass
    avg_elapsed = _mean(elapsed)
    forced_rate = forced_runs / max(1, len(runs))
    watch_rate = watch_runs / max(1, len(runs))

    budget_new = None
    if budget_cur is not None and avg_elapsed is not None:
        b = int(budget_cur)
        # Targets assume this project is running on a single box; keep inference under these budgets.
        target_sec = 12 * 60 if mode_name == "intraday" else 45 * 60
        # If too slow, reduce budget. If too many forced outputs (quality scarce) and time allows, increase.
        if avg_elapsed > target_sec * 1.10 and b > 8:
            b = max(8, b - 4)
        elif forced_rate > 0.35 and avg_elapsed < target_sec * 0.70:
            b = min(96, b + 4)
        # If watch-only often due to ops guard, do not change budget (it was likely offline/stale).
        if watch_rate < 0.60:
            budget_new = int(b)

    # Bucket quotas: if everything is BTC-high, shift budget to mid/low.
    # Keep within sane bounds and preserve sum=1.
    if mode_name == "intraday":
        q = {"high": 0.50, "mid": 0.35, "low": 0.15}
    else:
        q = {"high": 0.40, "mid": 0.40, "low": 0.20}

    if avg_high is not None and avg_low is not None:
        # avg_high/avg_low are counts averaged; normalize by typical budget if possible
        # Use only directional shifts.
        if avg_high >= 0.75 * max(1.0, avg_high + (avg_low or 0.0)):
            q["high"] = max(0.25, q["high"] - 0.05)
            q["low"] = min(0.30, q["low"] + 0.03)
            q["mid"] = 1.0 - q["high"] - q["low"]

    quota_env = f"high={q['high']:.2f},mid={q['mid']:.2f},low={q['low']:.2f}"

    return {
        "mode": mode_name,
        "window_runs": len(runs),
        "stats": {
            "avg_mean_pos_corr": avg_mean_pos,
            "avg_jaccard_markets": avg_jacc,
            "avg_corr_btc_high_count": avg_high,
            "avg_corr_btc_low_count": avg_low,
            "avg_elapsed_sec": avg_elapsed,
            "forced_rate": forced_rate,
            "watch_rate": watch_rate,
        },
        "current": {
            "selection_corr_max": corr_max_cur,
            "rotation_keep": rot_keep_cur,
            "market_budget": budget_cur,
        },
        "recommend": {
            "selection_corr_max": corr_max_new,
            "rotation_keep": rot_keep_new,
            "bucket_quotas_env": quota_env,
            "market_budget": budget_new,
        },
    }


def main():
    ap = argparse.ArgumentParser(description="Auto-tune ops parameters from run_markets_metrics jsonl logs.")
    ap.add_argument("--window_runs", type=int, default=42, help="How many most-recent runs to use per mode.")
    ap.add_argument(
        "--write",
        action="store_true",
        help="Write recommendations to analysis/ops_tuning.json (used by scripts/run_scheduled.py).",
    )
    args = ap.parse_args()

    base = os.path.join("analysis")
    intraday_path = os.path.join(base, "run_markets_metrics_intraday.jsonl")
    morning_path = os.path.join(base, "run_markets_metrics_morning.jsonl")

    intraday = _read_jsonl(intraday_path)[-max(1, int(args.window_runs)) :]
    morning = _read_jsonl(morning_path)[-max(1, int(args.window_runs)) :]

    rec = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "intraday": tune_from_runs(intraday, "intraday"),
        "morning": tune_from_runs(morning, "morning"),
    }

    os.makedirs(base, exist_ok=True)
    out_path = os.path.join(base, "ops_tuning.json")
    if args.write:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
    else:
        print(json.dumps(rec, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
