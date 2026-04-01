"""
Cross-sectional momentum backtest on Upbit 4h data.
Strategy: buy top-5 momentum coins (past 4h return), hold 4h.
Also test: reversal (buy bottom-5).
Filter: daily volume > 100M KRW.
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "crypto_data.db"

# ── 1. Load data ──────────────────────────────────────────────
conn = sqlite3.connect(str(DB_PATH))
df = pd.read_sql(
    "SELECT market, timestamp, close, volume FROM crypto_data "
    "WHERE timestamp >= '2025-12-28' ORDER BY timestamp",
    conn,
)
conn.close()

df["timestamp"] = pd.to_datetime(df["timestamp"])
print(f"Loaded {len(df):,} rows, {df['market'].nunique()} coins, "
      f"{df['timestamp'].min()} → {df['timestamp'].max()}")

# ── 2. Compute trade value (KRW) per hour ────────────────────
df["trade_value"] = df["close"] * df["volume"]

# ── 3. Resample to 4h bars ───────────────────────────────────
# For each (market, 4h-window): last close, sum of trade_value
df = df.set_index("timestamp")
grouped = df.groupby("market")

bars = []
for mkt, g in grouped:
    g = g.sort_index()
    r = g.resample("4h").agg({"close": "last", "trade_value": "sum"}).dropna()
    r["market"] = mkt
    bars.append(r)

bars = pd.concat(bars).reset_index()
bars = bars.sort_values(["timestamp", "market"]).reset_index(drop=True)
print(f"4h bars: {len(bars):,}")

# ── 4. Daily volume filter (>100M KRW / day) ─────────────────
bars["date"] = bars["timestamp"].dt.date
daily_vol = bars.groupby(["market", "date"])["trade_value"].sum().reset_index()
daily_vol.columns = ["market", "date", "daily_trade_value"]

# Coins that pass filter on a given date
MIN_DAILY_VOL = 1e8  # 100M KRW
liquid = daily_vol[daily_vol["daily_trade_value"] >= MIN_DAILY_VOL][["market", "date"]].copy()
liquid["liquid"] = True

bars = bars.merge(liquid, on=["market", "date"], how="left")
bars["liquid"] = bars["liquid"].fillna(False)
bars = bars[bars["liquid"]].drop(columns=["liquid", "date"]).reset_index(drop=True)

n_liquid = bars["market"].nunique()
print(f"Liquid coins (>100M KRW/day): {n_liquid}")

# ── 5. Compute 4h returns ────────────────────────────────────
bars = bars.sort_values(["market", "timestamp"])
bars["ret_4h"] = bars.groupby("market")["close"].pct_change()
bars = bars.dropna(subset=["ret_4h"]).reset_index(drop=True)

# ── 6. Cross-sectional backtest ──────────────────────────────
TOP_N = 5
timestamps = sorted(bars["timestamp"].unique())
print(f"Total 4h periods: {len(timestamps)}")

records = []

for i in range(1, len(timestamps)):
    ts_signal = timestamps[i - 1]  # signal period (past 4h return)
    ts_hold = timestamps[i]        # holding period (next 4h return)

    sig = bars[bars["timestamp"] == ts_signal][["market", "ret_4h"]].set_index("market")
    hold = bars[bars["timestamp"] == ts_hold][["market", "ret_4h"]].set_index("market")

    # Only coins present in both periods
    common = sig.index.intersection(hold.index)
    if len(common) < TOP_N * 2:
        continue

    sig = sig.loc[common]
    hold = hold.loc[common]

    # Rank by past 4h return
    ranked = sig["ret_4h"].sort_values(ascending=False)

    top5 = ranked.head(TOP_N).index
    bot5 = ranked.tail(TOP_N).index

    top5_ret = hold.loc[top5, "ret_4h"].mean()
    bot5_ret = hold.loc[bot5, "ret_4h"].mean()
    universe_mean = hold["ret_4h"].mean()
    universe_median = hold["ret_4h"].median()

    records.append({
        "timestamp": ts_hold,
        "n_coins": len(common),
        "top5_ret": top5_ret,
        "bot5_ret": bot5_ret,
        "universe_mean": universe_mean,
        "universe_median": universe_median,
    })

results = pd.DataFrame(records)
print(f"\nBacktest periods: {len(results)}")
print(f"Date range: {results['timestamp'].min()} → {results['timestamp'].max()}")
print(f"Avg coins per period: {results['n_coins'].mean():.0f}")

# ── 7. Metrics ───────────────────────────────────────────────

def calc_metrics(ret_series, label, benchmark_mean, benchmark_median):
    """Calculate strategy metrics."""
    excess = ret_series - benchmark_mean
    win = (ret_series > benchmark_median).mean()

    # Sharpe (annualized from 4h periods, ~2190 periods/year)
    periods_per_year = 365.25 * 24 / 4
    sharpe = (excess.mean() / excess.std()) * np.sqrt(periods_per_year) if excess.std() > 0 else 0

    # Cumulative PnL and MDD
    cum = (1 + ret_series).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    mdd = dd.min()

    # Also compute cum return
    total_ret = cum.iloc[-1] - 1

    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  Win rate (> universe median): {win:.1%}")
    print(f"  Avg 4h return:               {ret_series.mean():.4%}")
    print(f"  Avg universe mean:           {benchmark_mean.mean():.4%}")
    print(f"  Avg excess return:           {excess.mean():.4%}")
    print(f"  Excess return std:           {excess.std():.4%}")
    print(f"  Sharpe (annualized):         {sharpe:.2f}")
    print(f"  Max Drawdown:                {mdd:.2%}")
    print(f"  Total cumulative return:     {total_ret:.2%}")
    print(f"  Periods: {len(ret_series)}")

    return {
        "label": label,
        "win_rate": win,
        "avg_ret": ret_series.mean(),
        "avg_excess": excess.mean(),
        "sharpe": sharpe,
        "mdd": mdd,
        "total_cum_ret": total_ret,
    }


m1 = calc_metrics(
    results["top5_ret"], "MOMENTUM (buy top 5)",
    results["universe_mean"], results["universe_median"],
)

m2 = calc_metrics(
    results["bot5_ret"], "REVERSAL (buy bottom 5)",
    results["universe_mean"], results["universe_median"],
)

# Universe baseline
m3 = calc_metrics(
    results["universe_mean"], "UNIVERSE EQUAL-WEIGHT",
    results["universe_mean"], results["universe_median"],
)

# ── 8. Additional analysis ───────────────────────────────────
print(f"\n{'='*50}")
print(f"  ADDITIONAL ANALYSIS")
print(f"{'='*50}")

# Correlation between signal return and forward return
from scipy import stats

all_corrs = []
for i in range(1, len(timestamps)):
    ts_signal = timestamps[i - 1]
    ts_hold = timestamps[i]
    sig = bars[bars["timestamp"] == ts_signal][["market", "ret_4h"]].set_index("market")
    hold = bars[bars["timestamp"] == ts_hold][["market", "ret_4h"]].set_index("market")
    common = sig.index.intersection(hold.index)
    if len(common) < 10:
        continue
    corr, _ = stats.spearmanr(sig.loc[common, "ret_4h"], hold.loc[common, "ret_4h"])
    all_corrs.append(corr)

avg_corr = np.mean(all_corrs)
print(f"  Avg Spearman rank-corr (signal vs fwd ret): {avg_corr:.4f}")
print(f"  Positive corr periods: {np.mean(np.array(all_corrs) > 0):.1%}")
print(f"  Corr std: {np.std(all_corrs):.4f}")
print(f"  Corr 25/50/75 pctile: {np.percentile(all_corrs, [25,50,75])}")

# Weekly breakdown
results["week"] = results["timestamp"].dt.isocalendar().week.astype(int)
weekly = results.groupby("week").agg(
    mom_ret=("top5_ret", "mean"),
    rev_ret=("bot5_ret", "mean"),
    uni_ret=("universe_mean", "mean"),
).reset_index()
weekly["mom_excess"] = weekly["mom_ret"] - weekly["uni_ret"]
weekly["rev_excess"] = weekly["rev_ret"] - weekly["uni_ret"]

print(f"\n  Weekly momentum excess return:")
print(f"  Positive weeks: {(weekly['mom_excess'] > 0).sum()} / {len(weekly)}")
print(f"  Avg weekly mom excess: {weekly['mom_excess'].mean():.4%}")
print(f"  Avg weekly rev excess: {weekly['rev_excess'].mean():.4%}")

# ── 9. Summary table ─────────────────────────────────────────
print(f"\n{'='*50}")
print(f"  SUMMARY")
print(f"{'='*50}")
summary = pd.DataFrame([m1, m2, m3]).set_index("label")
print(summary.to_string(float_format=lambda x: f"{x:.4f}"))
