import pandas as pd
import numpy as np
import os
import json
from typing import Optional
from datetime import datetime, timedelta, timezone
from tqdm import tqdm
from scipy.stats import spearmanr

from utils.config import config
from utils.logger import logger
from data.database import load_data, get_latest_db_timestamp
from data.preprocessor import get_market_index
from inference import predictor, recommender
from utils import screener
from analysis.validate_uncertainty import calculate_ece

INITIAL_BALANCE = 10_000_000  # 1,000만원


def compute_pi_coverage(
    pi_low_80: np.ndarray,
    pi_high_80: np.ndarray,
    actual_returns: np.ndarray,
) -> dict:
    """Compute empirical coverage and width of 80% prediction intervals.

    Args:
        pi_low_80: Lower bounds of PI_80 intervals, shape (n,).
        pi_high_80: Upper bounds of PI_80 intervals, shape (n,).
        actual_returns: Observed returns, shape (n,).

    Returns:
        Dict with empirical_coverage_80, expected_coverage, calibration_gap,
        mean_interval_width.  Returns None-filled dict when inputs are empty.
    """
    pi_low_80 = np.asarray(pi_low_80, dtype=float)
    pi_high_80 = np.asarray(pi_high_80, dtype=float)
    actual_returns = np.asarray(actual_returns, dtype=float)

    # Drop rows where any value is NaN
    valid = ~(np.isnan(pi_low_80) | np.isnan(pi_high_80) | np.isnan(actual_returns))
    pi_low_80 = pi_low_80[valid]
    pi_high_80 = pi_high_80[valid]
    actual_returns = actual_returns[valid]

    if len(actual_returns) == 0:
        return {
            "empirical_coverage_80": None,
            "expected_coverage": 0.8,
            "calibration_gap": None,
            "mean_interval_width": None,
            "n_valid": 0,
        }

    covered = (actual_returns >= pi_low_80) & (actual_returns <= pi_high_80)
    empirical_coverage = float(covered.mean())
    mean_width = float((pi_high_80 - pi_low_80).mean())

    return {
        "empirical_coverage_80": empirical_coverage,
        "expected_coverage": 0.8,
        "calibration_gap": empirical_coverage - 0.8,
        "mean_interval_width": mean_width,
        "n_valid": int(len(actual_returns)),
    }

def _parse_end_time_env() -> Optional[datetime]:
    v = str(os.getenv("AETHER_BACKTEST_END_TIME_ISO", "") or "").strip()
    if not v:
        return None
    try:
        ts = pd.to_datetime(v, utc=True, errors="raise")
        if isinstance(ts, pd.Timestamp):
            return ts.to_pydatetime()
    except Exception as e:
        logger.warning(f"Invalid AETHER_BACKTEST_END_TIME_ISO={v!r}: {e}")
    return None


def _parse_recommender_mode_env() -> str:
    value = str(os.getenv("AETHER_BACKTEST_RECOMMENDER_MODE", "backtest") or "backtest").strip().lower()
    if value in {"backtest", "live"}:
        return value
    logger.warning(f"Invalid AETHER_BACKTEST_RECOMMENDER_MODE={value!r}; falling back to 'backtest'.")
    return "backtest"


def _parse_strategy_env() -> Optional[str]:
    value = str(os.getenv("AETHER_BACKTEST_STRATEGY", "") or "").strip().lower()
    return value or None


def _parse_screen_limit_env(default: int = 5) -> int:
    raw = str(os.getenv("AETHER_BACKTEST_SCREEN_LIMIT", str(int(default))) or str(int(default))).strip()
    try:
        return max(1, int(raw))
    except Exception:
        logger.warning(f"Invalid AETHER_BACKTEST_SCREEN_LIMIT={raw!r}; falling back to {int(default)}.")
        return int(default)


def _parse_min_k_env(default: int) -> int:
    raw = str(os.getenv("AETHER_BACKTEST_MIN_K", str(int(default))) or str(int(default))).strip()
    try:
        return max(1, int(raw))
    except Exception:
        logger.warning(f"Invalid AETHER_BACKTEST_MIN_K={raw!r}; falling back to {int(default)}.")
        return int(default)


def _configured_future_horizon_hours() -> int:
    try:
        return max(1, int(getattr(config.Data, "FUTURE_WINDOW_SIZE", 1) or 1))
    except Exception:
        return 1


def _is_watch_only_output(rec: dict) -> bool:
    status = str(rec.get("status", "") or "").strip().lower()
    if "watch" in status:
        return True
    try:
        position_size = float(rec.get("position_size", 0.0) or 0.0)
    except Exception:
        position_size = 0.0
    return position_size <= 0.0


def _calc_trade_pnl(row: pd.Series) -> float:
    signal = str(row.get("signal", "")).strip().lower()
    if signal not in {"long", "short"}:
        signal = "short" if float(row["predicted_return"]) < 0 else "long"
    raw_return = float(row["actual_return"])
    fee = float(getattr(config.Recommender, "TRADE_FEE_PER_LEG", 0.0005) or 0.0005)
    slippage = float(getattr(config.Recommender, "SLIPPAGE_PER_LEG", 0.0003) or 0.0003)
    round_trip_cost = 2.0 * (max(0.0, fee) + max(0.0, slippage))
    gross = -raw_return if signal == "short" else raw_return
    return gross - round_trip_cost


def _merge_count_dict(target: dict, incoming: dict | None) -> None:
    if not incoming:
        return
    for key, value in incoming.items():
        try:
            key_str = str(key)
            if isinstance(value, dict):
                nested_target = target.get(key_str)
                if not isinstance(nested_target, dict):
                    nested_target = {}
                target[key_str] = nested_target
                _merge_count_dict(nested_target, value)
                continue
            target[key_str] = target.get(key_str, 0) + int(value)
        except Exception:
            continue

def run(days_to_backtest: int = 10, stride_hours: int = 1, summary_tag: str = "") -> bool:
    """Runs a realistic, efficient backtest for the given number of days.

    Returns:
        bool: True if backtest produced analyzable trades, otherwise False.
    """
    stride_hours = max(1, int(stride_hours or 1))
    recommender_mode = _parse_recommender_mode_env()
    strategy_override = _parse_strategy_env()
    screen_limit = _parse_screen_limit_env(default=5)
    default_min_k = 1 if recommender_mode == "live" and strategy_override == "intraday" else 3
    min_k = _parse_min_k_env(default=default_min_k)
    future_horizon_hours = _configured_future_horizon_hours()

    logger.info(
        f"=== Starting Backtest for the last {days_to_backtest} days "
        f"(stride={stride_hours}h, recommender_mode={recommender_mode}, "
        f"strategy={strategy_override or 'default'}, screen_limit={screen_limit}, "
        f"min_k={min_k}, horizon={future_horizon_hours}h) ==="
    )

    # Load the model configuration to ensure consistent architecture
    config_path = os.path.join("models", "model_config.json")
    if os.path.exists(config_path):
        logger.info(f"Loading model configuration from {config_path} for backtest.")
        with open(config_path, 'r') as f:
            best_params = json.load(f)
            # Remap only architecture-related params to avoid overriding backtest-specific settings
            config.Gan.D_MODEL = best_params.get('d_model', config.Gan.D_MODEL)
            config.Gan.N_LAYERS = best_params.get('n_layers', config.Gan.N_LAYERS)
            config.Gan.N_HEADS = best_params.get('n_heads', config.Gan.N_HEADS)
            config.Gan.DROPOUT_P = best_params.get('dropout_p', config.Gan.DROPOUT_P)
            logger.info(f"Remapped model architecture parameters for backtest.")
    else:
        logger.warning("model_config.json not found. Backtest will use default model parameters.")

    # 1. Pre-load all data for the entire backtest period + buffer
    requested_end_time = _parse_end_time_env() or datetime.now(timezone.utc)
    db_latest = get_latest_db_timestamp()
    # Clamp end_time to DB latest to avoid simulating into the future.
    if db_latest is not None and requested_end_time > db_latest:
        end_time = db_latest
    else:
        end_time = requested_end_time
    start_time = end_time - timedelta(days=days_to_backtest)
    data_load_start_time = start_time - timedelta(hours=config.Data.SEQUENCE_LENGTH + 100) # Buffer for initial sequences

    logger.info(f"Loading all market data from {data_load_start_time} to {end_time}...")
    start_s = data_load_start_time.strftime('%Y-%m-%dT%H:%M:%S')
    end_s = end_time.strftime('%Y-%m-%dT%H:%M:%S')
    all_data_query = f"SELECT * FROM crypto_data WHERE timestamp >= '{start_s}' AND timestamp <= '{end_s}'"
    all_df = load_data(all_data_query)
    if all_df.empty:
        logger.error("Not enough data to run backtest.")
        return False
    all_df['timestamp'] = pd.to_datetime(all_df['timestamp']).dt.tz_localize('UTC')
    all_df.set_index('timestamp', inplace=True)
    logger.info(f"Loaded {len(all_df)} total records for the backtesting period.")

    market_index_df = get_market_index(start_date=data_load_start_time, end_date=end_time)

    # 2. Iterate through time, hour by hour, making predictions
    all_trades_for_analysis = []
    portfolio_values = [INITIAL_BALANCE]
    total_hours = int(days_to_backtest * 24)
    step1_reject_aggregate: dict[str, int] = {}
    step1_signal_aggregate: dict[str, int] = {}
    step1_signal_reject_aggregate: dict[str, dict] = {}
    step1_signal_stats_runs: list[dict] = []
    funnel_active_aggregate: dict[str, int] = {}

    for hour in tqdm(range(0, total_hours, stride_hours), desc="Backtesting Progress"):
        current_time = start_time + timedelta(hours=hour)

        # Warmup check
        if (current_time - data_load_start_time).total_seconds() / 3600 < config.Data.SEQUENCE_LENGTH:
            if hour % 24 == 0: logger.info(f"Backtest Hour: {hour+1} | Skipping - In Warmup Period")
            portfolio_values.append(portfolio_values[-1])
            continue

        try:
            # Use only data available up to the current point in time for decisions
            historical_data = all_df[all_df.index <= current_time]

            # A. Screener
            trending_markets = screener.get_trending_markets(
                historical_data,
                current_time=current_time,
                limit=screen_limit,
                mode=recommender_mode,
            )
            if not trending_markets:
                portfolio_values.append(portfolio_values[-1])
                continue

            # B. Predictor
            predictions = predictor.run(markets=trending_markets, market_index_df=market_index_df, historical_df=historical_data)
            if not predictions:
                portfolio_values.append(portfolio_values[-1])
                continue

            if strategy_override:
                for pred in predictions:
                    pred["strategy"] = strategy_override

            # C. Recommender (Filters)
            recommendations = recommender.run(
                predictions,
                historical_data,
                mode=recommender_mode,
                min_k=min_k,
                reference_time=current_time,
                emit_artifacts=False,
                collect_diagnostics=True,
            )
            diag_payload = recommender.get_last_run_diagnostics() or {}
            _merge_count_dict(step1_reject_aggregate, diag_payload.get("step1_reject_counts"))
            _merge_count_dict(step1_signal_aggregate, diag_payload.get("step1_signal_counts"))
            _merge_count_dict(step1_signal_reject_aggregate, diag_payload.get("step1_signal_reject_counts"))
            if diag_payload.get("step1_signal_stats"):
                step1_signal_stats_runs.append(diag_payload.get("step1_signal_stats"))
            for step in diag_payload.get("funnel_steps", []) or []:
                step_name = str(step.get("step", "") or "").strip()
                if not step_name:
                    continue
                try:
                    funnel_active_aggregate[step_name] = funnel_active_aggregate.get(step_name, 0) + int(step.get("active_n", 0) or 0)
                except Exception:
                    continue
            # In a real trading bot, you'd manage positions here. For this evaluation, we just log the outcome.

            # 3. Log trades and calculate ground truth for analysis
            for rec in recommendations:
                future_time = current_time + timedelta(hours=future_horizon_hours)
                future_price_df = all_df[(all_df['market'] == rec['market']) & (all_df.index >= future_time)]
                
                if not future_price_df.empty:
                    actual_future_price = future_price_df.iloc[0]['close']
                    actual_return = (actual_future_price / rec['current_price']) - 1
                    
                    # Use 'pattern' (from recommender) or fallback to 'predicted_pattern' (from predictor)
                    pattern = rec.get('pattern', rec.get('predicted_pattern'))
                    if pattern is None:
                        logger.warning(f"No pattern found for {rec['market']}, skipping")
                        continue
                    
                    pattern = np.array(pattern, dtype=float).reshape(-1)
                    predicted_return = float(np.prod(1 + pattern) - 1) if pattern.size else 0.0
                    error = np.abs(predicted_return - actual_return)
                    
                    is_watch_only = _is_watch_only_output(rec)
                    all_trades_for_analysis.append({
                        "timestamp": current_time,
                        "market": rec['market'],
                        "signal": rec.get('signal'),
                        "status": rec.get("status"),
                        "trend_alignment": rec.get("trend_alignment"),
                        "decision_score": rec.get("decision_score"),
                        "position_size": rec.get("position_size", 0.0),
                        "is_watch_only": is_watch_only,
                        "predicted_return": predicted_return,
                        "actual_return": actual_return,
                        "uncertainty": rec.get('uncertainty', 0),
                        "confidence": rec.get('confidence', 1 / (1 + rec.get('uncertainty', 0))),
                        "error": error,
                        "correct": np.sign(predicted_return) == np.sign(actual_return),
                        # PI_80 bounds (for coverage calibration)
                        "pi_low_80": rec.get('pi_low_80'),
                        "pi_high_80": rec.get('pi_high_80'),
                        # Gate Analysis (for paper figures)
                        "gate_value": rec.get('gate_value', 0.5),
                        "consensus_score": rec.get('consensus_score', 0.6)
                    })
            
            # Simplified portfolio value update for logging
            portfolio_values.append(portfolio_values[-1]) # Placeholder, real logic would be more complex

        except Exception as e:
            logger.error(f"Error during backtest hour {hour+1} at {current_time}: {e}", exc_info=True)
            portfolio_values.append(portfolio_values[-1])
            continue

    if not all_trades_for_analysis:
        logger.error("Backtest finished with no outputs to analyze.")
        return False

    # 4. --- Performance & Uncertainty Analysis ---
    outputs_df = pd.DataFrame(all_trades_for_analysis)
    outputs_df['pnl'] = outputs_df.apply(_calc_trade_pnl, axis=1)
    outputs_df['correct'] = outputs_df['pnl'] > 0

    results_df = outputs_df[~outputs_df['is_watch_only']].copy()
    watch_only_count = int(outputs_df['is_watch_only'].sum())
    output_count = int(len(outputs_df))
    outputs_by_signal = outputs_df['signal'].fillna('Unknown').astype(str).value_counts().to_dict()
    watch_outputs_by_signal = outputs_df[outputs_df['is_watch_only']]['signal'].fillna('Unknown').astype(str).value_counts().to_dict()
    trades_by_signal = results_df['signal'].fillna('Unknown').astype(str).value_counts().to_dict()

    if results_df.empty:
        logger.warning("Backtest produced outputs, but all were watch-only. No executable trades to score.")
        pnl = pd.Series(dtype=float)
        sharpe_ratio = 0.0
        win_rate = 0.0
        max_drawdown_pct = 0.0
    else:
        pnl = results_df['pnl']
        results_df['correct'] = pnl > 0

        # Group by timestamp for portfolio-level MDD and Sharpe (avoid sequential compounding of concurrent trades)
        if 'position_size' in results_df.columns and 'timestamp' in results_df.columns:
            time_pnl = results_df.groupby('timestamp').apply(
                lambda g: (g['position_size'] * g['pnl']).sum()
            )
        else:
            time_pnl = pnl  # fallback: treat each trade as sequential

        # Sharpe: annualized using actual number of trading periods observed
        backtest_days = max(1, days_to_backtest)
        actual_periods = len(time_pnl)
        periods_per_year = actual_periods / backtest_days * 365
        sharpe_ratio = (time_pnl.mean() / (time_pnl.std() + 1e-9)) * np.sqrt(periods_per_year)

        # MDD: portfolio value series from per-timestamp weighted returns
        portfolio_value_series = (1 + time_pnl).cumprod() * INITIAL_BALANCE
        rolling_max = portfolio_value_series.cummax()
        drawdown = (portfolio_value_series - rolling_max) / rolling_max
        max_drawdown_pct = (drawdown.min() * 100) if not drawdown.empty else 0
        win_rate = results_df['correct'].mean() * 100

    logger.info("\n--- Backtest Results ---")
    logger.info(f"Period: {days_to_backtest} days")
    logger.info(f"Stride: {stride_hours} hours")
    logger.info(f"Outputs Collected: {output_count} (watch-only={watch_only_count}, executable={len(results_df)})")
    wins = int(results_df['correct'].sum()) if not results_df.empty else 0
    losses = int(len(results_df) - wins)
    logger.info(f"Win Rate: {win_rate:.2f}% ({wins} wins / {losses} losses)")
    logger.info(f"Avg Return per Trade: {(float(pnl.mean()) if not pnl.empty else 0.0):.2%}")
    logger.info(f"Sharpe Ratio (annualized): {sharpe_ratio:.2f}")
    logger.info(f"Max Drawdown: {max_drawdown_pct:.2f}%")
    logger.info("========================")

    # --- Detailed Analysis Reports ---
    ece = None
    spearman_corr = None
    logger.info("\n--- Uncertainty & Calibration Analysis ---")
    if results_df.empty:
        logger.warning("Skipping calibration analysis because there are no executable trades.")
    else:
        y_true_binary = (results_df['pnl'] > 0).astype(int)
        y_prob = results_df['confidence']
        
        # a) ECE Calculation (Using robust, imported function)
        ece = calculate_ece(y_true_binary.to_numpy(), y_prob.to_numpy(), n_bins=15)
        logger.info(f"Expected Calibration Error (ECE): {ece:.4f}")

        # b) Spearman Correlation
        errors = results_df['error']
        uncertainties = results_df['uncertainty']
        spearman_corr, _ = spearmanr(errors, uncertainties)
        logger.info(f"Spearman Correlation (|Error|, Uncertainty): {spearman_corr:.4f}")

        # c) Cutoff Optimization Curve Data
        logger.info("--- Uncertainty Cutoff vs. Performance ---")
        for percentile in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
            threshold = np.percentile(results_df['uncertainty'], percentile)
            filtered_df = results_df[results_df['uncertainty'] <= threshold]
            if not filtered_df.empty:
                profitable_trades = filtered_df[filtered_df['pnl'] > 0]
                mean_win_return = profitable_trades['pnl'].mean() if not profitable_trades.empty else 0.0
                recall = len(filtered_df) / len(results_df)
                win_rate_filtered = (filtered_df['pnl'] > 0).mean()
                logger.info(f"Threshold (Top {100-percentile}% Conf): {threshold:.4f} | Kept: {recall:.1%} | Win Rate: {win_rate_filtered:.1%} | Mean Return on Wins: {mean_win_return:+.4f}")
        logger.info("============================================")

    # --- PI_80 Coverage Calibration ---
    pi_coverage = None
    if "pi_low_80" in outputs_df.columns and "pi_high_80" in outputs_df.columns:
        pi_coverage = compute_pi_coverage(
            pi_low_80=outputs_df["pi_low_80"].values,
            pi_high_80=outputs_df["pi_high_80"].values,
            actual_returns=outputs_df["actual_return"].values,
        )
        if pi_coverage["empirical_coverage_80"] is not None:
            cov = pi_coverage["empirical_coverage_80"]
            mpiw = pi_coverage["mean_interval_width"]
            gap = pi_coverage["calibration_gap"]
            n_valid = pi_coverage["n_valid"]
            logger.info(f"PI_80 Coverage: {cov:.1%} (target 80%) | Width: {mpiw:.4f} | Gap: {gap:+.1%} | n={n_valid}")
        else:
            logger.warning("PI_80 coverage: no valid prediction intervals found in outputs.")
    else:
        logger.info("PI_80 coverage: pi_low_80/pi_high_80 not present in outputs, skipping.")

    # --- Export Gate Values for Paper Visualization ---
    os.makedirs("analysis", exist_ok=True)

    # Avoid clobbering live/intraday gate_values.csv (used for dashboards).
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{summary_tag}" if str(summary_tag).strip() else ""
    gate_csv_path = os.path.join("analysis", f"backtest_gate_values{tag}_{ts}.csv")
    outputs_df.to_csv(gate_csv_path, index=False)
    logger.info(f"Backtest gate analysis data saved to {gate_csv_path}")

    # Write a machine-readable summary for eval pipelines.
    summary = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "days": int(days_to_backtest),
        "stride_hours": int(stride_hours),
        "recommender_mode": recommender_mode,
        "strategy": strategy_override,
        "screen_limit": int(screen_limit),
        "min_k": int(min_k),
        "future_horizon_hours": int(future_horizon_hours),
        "end_time": end_time.isoformat(),
        "requested_end_time": requested_end_time.isoformat(),
        "db_latest": db_latest.isoformat() if db_latest is not None else None,
        "n_outputs": int(output_count),
        "n_watch_only_outputs": int(watch_only_count),
        "n_trades": int(len(results_df)),
        "win_rate_pct": float(win_rate),
        "avg_return_per_trade": float(pnl.mean()) if not pnl.empty else 0.0,
        "sharpe_annualized": float(sharpe_ratio),
        "max_drawdown_pct": float(max_drawdown_pct),
        "ece": float(ece) if ece is not None else None,
        "spearman_abs_error_uncertainty": float(spearman_corr) if spearman_corr is not None else None,
        "step1_reject_aggregate": step1_reject_aggregate,
        "step1_signal_aggregate": step1_signal_aggregate,
        "step1_signal_reject_aggregate": step1_signal_reject_aggregate,
        "step1_signal_stats_runs": step1_signal_stats_runs,
        "funnel_active_aggregate": funnel_active_aggregate,
        "outputs_by_signal": outputs_by_signal,
        "watch_outputs_by_signal": watch_outputs_by_signal,
        "trades_by_signal": trades_by_signal,
        "gate_csv": gate_csv_path,
        "pi_coverage_80": pi_coverage,
    }
    try:
        latest_path = os.path.join("analysis", "backtest_summary_latest.json")
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        run_path = os.path.join("analysis", f"backtest_summary{tag}_{ts}.json")
        with open(run_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info(f"Backtest summary saved to {run_path}")
    except Exception as e:
        logger.warning(f"Failed to write backtest summary JSON: {e}")
    return True
