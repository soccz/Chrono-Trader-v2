import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta, timezone
from tqdm import tqdm
from scipy.stats import spearmanr

from utils.config import config
from utils.logger import logger
from data.database import load_data
from data.preprocessor import get_market_index
from inference import predictor, recommender
from utils import screener
from analysis.validate_uncertainty import calculate_ece

INITIAL_BALANCE = 10_000_000  # 1,000만원

def run(days_to_backtest: int = 10, stride_hours: int = 1, summary_tag: str = "") -> bool:
    """Runs a realistic, efficient backtest for the given number of days.

    Returns:
        bool: True if backtest produced analyzable trades, otherwise False.
    """
    stride_hours = max(1, int(stride_hours or 1))
    logger.info(f"=== Starting Backtest for the last {days_to_backtest} days (stride={stride_hours}h) ===")

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
    end_time = datetime.now(timezone.utc)
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
            trending_markets = screener.get_trending_markets(historical_data, mode='backtest')
            if not trending_markets:
                portfolio_values.append(portfolio_values[-1])
                continue

            # B. Predictor
            predictions = predictor.run(markets=trending_markets, market_index_df=market_index_df, historical_df=historical_data)
            if not predictions:
                portfolio_values.append(portfolio_values[-1])
                continue

            # C. Recommender (Filters)
            recommendations = recommender.run(predictions, historical_data, mode='backtest')
            # In a real trading bot, you'd manage positions here. For this evaluation, we just log the outcome.

            # 3. Log trades and calculate ground truth for analysis
            for rec in recommendations:
                future_time = current_time + timedelta(hours=6)
                future_price_df = all_df[(all_df['market'] == rec['market']) & (all_df.index >= future_time)]
                
                if not future_price_df.empty:
                    actual_future_price = future_price_df.iloc[0]['close']
                    actual_return = (actual_future_price / rec['current_price']) - 1
                    
                    # Use 'pattern' (from recommender) or fallback to 'predicted_pattern' (from predictor)
                    pattern = rec.get('pattern', rec.get('predicted_pattern'))
                    if pattern is None:
                        logger.warning(f"No pattern found for {rec['market']}, skipping")
                        continue
                    
                    predicted_return = np.sum(pattern)
                    error = np.abs(predicted_return - actual_return)
                    
                    all_trades_for_analysis.append({
                        "timestamp": current_time,
                        "market": rec['market'],
                        "signal": rec.get('signal'),
                        "predicted_return": predicted_return,
                        "actual_return": actual_return,
                        "uncertainty": rec.get('uncertainty', 0),
                        "confidence": 1 / (1 + rec.get('uncertainty', 0)),
                        "error": error,
                        "correct": np.sign(predicted_return) == np.sign(actual_return),
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
        logger.error("Backtest finished with no trades to analyze.")
        return False

    # 4. --- Performance & Uncertainty Analysis ---
    results_df = pd.DataFrame(all_trades_for_analysis)

    def _calc_trade_pnl(row):
        # Respect trade direction first. If missing, infer direction from predicted return sign.
        signal = str(row.get('signal', '')).strip().lower()
        if signal not in {'long', 'short'}:
            signal = 'short' if row['predicted_return'] < 0 else 'long'
        raw_return = float(row['actual_return'])
        return -raw_return if signal == 'short' else raw_return

    # Direction-aware PnL so short wins are positive and short losses are negative.
    results_df['pnl'] = results_df.apply(_calc_trade_pnl, axis=1)
    pnl = results_df['pnl']
    results_df['correct'] = pnl > 0

    sharpe_ratio = (pnl.mean() / (pnl.std() + 1e-9)) * np.sqrt(365*4) # Annualized for 6-hour trades
    win_rate = results_df['correct'].mean() * 100
    
    # Corrected Drawdown Calculation
    # Create a series of portfolio values by compounding returns
    portfolio_value_series = (1 + pnl).cumprod() * INITIAL_BALANCE
    rolling_max = portfolio_value_series.cummax()
    drawdown = (portfolio_value_series - rolling_max) / rolling_max
    max_drawdown_pct = (drawdown.min() * 100) if not drawdown.empty else 0

    logger.info("\n--- Backtest Results ---")
    logger.info(f"Period: {days_to_backtest} days")
    logger.info(f"Stride: {stride_hours} hours")
    logger.info(f"Total Trades Analyzed: {len(results_df)}")
    wins = int(results_df['correct'].sum())
    losses = int(len(results_df) - wins)
    logger.info(f"Win Rate: {win_rate:.2f}% ({wins} wins / {losses} losses)")
    logger.info(f"Avg Return per Trade: {pnl.mean():.2%}")
    logger.info(f"Sharpe Ratio (annualized): {sharpe_ratio:.2f}")
    logger.info(f"Max Drawdown: {max_drawdown_pct:.2f}%")
    logger.info("========================")

    # --- Detailed Analysis Reports ---
    logger.info("\n--- Uncertainty & Calibration Analysis ---")
    y_true_binary = (results_df['actual_return'] > 0).astype(int)
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
    
    # --- Export Gate Values for Paper Visualization ---
    os.makedirs("analysis", exist_ok=True)

    # Avoid clobbering live/intraday gate_values.csv (used for dashboards).
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{summary_tag}" if str(summary_tag).strip() else ""
    gate_csv_path = os.path.join("analysis", f"backtest_gate_values{tag}_{ts}.csv")
    results_df.to_csv(gate_csv_path, index=False)
    logger.info(f"Backtest gate analysis data saved to {gate_csv_path}")

    # Write a machine-readable summary for eval pipelines.
    summary = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "days": int(days_to_backtest),
        "stride_hours": int(stride_hours),
        "n_trades": int(len(results_df)),
        "win_rate_pct": float(win_rate),
        "avg_return_per_trade": float(pnl.mean()),
        "sharpe_annualized": float(sharpe_ratio),
        "max_drawdown_pct": float(max_drawdown_pct),
        "ece": float(ece),
        "spearman_abs_error_uncertainty": float(spearman_corr),
        "gate_csv": gate_csv_path,
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
