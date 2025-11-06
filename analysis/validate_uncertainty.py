import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from scipy.stats import spearmanr
from sklearn.calibration import calibration_curve

from utils.config import config
from utils.logger import logger
from utils.metrics import calculate_ece
from data.database import load_data
from inference import predictor



def main(args):
    logger.info(f"--- Starting Uncertainty Validation for the last {args.days} days ---")

    # 1. Define time range for analysis, now using timezone-aware UTC
    analysis_end_time = datetime.now(timezone.utc) - timedelta(days=1)
    analysis_start_time = analysis_end_time - timedelta(days=args.days)

    # We need historical data *before* the prediction time, and future data *after*
    data_load_start_time = analysis_start_time - timedelta(hours=config.SEQUENCE_LENGTH + 24)
    data_load_end_time = datetime.now(timezone.utc)

    logger.info(f"Loading market data from {data_load_start_time} to {data_load_end_time}...")
    all_data_query = f"SELECT * FROM crypto_data WHERE timestamp >= '{data_load_start_time}' AND timestamp <= '{data_load_end_time}'"
    all_df = load_data(all_data_query)
    if all_df.empty:
        logger.error("Not enough data to run analysis.")
        return
    
    # Standardize all timestamps to UTC
    all_df['timestamp'] = pd.to_datetime(all_df['timestamp']).dt.tz_localize('UTC')
    logger.info(f"Loaded {len(all_df)} total records.")

    # 3. Get a snapshot of predictions at the END of the analysis period
    target_time = analysis_end_time # Use the most recent time for prediction
    logger.info(f"Generating predictions for snapshot at {target_time}...")
    
    historical_data = all_df[all_df['timestamp'] <= target_time]
    active_markets_df = historical_data.groupby('market')['close'].count()
    markets_to_predict = active_markets_df[active_markets_df > 100].nlargest(20).index.tolist()
    if not markets_to_predict:
        logger.error(f"No markets with sufficient data found at {target_time} to make predictions.")
        return

    predictions = predictor.run(markets=markets_to_predict)
    if not predictions:
        logger.error("Predictor did not return any predictions.")
        return

    # 4. Collect ground truth and uncertainty data
    uncertainty_analysis_data = []
    for pred in predictions:
        future_time = target_time + timedelta(hours=6)
        future_price_df = all_df[(all_df['market'] == pred['market']) & (all_df['timestamp'] >= future_time)]
        
        if not future_price_df.empty:
            current_price = pred['current_price']
            future_price = future_price_df.iloc[0]['close']
            actual_return = (future_price / current_price) - 1
            predicted_return = np.sum(pred['predicted_pattern'])
            
            uncertainty_analysis_data.append({
                'predicted_return': predicted_return,
                'actual_return': actual_return,
                'uncertainty': pred['uncertainty'],
                'confidence': 1 / (1 + pred['uncertainty'])
            })

    if not uncertainty_analysis_data:
        logger.error("Could not gather any ground truth data to analyze.")
        return

    # 5. Perform and Log Uncertainty Analysis (Q2)
    ua_df = pd.DataFrame(uncertainty_analysis_data)
    logger.info("\n--- MC Dropout Uncertainty Analysis (Q2) ---")

    # a) ECE Calculation
    y_true_binary = (ua_df['actual_return'] > 0).astype(int)
    y_prob = ua_df['confidence']
    ece = calculate_ece(y_true_binary, y_prob)
    logger.info(f"Expected Calibration Error (ECE): {ece:.4f}")

    # b) Spearman Correlation
    error = np.abs(ua_df['predicted_return'] - ua_df['actual_return'])
    uncertainty = ua_df['uncertainty']
    spearman_corr, _ = spearmanr(error, uncertainty)
    logger.info(f"Spearman Correlation (|Error|, Uncertainty): {spearman_corr:.4f}")

    # c) Cutoff Optimization Curve Data
    logger.info("--- Uncertainty Cutoff vs. Performance ---")
    for percentile in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
        threshold = np.percentile(ua_df['uncertainty'], percentile)
        filtered_preds = ua_df[ua_df['uncertainty'] <= threshold]
        if not filtered_preds.empty:
            correct_direction = np.sign(filtered_preds['predicted_return']) == np.sign(filtered_preds['actual_return'])
            profitable_trades = filtered_preds[correct_direction]
            performance = profitable_trades['actual_return'].mean() if not profitable_trades.empty else 0
            recall = len(filtered_preds) / len(ua_df)
            logger.info(f"Threshold (Top {100-percentile}% Conf): {threshold:.4f} | Kept: {recall:.1%} | Mean Return: {performance:+.4f}")
    logger.info("============================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MC Dropout Uncertainty Validation Script")
    parser.add_argument('--days', type=int, default=3, help="Number of past days to use for analysis.")
    args = parser.parse_args()
    main(args)
