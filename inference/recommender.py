import pandas as pd
import numpy as np
import os
from datetime import datetime, timezone

from utils.config import config
from utils.logger import logger
from data.database import get_trading_values_for_markets
from data.collector import get_current_price
from data.preprocessor import get_historical_success_patterns
from inference.predictor import get_pattern_similarity

def _log_recommendation_table(stage_name: str, trades: list):
    """Helper function to log the state of trades at each funnel step in a detailed table."""
    logger.info(f"\n--- {stage_name} ({len(trades)} candidates) ---")
    if not trades:
        logger.info("  No candidates to display.")
        return

    # Sort by confidence for consistent logging
    sorted_trades = sorted(trades, key=lambda x: x.get('confidence', 0), reverse=True)
    
    headers = ["Market", "Signal", "Strategy", "Exp. (6H)", "Conf.", "H+1", "H+2", "H+3", "H+4", "H+5", "H+6", "Status"]
    
    # Dynamically calculate column widths
    col_widths = {h: len(h) for h in headers}
    for trade in sorted_trades:
        col_widths["Market"] = max(col_widths["Market"], len(trade.get('market', '')))
        col_widths["Strategy"] = max(col_widths["Strategy"], len(trade.get('strategy', '')))
        col_widths["Signal"] = max(col_widths["Signal"], len(trade.get('signal', '')))
    
    # Set fixed widths for numeric/status columns
    col_widths["Market"] = max(col_widths["Market"], 6) + 2
    col_widths["Signal"] = max(col_widths["Signal"], 6) + 2
    col_widths["Strategy"] = max(col_widths["Strategy"], 8) + 2
    col_widths["Exp. (6H)"] = 11
    col_widths["Conf."] = 8
    for i in range(1, 7):
        col_widths[f"H+{i}"] = 8
    col_widths["Status"] = 45

    # Print Header
    header_line = " | ".join([f"{h:<{col_widths[h]}}" for h in headers])
    separator = "-+-".join(["-" * col_widths[h] for h in headers])
    logger.info(header_line)
    logger.info(separator)

    # Print Rows
    for trade in sorted_trades:
        status_color = "\033[92m" if trade['status'] == 'Recommended' else "\033[91m" if 'Failed' in trade['status'] else "\033[0m"
        
        pattern_cols = " | ".join([f"{p:>+7.2%}" for p in trade['pattern']])

        row_str = f"{trade['market']:<{col_widths['Market']}} | " \
                  f"{trade.get('signal', 'N/A'):<{col_widths['Signal']}} | " \
                  f"{trade['strategy']:<{col_widths['Strategy']}} | " \
                  f"{trade['expected_return']:>+10.2%} | " \
                  f"{trade['confidence']:>7.2%} | " \
                  f"{pattern_cols} | " \
                  f"{status_color}{trade['status']:<{col_widths['Status']}}\033[0m"
        logger.info(row_str)


def run(predictions: list, historical_data: pd.DataFrame = None, mode: str = 'live'):
    """
    Analyzes predictions through a multi-stage filtering funnel and presents
    a clear, visual table of the entire process.
    """
    logger.info("=== Starting Recommendation Generation Funnel ===")
    
    if not predictions:
        logger.warning("Recommender received no predictions to analyze.")
        return []

    # --- [Step 1] Initial Predictions ---
    funnel_data = []
    for pred in predictions:
        current_price = pred.get('current_price')
        if current_price is None or current_price <= 0: continue
        pattern = pred['predicted_pattern']
        expected_return = float(np.prod(1 + pattern) - 1) # Calculate expected return here
        
        # Assign signal unconditionally at the start
        signal = 'Long' if expected_return > 0 else ('Short' if expected_return < 0 else 'Neutral')

        funnel_data.append({
            'market': pred['market'],
            'expected_return': expected_return,
            'confidence': 1 / (1 + pred['uncertainty']),
            'uncertainty': pred['uncertainty'],
            'current_price': current_price,
            'strategy': pred.get('strategy', 'trending'),
            'pattern': pattern,
            'status': 'Initial Candidate',
            'dtw_distance': 999.0,
            'signal': signal # Assign signal here
        })

    _log_recommendation_table("[Funnel Step 1] Initial Predictions", funnel_data)

    # --- [Step 2] Liquidity Filter ---
    all_markets = [p['market'] for p in funnel_data]
    current_time = datetime.now(timezone.utc)
    trading_values = get_trading_values_for_markets(all_markets, end_time=current_time, hours=24)
    threshold = config.LIQUIDITY_THRESHOLDS.get(mode, config.LIQUIDITY_THRESHOLDS['live'])
    
    for trade in funnel_data:
        if trade['status'] == 'Initial Candidate': # Only process candidates that passed previous filters
            market_value = trading_values.get(trade['market'], 0)
            if market_value < threshold:
                trade['status'] = f"Failed: Low Liquidity"

    _log_recommendation_table(f"[Funnel Step 2] After Liquidity Filter", [t for t in funnel_data if t['status'] == 'Initial Candidate' or 'Failed: Low Liquidity' in t['status']])

    # --- [Step 3] Minimum Expected Return Filtering ---
    min_signal_return = getattr(config, 'MIN_SIGNAL_RETURN', 0.02)
    for trade in funnel_data:
        if trade['status'] == 'Initial Candidate': # Still an active candidate
            # Signal is already assigned, just check status
            if abs(trade['expected_return']) < min_signal_return:
                trade['status'] = f"Failed: Low Return"
    
    _log_recommendation_table(f"[Funnel Step 3] After Expected Return Filter", [t for t in funnel_data if t['status'] == 'Initial Candidate' or 'Failed: Low Return' in t['status']])

    # --- [Step 4] Uncertainty Filtering ---
    uncertainty_threshold = config.UNCERTAINTY_THRESHOLD
    for trade in funnel_data:
        if trade['status'] == 'Initial Candidate': # Still an active candidate
            if trade['uncertainty'] > uncertainty_threshold:
                trade['status'] = f"Failed: High Uncertainty"

    _log_recommendation_table(f"[Funnel Step 4] After Uncertainty Filter", [t for t in funnel_data if t['status'] == 'Initial Candidate' or 'Failed: High Uncertainty' in t['status']])

    # --- [Step 5] DTW Pattern Filtering ---
    success_patterns = get_historical_success_patterns()
    if success_patterns.any():
        for trade in funnel_data:
            if trade['status'] == 'Initial Candidate':
                min_dist = min([get_pattern_similarity(trade['pattern'], sp) for sp in success_patterns])
                trade['dtw_distance'] = min_dist
                if min_dist >= config.DTW_THRESHOLD:
                    trade['status'] = f"Failed: Low Similarity"
    else:
        logger.warning("No success patterns loaded, skipping DTW filter.")

    _log_recommendation_table(f"[Funnel Step 5] After DTW Filter", [t for t in funnel_data if t['status'] == 'Initial Candidate' or 'Failed: Low Similarity' in t['status']])
    
    # --- Final Status Update & Logging ---
    final_recommendations = []
    for trade in funnel_data:
        if trade['status'] == 'Initial Candidate':
            trade['status'] = 'Recommended'
            final_recommendations.append(trade)

    if final_recommendations:
        _log_recommendation_table("Final Recommendations", final_recommendations)
        
        # --- Save to CSV ---
        df = pd.DataFrame(final_recommendations)
        df['pattern'] = df['pattern'].apply(lambda p: ','.join([f'{x:+.4f}' for x in p])) # Save pattern as string
        
        cols_to_save = ['market', 'signal', 'strategy', 'expected_return', 'confidence', 'dtw_distance', 'current_price', 'pattern']
        df = df[[c for c in cols_to_save if c in df.columns]]

        output_dir = 'recommendations'
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"recs_{timestamp}.csv")
        try:
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            logger.info(f"Recommendations successfully saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save recommendations to CSV: {e}")
    else:
        logger.warning("No recommendations remained after all filtering stages.")

    logger.info("======================================================")
    return final_recommendations
