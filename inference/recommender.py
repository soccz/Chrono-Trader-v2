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

def run(predictions: list, historical_data: pd.DataFrame = None, mode: str = 'live'):
    """Analyzes predictions and uncertainty to generate user-friendly recommendations."""
    logger.info("=== Starting Recommendation Generation (Ensembled + User-Friendly) ===")
    
    if not predictions:
        logger.warning("Recommender received no predictions to analyze.")
        return []

    if historical_data is not None:
        if not isinstance(historical_data.index, pd.DatetimeIndex):
            historical_data = historical_data.copy()
            historical_data.index = pd.to_datetime(historical_data.index, utc=True, errors='coerce')
        current_time = historical_data.index.max()
        if pd.isna(current_time):
            logger.warning("Historical data index did not yield a valid timestamp; falling back to current UTC time.")
            current_time = datetime.now(timezone.utc)
    else:
        current_time = datetime.now(timezone.utc)

    # Load historical success patterns for DTW comparison
    success_patterns = get_historical_success_patterns()

    logger.info(f"[Funnel] Step 2 (Predictor): Received {len(predictions)} initial predictions.")

    # --- Liquidity Filter (Q8) ---
    all_markets = list(set(p['market'] for p in predictions))
    trading_values = get_trading_values_for_markets(all_markets, end_time=current_time, hours=24)
    
    threshold = config.LIQUIDITY_THRESHOLDS.get(mode, config.LIQUIDITY_THRESHOLDS['live'])
    logger.info(f"Applying liquidity filter (mode: {mode}, threshold: {threshold:,.0f} KRW)")

    initial_pred_count = len(predictions)
    liquid_predictions = []
    for pred in predictions:
        market_value = trading_values.get(pred['market'], 0)
        if market_value >= threshold:
            liquid_predictions.append(pred)
        else:
            logger.info(f"[Liquidity Filter] Filtering out {pred['market']} (24h value: {market_value:,.0f} KRW < {threshold:,.0f} KRW)")
    
    if len(liquid_predictions) < initial_pred_count:
        logger.info(f"Filtered out {initial_pred_count - len(liquid_predictions)} predictions due to low liquidity.")
    
    logger.info(f"[Funnel] Step 3 (Liquidity Filter): {len(liquid_predictions)} predictions remaining.")
    
    predictions = liquid_predictions
    if not predictions:
        logger.warning("No liquid predictions remaining after filtering.")
        return []
    # --- End Liquidity Filter ---

    potential_trades = []
    for pred in predictions:
        current_price = pred.get('current_price')
        if current_price is None or current_price <= 0:
            logger.info(f"Skipping {pred['market']} due to invalid current price: {current_price}")
            continue

        pattern = pred['predicted_pattern']
        total_change_sum = float(np.sum(pattern))
        compounded_return = float(np.prod(1 + pattern) - 1)
        confidence = 1 / (1 + pred['uncertainty'])

        potential_trades.append({
            'market': pred['market'],
            'potential': compounded_return,
            'expected_return': compounded_return,
            'raw_sum_return': total_change_sum,
            'pattern': pattern,
            'confidence': confidence,
            'uncertainty': pred['uncertainty'],
            'current_price': current_price,
            'strategy': pred.get('strategy', 'trending')
        })

    # --- Minimum Expected Return Filtering ---
    min_signal_return = getattr(config, 'MIN_SIGNAL_RETURN', 0.0)
    return_filtered_trades = []
    dropped_for_return = 0
    for trade in potential_trades:
        expected_return = trade['expected_return']
        if expected_return >= min_signal_return:
            trade['signal'] = 'Long'
        elif expected_return <= -min_signal_return:
            trade['signal'] = 'Short'
        else:
            dropped_for_return += 1
            continue
        return_filtered_trades.append(trade)

    if dropped_for_return > 0:
        logger.info(f"Filtered out {dropped_for_return} candidates due to low expected return (< {min_signal_return:.2%}).")

    potential_trades = return_filtered_trades
    if not potential_trades:
        logger.warning("No trades remained after expected return filtering.")
        return []
    # --- End Minimum Expected Return Filtering ---

    # --- Uncertainty Filtering ---
    initial_count = len(potential_trades)
    uncertainty_threshold = config.UNCERTAINTY_THRESHOLD
    filtered_trades = [t for t in potential_trades if t['uncertainty'] <= uncertainty_threshold]
    if len(filtered_trades) < initial_count:
        logger.info(f"Filtered out {initial_count - len(filtered_trades)} recommendations due to high uncertainty (> {uncertainty_threshold}).")
    
    logger.info(f"[Funnel] Step 4 (Uncertainty Filter): {len(filtered_trades)} predictions remaining.")
    # --------------------------

    # --- DTW Pattern Filtering (Q3) ---
    initial_count = len(filtered_trades)
    dtw_filtered_trades = []
    if success_patterns.any():
        for trade in filtered_trades:
            predicted_pattern = trade['pattern']
            min_dist = min([get_pattern_similarity(predicted_pattern, success_pattern) for success_pattern in success_patterns])
            trade['dtw_distance'] = min_dist
            if min_dist < config.DTW_THRESHOLD:
                dtw_filtered_trades.append(trade)
            else:
                logger.info(f"[DTW Filter] Filtering out {trade['market']} (DTW distance: {min_dist:.4f} >= {config.DTW_THRESHOLD})")
    else:
        logger.warning("No success patterns loaded, skipping DTW filter.")
        dtw_filtered_trades = filtered_trades

    if len(dtw_filtered_trades) < initial_count:
        logger.info(f"Filtered out {initial_count - len(dtw_filtered_trades)} recommendations due to low DTW similarity.")

    logger.info(f"[Funnel] Step 5 (DTW Filter): {len(dtw_filtered_trades)} predictions remaining.")
    # --- End DTW Filter ---

    trending_trades = [t for t in dtw_filtered_trades if t['strategy'] == 'trending']
    pattern_trades = [t for t in dtw_filtered_trades if t['strategy'] == 'pattern']

    trending_trades.sort(key=lambda x: x['confidence'], reverse=True)
    top_trending = trending_trades[:5]

    pattern_trades.sort(key=lambda x: x['confidence'], reverse=True)

    top_trades = top_trending + pattern_trades
    top_trades.sort(key=lambda x: (x['confidence'], abs(x['potential'])), reverse=True)

    df_to_save = []
    logger.info(f"\n=== 상세 암호화폐 거래 추천 [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ===")
    logger.info(f"분석된 유망 코인 수: {len(predictions)}개 (유동성 필터링 후)")
    logger.info(f"최종 추천 개수: {len(top_trades)}개 (불확실성/DTW 필터링 적용)")

    for i, trade in enumerate(top_trades):
        signal_type = trade.get('signal')
        if not signal_type:
            signal_type = "Long" if trade['potential'] > 0 else "Short"
        signal = f"{signal_type} (매수)" if signal_type == "Long" else f"{signal_type} (매도)"
        
        live_price = get_current_price(trade['market'])
        entry_price = live_price if live_price is not None else trade['current_price']
        if entry_price is None or entry_price <= 0:
            entry_price = trade['current_price']

        trade_data = trade.copy()
        trade_data['signal'] = signal_type
        trade_data['current_price'] = entry_price
        trade_data['predicted_pattern'] = trade['pattern']
        df_to_save.append(trade_data)

        logger.info(f"\n--- {i+1}. {trade['market']} ({trade['strategy'].upper()}) ---")
        logger.info(f"*   추천 신호: {signal}")
        if entry_price < 1:
            logger.info(f"*   현재 가격: {entry_price:,.4f}원")
        else:
            logger.info(f"*   현재 가격: {entry_price:,.0f}원")
        logger.info(f"*   신뢰도: {trade['confidence']:.2%}")
        if 'dtw_distance' in trade:
            logger.info(f"*   유사도(DTW): {trade['dtw_distance']:.4f}")
        logger.info("*   [시스템 출력 1] 시간별 예상 등락률:")
        for hour, p_val in enumerate(trade['pattern']):
            logger.info(f"    *   {hour+1}시간 후: {p_val:+.2%}")
        logger.info("*   [시스템 출력 2] 종합 예상 수익률 (6시간 합산):")
        logger.info(f"    *   {trade['potential']:.2%}")

    if df_to_save:
        df = pd.DataFrame(df_to_save)
        if 'pattern' in df.columns:
            df['pattern'] = df['pattern'].apply(lambda p: ','.join([f'{x:.4f}' for x in p]))
        
        # Define columns to save
        cols_to_save = ['market', 'signal', 'strategy', 'expected_return', 'potential', 'confidence', 'dtw_distance', 'current_price', 'pattern']
        # Filter columns that actually exist in the dataframe
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

    logger.info("======================================================")
    return df_to_save
