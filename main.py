import argparse
from utils.logger import logger
from utils.config import config
from data import database, collector, preprocessor
from data.database import get_data_period
from training import trainer
from inference import predictor, recommender
from utils import screener
from datetime import datetime, timedelta
import pandas as pd
import time
import os
import numpy as np

def display_pump_candidates(potential_pumps):
    """Helper function to print pump candidates in a structured format."""
    if not potential_pumps:
        logger.info("--- No Potential Pump Candidates Found ---")
        return

    logger.info("--- 🚀 급등 가능성 포착 결과 ---")
    for pump in potential_pumps:
        market = pump['market']
        current_price = pump['current_price']
        target_price = current_price * 1.10
        probs = pump['probabilities']
        total_pump_prob = pump['total_pump_prob']

        logger.info(f"▶ {market}")
        logger.info(f"  - 현재가: {current_price:,.0f}원 | 10% 상승 목표가: {target_price:,.0f}원")
        logger.info(f"  - 급등 확률 (총합): {total_pump_prob:.2%}")
        logger.info(f"  - 분포: [10-15%]: {probs[1]:.2%} | [15-20%]: {probs[2]:.2%} | [20%+]: {probs[3]:.2%}")
    logger.info("------------------------------------")

def save_pump_predictions_to_csv(pump_predictions: list):
    """Saves pump prediction results to a CSV file."""
    if not pump_predictions:
        logger.info("No pump predictions to save to CSV.")
        return

    df_to_save = []
    for pump in pump_predictions:
        row = {
            'market': pump['market'],
            'current_price': pump['current_price'],
            'target_price_10_pct_up': pump['current_price'] * 1.10,
            'total_pump_probability': pump['total_pump_prob'],
            'prob_0_10_pct': pump['probabilities'][0], # No pump
            'prob_10_15_pct': pump['probabilities'][1],
            'prob_15_20_pct': pump['probabilities'][2],
            'prob_20_plus_pct': pump['probabilities'][3]
        }
        df_to_save.append(row)

    df = pd.DataFrame(df_to_save)
    output_dir = 'predictions'
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"pump_preds_{timestamp}.csv")
    
    try:
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        logger.info(f"Pump predictions successfully saved to {filename}")
    except Exception as e:
        logger.error(f"Failed to save pump predictions to CSV: {e}")

def find_pattern_followers(leader_market, all_markets):
    """Finds coins that currently exhibit a pattern similar to the leader's past patterns."""
    now = datetime.now()
    candidates = []
    logger.info(f"Searching for followers based on {leader_market}'s past setup patterns...")

    for lag_days in [1, 2, 3]:
        lag_hours = lag_days * 24
        leader_past_time = now - timedelta(hours=lag_hours)
        leader_pattern = preprocessor.get_recent_pattern(leader_market, leader_past_time, hours=config.PATTERN_LOOKBACK_HOURS)
        
        if len(leader_pattern) != config.PATTERN_LOOKBACK_HOURS: continue

        for other_market in all_markets:
            if other_market == leader_market: continue
            follower_pattern = preprocessor.get_recent_pattern(other_market, now, hours=config.PATTERN_LOOKBACK_HOURS)
            if len(follower_pattern) != config.PATTERN_LOOKBACK_HOURS: continue

            window_size = int(config.PATTERN_LOOKBACK_HOURS * 0.15)
            similarity = predictor.get_pattern_similarity(leader_pattern, follower_pattern)
            candidates.append({'market': other_market, 'similarity': similarity, 'lag_days': lag_days, 'interpretation': f"Matches {leader_market}'s pattern from {lag_days} day(s) ago."})
    
    if not candidates: return []

    all_distances = [c['similarity'] for c in candidates]
    logger.info(f"DTW stats: min={np.min(all_distances):.3f}, p25={np.percentile(all_distances,25):.3f}, median={np.median(all_distances):.3f}, p75={np.percentile(all_distances,75):.3f}, max={np.max(all_distances):.3f}")

    threshold = np.median(all_distances)
    logger.info(f"Applying DTW similarity threshold: <= {threshold:.4f}")
    candidates = [c for c in candidates if c['similarity'] <= threshold]
    
    if not candidates: return []

    candidates.sort(key=lambda x: x['similarity'])
    top_candidates = []
    seen_markets = set()
    for cand in candidates:
        if cand['market'] not in seen_markets:
            top_candidates.append(cand)
            seen_markets.add(cand['market'])
        if len(top_candidates) >= 5: break

    return top_candidates

def main():
    parser = argparse.ArgumentParser(description="Crypto Predictor CLI v3")
    parser.add_argument(
        '--mode',
        choices=['init_db', 'collect-all', 'train', 'daily', 'screen', 'quick-recommend', 'backtest', 'train-pump', 'find-pumps', 'explain'],
        required=True,
        help="The mode to run the script in."
    )
    parser.add_argument('--days', type=int, default=30, help="Number of days for data collection or backtesting.")
    parser.add_argument('--symbol', type=str, help="A specific crypto symbol to predict (e.g., KRW-BTC).")
    parser.add_argument('--tune', action='store_true', help="Enable hyperparameter tuning during training.")
    
    # Arguments with defaults from config
    parser.add_argument('--model_path', type=str, default=config.MODEL_PATH, help="Path to the model file for analysis.")
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE_G, help="Override learning rate for training.")
    parser.add_argument('--epochs', type=int, default=config.EPOCHS, help="Override number of epochs for training.")
    parser.add_argument('--d_model', type=int, default=config.D_MODEL, help="Override d_model for Transformer/GAN.")
    parser.add_argument('--n_layers', type=int, default=config.N_LAYERS, help="Override n_layers for Transformer.")
    parser.add_argument('--n_heads', type=int, default=config.N_HEADS, help="Override n_heads for Transformer.")
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE, help="Override batch size for training.")

    args = parser.parse_args()
    logger.info(f"--- Running in {args.mode.upper()} mode ---")

    # Update config with any arguments passed
    config.MODEL_PATH = args.model_path
    config.LEARNING_RATE = args.lr
    config.EPOCHS = args.epochs
    config.D_MODEL = args.d_model
    config.N_LAYERS = args.n_layers
    config.N_HEADS = args.n_heads
    config.BATCH_SIZE = args.batch_size

    if args.mode == 'init_db':
        database.init_db()
    
    elif args.mode == 'collect-all':
        from data import collector
        collector.run_all(days=args.days)

    elif args.mode == 'train':
        from data import collector
        logger.info("Starting full model training...")
        days_available = get_data_period()
        training_days = min(max(days_available, 90), 365)
        logger.info(f"Data available for {days_available} days. Training will use data from the last {training_days} days.")
        for market in config.TARGET_MARKETS:
             collector.collect_market_data(market, days=training_days)
             time.sleep(0.5)
        trainer.run(tune=args.tune)

    elif args.mode == 'train-pump':
        from training import pump_trainer
        pump_trainer.run(tune=args.tune)

    elif args.mode == 'find-pumps':
        from inference import pump_predictor
        potential_pumps = pump_predictor.run()
        display_pump_candidates(potential_pumps)
        save_pump_predictions_to_csv(potential_pumps)

    elif args.mode == 'backtest':
        from training import evaluator
        evaluator.run(days_to_backtest=args.days)

    elif args.mode == 'daily':
        from data import collector
        from training import pump_trainer
        from inference import pump_predictor
        logger.info("Starting daily run (including model reinforcement)...")
        trending_markets = screener.get_trending_markets(mode='live')
        if trending_markets:
            logger.info(f"Collecting latest data for {len(trending_markets)} trending markets...")
            for market in trending_markets:
                collector.collect_market_data(market, days=30)
                time.sleep(0.5)
            logger.info("Fine-tuning main trend model...")
            trainer.run(markets=trending_markets)
            logger.info("Fine-tuning pump prediction model...")
            pump_trainer.run()
            logger.info("--- Running Main Trend Prediction Module ---")
            all_krw_markets_df = database.load_data("SELECT DISTINCT market FROM crypto_data WHERE market LIKE 'KRW-%'")
            all_krw_markets = all_krw_markets_df['market'].tolist() if not all_krw_markets_df.empty else []
            initial_predictions = predictor.run(markets=trending_markets)
            pattern_follower_predictions = []
            if trending_markets and initial_predictions:
                leader_market = trending_markets[0]
                other_markets = [m for m in all_krw_markets if m not in trending_markets]
                top_pattern_followers = find_pattern_followers(leader_market, other_markets)
                logger.info(f"Found {len(top_pattern_followers)} pattern-following candidates:")
                for cand in top_pattern_followers:
                    logger.info(f"  - {cand['market']} (Similarity: {cand['similarity']:.4f}, {cand['interpretation']})")
                if top_pattern_followers:
                    follower_markets = [c['market'] for c in top_pattern_followers]
                    logger.info(f"Collecting latest data for {len(follower_markets)} pattern-following coins...")
                    for market in follower_markets:
                        collector.collect_market_data(market, days=30)
                        time.sleep(0.5)
                    logger.info("Making predictions for pattern-following coins...")
                    pattern_follower_predictions = predictor.run(markets=follower_markets)
            for pred in initial_predictions: pred['strategy'] = 'trending'
            for pred in pattern_follower_predictions: pred['strategy'] = 'pattern'
            all_predictions = initial_predictions + pattern_follower_predictions
            recommender.run(predictions=all_predictions, mode='live')
            potential_pumps = pump_predictor.run()
            display_pump_candidates(potential_pumps)
            save_pump_predictions_to_csv(potential_pumps)
        else:
            logger.info("No trending markets found today.")
        logger.info("Daily run finished.")

    elif args.mode == 'quick-recommend':
        from data import collector
        logger.info("Starting quick recommendation run (no training)...")
        trending_markets = screener.get_trending_markets(mode='live')
        if trending_markets:
            logger.info(f"Collecting latest data for {len(trending_markets)} trending markets...")
            for market in trending_markets:
                collector.collect_market_data(market, days=30)
                time.sleep(0.5)
            logger.info("Making predictions with the existing model...")
            predictions = predictor.run(markets=trending_markets)
            for pred in predictions: pred['strategy'] = 'trending'
            recommender.run(predictions=predictions, mode='live')
        else:
            logger.info("No trending markets found today.")
        logger.info("Quick recommendation run finished.")

    elif args.mode == 'screen':
        screener.get_trending_markets(mode='live')

if __name__ == "__main__":
    main()