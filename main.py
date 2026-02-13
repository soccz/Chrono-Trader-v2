# -*- coding: utf-8 -*-
import sys
import os
# Ensure the project root is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
from utils.logger import logger
from utils.config import config
from data import database, collector, preprocessor
from data.database import get_data_period
from training import trainer
from inference import predictor, recommender
from utils import screener, research_reporter
from datetime import datetime, timedelta, timezone
import pandas as pd
import time
import os
import numpy as np

def display_pump_candidates(potential_pumps):
    """Helper function to print pump candidates in a structured format."""
    if not potential_pumps:
        logger.info("--- No Potential Pump Candidates Found ---")
        return

    logger.info("--- [Potential Pumps] ---")
    for pump in potential_pumps:
        market = pump['market']
        current_price = pump['current_price']
        target_price = current_price * 1.10
        probs = pump['probabilities']
        total_pump_prob = pump['total_pump_prob']

        logger.info(f"-> {market}")
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
        choices=[
            'init_db', 'collect-all', 'train', 'daily', 'continuous', 'screen',
            'refresh-db',
            'quick-recommend', 'intraday', 'morning-report',
            'backtest', 'train-pump', 'find-pumps', 'explain'
        ],
        required=True,
        help="The mode to run the script in."
    )
    parser.add_argument('--days', type=int, default=30, help="Number of days for data collection or backtesting.")
    parser.add_argument('--symbol', type=str, help="A specific crypto symbol to predict (e.g., KRW-BTC).")
    parser.add_argument('--tune', action='store_true', help="Enable hyperparameter tuning during training.")
    parser.add_argument(
        '--no_collect',
        action='store_true',
        help="For --mode train: skip network data collection and train using existing DB only."
    )
    parser.add_argument('--daily_epochs', type=int, default=2, help="Number of epochs for daily fine-tuning.")
    parser.add_argument('--limit', type=int, default=5, help="Number of markets to screen for inference-only modes.")
    parser.add_argument('--lookback_days', type=int, default=1, help="Screening lookback days for inference-only modes.")
    parser.add_argument('--min_k', type=int, default=3, help="Target number of recommendations (min_k) for inference-only modes.")
    parser.add_argument('--send_telegram', action='store_true', help="Force sending Telegram (overrides defaults).")
    parser.add_argument('--no_telegram', action='store_true', help="Disable Telegram sending for this run.")
    parser.add_argument('--refresh_data', action='store_true', help="Refresh recent data via collector before inference (requires network).")
    parser.add_argument('--refresh_days', type=int, default=2, help="Days of recent data to refresh when --refresh_data is set.")
    parser.add_argument(
        '--no_auto_refresh',
        action='store_true',
        help="Disable automatic pre-refresh for scheduled inference modes (intraday/morning-report)."
    )
    parser.add_argument(
        '--refresh_top_n',
        type=int,
        default=20,
        help="Auto-refresh: also refresh top N markets by 24h trading value from DB."
    )
    parser.add_argument(
        '--refresh_tv_hours',
        type=int,
        default=24,
        help="Auto-refresh: trading value lookback window (hours) used to pick top markets from DB."
    )
    parser.add_argument(
        '--market_budget',
        type=int,
        default=0,
        help="Scheduled inference market budget (0 = auto based on device/mode)."
    )
    parser.add_argument(
        '--rotation_keep',
        type=float,
        default=0.7,
        help="Scheduled inference: keep this fraction of previous run_markets (stability)."
    )
    parser.add_argument(
        '--timeout_sec',
        type=int,
        default=0,
        help="Watchdog timeout (seconds) for scheduled modes (0 = mode default, negative = disable)."
    )
    parser.add_argument(
        '--offline_ok',
        action='store_true',
        help="For refresh-db: if offline, skip refresh and exit 0 (optionally alert)."
    )
    parser.add_argument(
        '--enable_continuous',
        action='store_true',
        help="Allow running --mode continuous (deprecated)."
    )
    parser.add_argument(
        '--allow_stale_data',
        action='store_true',
        help="Allow scheduled inference to proceed even if DB freshness gate fails (NOT recommended)."
    )
    parser.add_argument(
        '--skip_aux',
        action='store_true',
        help="Skip auxiliary sections (morning-report pattern followers + pump radar) for faster, more stable ops runs."
    )
    parser.add_argument(
        '--skip_pattern_followers',
        action='store_true',
        help="(morning-report) Skip pattern followers section (DTW-heavy) for faster runs."
    )
    parser.add_argument(
        '--skip_pump_radar',
        action='store_true',
        help="(morning-report) Skip pump radar section."
    )
    
    # Arguments with defaults from config
    parser.add_argument('--model_path', type=str, default=config.General.MODEL_PATH, help="Path to the model file for analysis.")
    parser.add_argument('--lr', type=float, default=config.Gan.LEARNING_RATE_G, help="Override learning rate for training.")
    parser.add_argument('--epochs', type=int, default=config.Gan.EPOCHS, help="Override number of epochs for training.")
    parser.add_argument('--d_model', type=int, default=config.Gan.D_MODEL, help="Override d_model for Transformer/GAN.")
    parser.add_argument('--n_layers', type=int, default=config.Gan.N_LAYERS, help="Override n_layers for Transformer.")
    parser.add_argument('--n_heads', type=int, default=config.Gan.N_HEADS, help="Override n_heads for Transformer.")
    parser.add_argument('--batch_size', type=int, default=config.Gan.BATCH_SIZE, help="Override batch size for training.")

    args = parser.parse_args()
    logger.info(f"--- Running in {args.mode.upper()} mode ---")

    # Update config with any arguments passed
    config.General.MODEL_PATH = args.model_path
    config.Gan.LEARNING_RATE_G = args.lr
    config.Gan.EPOCHS = args.epochs
    config.Gan.D_MODEL = args.d_model
    config.Gan.N_LAYERS = args.n_layers
    config.Gan.N_HEADS = args.n_heads
    config.Gan.BATCH_SIZE = args.batch_size

    if args.mode == 'init_db':
        database.init_db()
    
    elif args.mode == 'collect-all':
        # collector is already globally imported
        collector.run_all(days=args.days)

    elif args.mode == 'refresh-db':
        # Dedicated DB refresh job (separate from inference): refresh a diversified watchlist.
        logger.info("Starting DB refresh job...")
        from utils.run_lock import run_lock
        from utils.netcheck import can_resolve
        from utils.market_selection import select_markets_for_scheduled_run
        from utils.run_markets_cache import rotate_markets
        from utils.run_markets_metrics import record_run
        from utils.watchdog import time_limit

        send_tg = (not args.no_telegram)
        if args.send_telegram:
            send_tg = True

        with run_lock("refresh-db"):
            timeout = int(args.timeout_sec)
            if timeout == 0:
                timeout = 20 * 60
            if timeout < 0:
                timeout = 0

            with time_limit(timeout, name="refresh-db"):
                t0 = time.time()
                offline = not can_resolve("api.upbit.com")
                if offline:
                    logger.warning("Upbit host resolution failed; refresh-db cannot proceed (offline).")
                    if send_tg:
                        try:
                            from utils.telegram_bot import send_alert
                            send_alert("<b>⚠️ REFRESH-DB OFFLINE</b>\n\nDNS resolution failed for api.upbit.com.", bypass_dedup=True)
                        except Exception as e:
                            logger.error(f"Failed to send refresh-db offline alert: {e}")
                    if args.offline_ok:
                        logger.info("offline_ok enabled; skipping refresh-db and exiting 0.")
                        raise SystemExit(0)
                    raise SystemExit(2)

                # Seed from screener (fallbacks to DB if API fails)
                seed_markets = screener.get_trending_markets(mode='live', limit=args.limit, lookback_days=max(args.lookback_days, 1))
                budget = int(args.market_budget or 0)
                if budget <= 0:
                    budget = 32 if str(config.Device.DEVICE).startswith("cuda") else 24

                run_meta = {
                    "mode": "refresh-db",
                    "budget": budget,
                    "seed_n": len(seed_markets or []),
                }

                # For refresh, do not filter by freshness; stale is exactly what we want to refresh.
                refresh_markets, sel_meta = select_markets_for_scheduled_run(
                    mode="intraday",
                    seed_markets=seed_markets,
                    budget=budget,
                    tv_hours=int(args.refresh_tv_hours),
                    candidate_top=max(200, budget * 5),
                    lookback_hours=168,
                    exploit_target=int(args.refresh_top_n),
                    max_holdings=5,
                    max_core=10,
                    max_lag_h=None,
                    return_meta=True,
                )
                run_meta.update(sel_meta or {})

                try:
                    refresh_markets, rot_meta = rotate_markets(
                        mode="refresh-db",
                        new_markets=refresh_markets,
                        budget=budget,
                        keep_ratio=float(args.rotation_keep),
                        max_lag_h=None,
                    )
                    logger.info(f"[RefreshDB] rotation kept={rot_meta.get('kept')} added={rot_meta.get('added')} budget={budget}")
                    run_meta["rotation_kept"] = rot_meta.get("kept")
                    run_meta["rotation_added"] = rot_meta.get("added")
                except Exception as e:
                    logger.warning(f"[RefreshDB] rotation failed (proceeding): {e}")

                logger.info(f"Refreshing latest data for {len(refresh_markets)} markets (days={args.refresh_days})...")
                for market in refresh_markets:
                    collector.collect_market_data(market, days=args.refresh_days)
                    time.sleep(0.15)

                try:
                    run_meta["elapsed_sec"] = float(time.time() - t0)
                    record_run("refresh-db", refresh_markets, meta=run_meta, recs=[])
                except Exception as e:
                    logger.debug(f"[Metrics] refresh-db record_run failed: {e}")

                if send_tg:
                    try:
                        from utils.telegram_bot import send_alert
                        msg = (
                            "<b>✅ REFRESH-DB DONE</b>\n"
                            f"<code>markets={len(refresh_markets)}</code>\n"
                            f"<code>days={args.refresh_days}</code>\n"
                        )
                        send_alert(msg, bypass_dedup=True)
                    except Exception as e:
                        logger.error(f"Failed to send refresh-db done alert: {e}")

        logger.info("DB refresh job finished.")

    elif args.mode == 'train':
        # collector is already globally imported
        logger.info("Starting full model training...")
        days_available = get_data_period()
        # Compute-aware data horizon:
        # - GPU: allow deeper history for robustness
        # - CPU: keep upper bound conservative for practical runtime
        if str(config.Device.DEVICE).startswith("cuda"):
            min_days, max_days = 180, 720
        else:
            min_days, max_days = 120, 365

        training_days = min(max(days_available, min_days), max_days)
        logger.info(f"Data available for {days_available} days. Training will use data from the last {training_days} days.")
        # Use TRAIN_COINS for full training (top liquidity coins)
        target_markets = config.Data.TRAIN_COINS
        if args.no_collect:
            logger.warning(
                "--no_collect enabled: skipping network data collection. "
                "Training will rely on existing DB snapshots only."
            )
        else:
            for market in target_markets:
                try:
                    collector.collect_market_data(market, days=training_days)
                except Exception as e:
                    # Keep training resilient in offline/unstable network environments.
                    logger.error(f"collect_market_data failed for {market}: {e}")
                    if args.offline_ok:
                        logger.warning("--offline_ok enabled: continuing despite collection failure.")
                        continue
                    raise
                time.sleep(0.5)
        trainer.run(tune=args.tune, epochs=args.epochs)

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
        stride = int(os.getenv("AETHER_BACKTEST_STRIDE_HOURS", "1") or 1)
        summary_tag = str(os.getenv("AETHER_BACKTEST_SUMMARY_TAG", "main") or "main").strip() or "main"
        success = evaluator.run(days_to_backtest=args.days, stride_hours=stride, summary_tag=summary_tag)
        if not success:
            raise SystemExit(1)

    elif args.mode == 'daily':
        config.General.REC_TAG = "daily" # Set tag for daily recommendations
        # collector is globally imported
        from training import pump_trainer
        from inference import pump_predictor
        from utils.telegram_bot import send_alert, format_daily_report, format_short_term_report
        # --- [Portfolio Manager Integration] ---
        from utils.portfolio_manager import PortfolioManager
        from data.collector import get_current_price
        
        logger.info("Starting daily run (including model reinforcement)...")
        pm = PortfolioManager()
        
        # --- [Step 0] Performance Tracker & Portfolio Closing ---
        logger.info("=== PERFORMANCE TRACKER: Checking Yesterday's Recommendations ===")
        
        # 1. Portfolio Close Logic
        try:
            conn = pm._get_conn()
            cur = conn.cursor()
            cur.execute("SELECT DISTINCT market FROM trades WHERE status='OPEN' AND strategy='daily'")
            open_markets = [row[0] for row in cur.fetchall()]
            conn.close()
            
            current_prices_map = {}
            if open_markets:
                for m in open_markets:
                    p = get_current_price(m)
                    if p: current_prices_map[m] = p
                pm.close_open_trades('daily', current_prices_map)
                logger.info("Portfolio Manager: Closed previous Daily trades.")
        except Exception as e:
             logger.error(f"Portfolio Manager Error (Close): {e}")

        # [REAL Performance Tracker - Feeds into ModelPerformanceTracker]
        from utils.model_tracker import get_tracker
        perf_tracker = get_tracker(n_models=5)
        
        # Strategy → Model ID 매핑
        strategy_to_model = {
            'trending': 0,       # Model 1: Trend Following
            'mean_reversion': 1, # Model 2: Mean Reversion
            'continuous': 2,     # Model 3: Volatility Breakout
            'pattern': 3,        # Model 4: Pattern Recognition
            'daily': 4,          # Model 5: Market Neutral / General
        }
        
        try:
            import glob
            from data.collector import get_current_price
            
            # Find the most recent recommendation file (from yesterday or latest)
            rec_files = sorted(glob.glob(os.path.join("recommendations", "recs_*.csv")), reverse=True)
            if rec_files:
                latest_rec_file = rec_files[0]
                logger.info(f"Found previous recommendations: {os.path.basename(latest_rec_file)}")
                
                past_recs = pd.read_csv(latest_rec_file)
                if not past_recs.empty:
                    results = []
                    tracker_updates = []  # (model_id, was_correct)
                    
                    for idx, row in past_recs.iterrows():
                        market = row['market']
                        predicted_return = row['expected_return']
                        entry_price = row['current_price']
                        signal = row.get('signal', 'Unknown')
                        strategy = str(row.get('strategy', 'daily'))
                        
                        # Get current price
                        current_price = get_current_price(market)
                        if current_price and current_price > 0:
                            actual_return = (current_price - entry_price) / entry_price
                            
                            # For Short signals, invert the return
                            if signal == 'Short':
                                actual_return = -actual_return
                            
                            error = abs(predicted_return - actual_return)
                            direction_correct = (predicted_return * actual_return) > 0  # Same sign?
                            
                            results.append({
                                'market': market,
                                'signal': signal,
                                'predicted_%': predicted_return,
                                'actual_%': actual_return,
                                'error_%': error,
                                'direction_ok': direction_correct
                            })
                            
                            # === FEEDBACK LOOP: Feed into ModelPerformanceTracker ===
                            model_id = 4  # Default
                            for key, mid in strategy_to_model.items():
                                if key in strategy.lower() or key in os.path.basename(latest_rec_file).lower():
                                    model_id = mid
                                    break
                            tracker_updates.append((model_id, direction_correct))
                    
                    if results:
                        results_df = pd.DataFrame(results)
                        accuracy = results_df['direction_ok'].mean() * 100
                        avg_error = results_df['error_%'].mean() * 100
                        
                        logger.info(f"  PERFORMANCE REPORT:")
                        logger.info(f"   - Total Tested: {len(results)} recommendations")
                        logger.info(f"   - Direction Accuracy: {accuracy:.1f}%")
                        logger.info(f"   - Average Prediction Error: {avg_error:.2f}%")
                        
                        for res in results:
                            status = "[OK]" if res['direction_ok'] else "[X]"
                            logger.info(f"   {status} {res['market']} ({res['signal']}): Predicted {res['predicted_%']:+.2%}, Actual {res['actual_%']:+.2%}")
                        
                        # === SAVE to ModelPerformanceTracker ===
                        if tracker_updates:
                            perf_tracker.update_batch(tracker_updates)
                            perf_tracker.save()
                            weights = perf_tracker.get_weights()
                            logger.info(f"  📊 ModelPerformanceTracker 업데이트: {len(tracker_updates)}건 반영")
                            logger.info(f"  📊 새 가중치: {[f'{w:.3f}' for w in weights]}")
                else:
                    logger.info("No past recommendations to evaluate.")
            else:
                logger.info("No previous recommendation files found. (This is normal for first run.)")
        except Exception as e:
            logger.error(f"Performance tracking failed: {e}. Continuing with daily run.")
        
        logger.info("=== END PERFORMANCE REPORT ===")
        
        # --- [Step 1] Trending Strategy ---
        trending_markets = screener.get_trending_markets(mode='live')
        trending_recs = []
        
        if trending_markets:
            logger.info(f"Collecting latest data for {len(trending_markets)} trending markets...")
            for market in trending_markets:
                collector.collect_market_data(market, days=30)
                time.sleep(0.5)
            logger.info("Fine-tuning main trend model...")
            trainer.run(markets=trending_markets, epochs=args.daily_epochs)
            
            logger.info("Generating TRENDING predictions...")
            trend_preds = predictor.run(markets=trending_markets)
            for pred in trend_preds: pred['strategy'] = 'trending'
            
            # Filter and get Top 3
            trending_recs = recommender.run(predictions=trend_preds, mode='live', min_k=3)
        else:
            logger.info("No trending markets found today.")

        # --- [Step 2] Pattern Strategy ---
        # Pattern strategy depends on finding followers of the leader.
        # We need ALL KRW markets to find followers.
        pattern_recs = []
        
        if trending_markets:
            leader_market = trending_markets[0]
            logger.info(f"Leader Market for Pattern Matching: {leader_market}")
            
            all_krw_markets_df = database.load_data("SELECT DISTINCT market FROM crypto_data WHERE market LIKE 'KRW-%'")
            all_krw_markets = all_krw_markets_df['market'].tolist() if not all_krw_markets_df.empty else []
            other_markets = [m for m in all_krw_markets if m not in trending_markets] # Exclude already processed trending ones
            
            top_pattern_followers = find_pattern_followers(leader_market, other_markets)
            
            if top_pattern_followers:
                follower_markets = [c['market'] for c in top_pattern_followers]
                logger.info(f"Collecting latest data for {len(follower_markets)} pattern-following coins...")
                for market in follower_markets:
                    collector.collect_market_data(market, days=30)
                    time.sleep(0.5)
                
                logger.info("Generating PATTERN predictions...")
                pattern_preds = predictor.run(markets=follower_markets)
                for pred in pattern_preds: pred['strategy'] = 'pattern'
                
                # Filter and get Top 3
                pattern_recs = recommender.run(predictions=pattern_preds, mode='live', min_k=3)
        
        # --- [Step 3] Pump Strategy ---
        logger.info("Fine-tuning pump prediction model...")
        pump_trainer.run()
        potential_pumps = pump_predictor.run()
        display_pump_candidates(potential_pumps)
        save_pump_predictions_to_csv(potential_pumps)

        # --- [Step 4] Portfolio Update & Reporting ---
        all_final_recs = trending_recs + pattern_recs
        
        # [Portfolio Manager] Record new trades
        if all_final_recs:
            logger.info(f"Portfolio Manager: Opening {len(all_final_recs)} new 'Daily' trades.")
            for rec in all_final_recs:
                pm.add_trade(
                    market=rec['market'],
                    strategy='daily',
                    signal=rec['signal'],
                    entry_price=rec['current_price']
                )

        # [Telegram Report]
        try:
            report_msg = format_daily_report(trending_recs, pattern_recs, potential_pumps)
            send_alert(report_msg, bypass_dedup=True)
        except Exception as e:
            logger.error(f"Failed to send daily telegram report: {e}")

        # [Step 6] Generate Research Report
        logger.info("Generating Daily Research Report...")
        try: research_reporter.run()
        except Exception as e: logger.error(f"Failed to generate research report: {e}")

        logger.info("Daily run finished.")

    elif args.mode == 'quick-recommend':
        # collector is already globally imported
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

    elif args.mode == 'intraday':
        # Inference-only 4H runner (cron-friendly): no training, uses existing ensemble weights.
        # Recommended schedule: every 4 hours (00/04/08/12/16/20).
        config.General.REC_TAG = "intraday"
        logger.info("Starting intraday inference run (no training)...")

        from utils.run_lock import run_lock
        from utils.freshness import get_db_latest_and_lag_hours

        send_tg = (not args.no_telegram)
        if args.send_telegram:
            send_tg = True

        with run_lock("intraday"):
            from utils.watchdog import time_limit
            timeout = int(args.timeout_sec)
            if timeout == 0:
                timeout = 20 * 60
            if timeout < 0:
                timeout = 0
            def _intraday_body():
                t0 = time.time()
                seed_markets = screener.get_trending_markets(mode='live', limit=args.limit, lookback_days=args.lookback_days)
                seed_fallback_index = False
                if not seed_markets:
                    seed_markets = list(getattr(config.Data, "MARKET_INDEX_COINS", [])) or ["KRW-BTC", "KRW-ETH"]
                    seed_fallback_index = True
                    logger.warning(
                        f"No markets found for intraday screening; falling back to index coins: {seed_markets}"
                    )

                # Auto-refresh: keep DB reasonably up-to-date for (a) screened markets and (b) top liquidity markets.
                # Also expand inference universe using DB-based selection for diversification.
                from utils.market_selection import select_markets_for_scheduled_run
                from utils.netcheck import can_resolve

                # Compute-aware default budget
                budget = int(args.market_budget or 0)
                if budget <= 0:
                    budget = 32 if str(config.Device.DEVICE).startswith("cuda") else 24

                max_lag_sel = float(getattr(config.Recommender, "DATA_FRESHNESS_MAX_LAG_HOURS_INTRADAY", 6))
                run_meta = {
                    "mode": "intraday",
                    "budget": budget,
                    "seed_n": len(seed_markets or []),
                    "seed_fallback_index": bool(seed_fallback_index),
                }

                run_markets, sel_meta = select_markets_for_scheduled_run(
                    mode="intraday",
                    seed_markets=seed_markets,
                    budget=budget,
                    tv_hours=int(args.refresh_tv_hours),
                    candidate_top=max(200, budget * 5),
                    lookback_hours=168,
                    exploit_target=int(args.refresh_top_n),
                    max_holdings=5,
                    max_core=10,
                    max_lag_h=max_lag_sel,
                    return_meta=True,
                )
                run_meta.update(sel_meta or {})

                try:
                    from utils.run_markets_cache import rotate_markets
                    rot_meta = {}
                    run_markets, rot_meta = rotate_markets(
                        mode="intraday",
                        new_markets=run_markets,
                        budget=budget,
                        keep_ratio=float(args.rotation_keep),
                        max_lag_h=max_lag_sel,
                    )
                    logger.info(f"[RunMarkets] rotation kept={rot_meta.get('kept')} added={rot_meta.get('added')} budget={budget}")
                    run_meta["rotation_kept"] = rot_meta.get("kept")
                    run_meta["rotation_added"] = rot_meta.get("added")
                except Exception as e:
                    logger.warning(f"[RunMarkets] rotation failed (proceeding): {e}")

                auto_refresh_enabled = (not args.no_auto_refresh) or args.refresh_data
                if auto_refresh_enabled:
                    try:
                        offline = not can_resolve("api.upbit.com")
                        run_meta["auto_refresh_enabled"] = True
                        run_meta["auto_refresh_skipped_offline"] = bool(offline)
                        if not offline:
                            refresh_markets = run_markets
                            logger.info(
                                f"Refreshing latest data for {len(refresh_markets)} markets (days={args.refresh_days})..."
                            )
                            for market in refresh_markets:
                                collector.collect_market_data(market, days=args.refresh_days)
                                time.sleep(0.15)
                        else:
                            logger.warning("Upbit host resolution failed; skipping auto-refresh (offline mode).")
                    except Exception as e:
                        logger.warning(f"Auto-refresh failed (proceeding): {e}")
                else:
                    run_meta["auto_refresh_enabled"] = False

                # Freshness gate: scheduled inference should not run on stale DB snapshots.
                run_meta["run_markets_selected"] = len(run_markets)
                try:
                    max_lag_h = float(getattr(config.Recommender, "DATA_FRESHNESS_MAX_LAG_HOURS_INTRADAY", 6))
                    fail_on_stale = bool(getattr(config.Recommender, "FAIL_ON_STALE_DATA_LIVE", True))
                    if fail_on_stale and (not args.allow_stale_data):
                        from data.database import get_latest_db_timestamps_by_market
                        now_utc = datetime.now(timezone.utc)
                        ts_map = get_latest_db_timestamps_by_market(run_markets)
                        kept = []
                        dropped = []
                        kept_lags = []
                        for m in run_markets:
                            ts = ts_map.get(m)
                            if ts is None:
                                dropped.append((m, None))
                                continue
                            lag_h = (now_utc - ts).total_seconds() / 3600.0
                            if lag_h > max_lag_h:
                                dropped.append((m, lag_h))
                                continue
                            kept.append(m)
                            kept_lags.append(lag_h)

                        if dropped:
                            sample = ", ".join([f"{m}({('None' if lag is None else f'{lag:.1f}h')})" for m, lag in dropped[:5]])
                            logger.warning(
                                f"[Freshness] Dropped {len(dropped)} stale/empty markets over {max_lag_h}h: {sample}"
                            )

                        if not kept:
                            fallback_markets = list(getattr(config.Data, "MARKET_INDEX_COINS", [])) or ["KRW-BTC", "KRW-ETH"]
                            ts_map_fb = get_latest_db_timestamps_by_market(fallback_markets)
                            fb_kept = []
                            fb_lags = []
                            for m in fallback_markets:
                                ts = ts_map_fb.get(m)
                                if ts is None:
                                    continue
                                lag_h = (now_utc - ts).total_seconds() / 3600.0
                                if lag_h <= max_lag_h:
                                    fb_kept.append(m)
                                    fb_lags.append(lag_h)
                            if fb_kept:
                                run_markets = fb_kept
                                logger.warning(
                                    f"[Freshness] Screener markets stale; falling back to index coins: {fb_kept} "
                                    f"(worst_lag={max(fb_lags):.2f}h)"
                                )
                            else:
                                msg = (
                                    f"<b>⚠️ DATA STALE (INTRADAY)</b>\n"
                                    f"<code>kept=0 dropped={len(dropped)}</code>\n"
                                    f"<code>max_lag_h={max_lag_h}</code>\n"
                                    f"\nNo recommendations generated."
                                )
                                if send_tg and bool(getattr(config.Recommender, "STALE_DATA_ALERT_SEND_TELEGRAM", True)):
                                    try:
                                        from utils.telegram_bot import send_alert
                                        send_alert(msg, bypass_dedup=True)
                                    except Exception as e:
                                        logger.error(f"Failed to send stale-data telegram alert: {e}")
                                raise SystemExit(2)
                        if kept:
                            run_markets = kept
                            logger.info(
                                f"[Freshness] Kept {len(kept)}/{len(kept)+len(dropped)} markets within {max_lag_h}h "
                                f"(worst_lag={max(kept_lags):.2f}h)"
                            )
                        run_meta["freshness_dropped"] = len(dropped)
                        run_meta["run_markets_kept"] = len(run_markets)
                    else:
                        db_latest, lag_h = get_db_latest_and_lag_hours(markets=run_markets)
                        if db_latest is not None and lag_h is not None:
                            logger.info(f"[Freshness] DB latest candle (max): {db_latest.isoformat()} (lag={lag_h:.2f}h)")
                except SystemExit:
                    raise
                except Exception as e:
                    logger.warning(f"[Freshness] Freshness gate check failed (proceeding): {e}")

                run_meta.setdefault("run_markets_kept", len(run_markets))
                run_meta.setdefault("freshness_dropped", 0)

                predictions = predictor.run(markets=run_markets)
                for pred in predictions:
                    pred['strategy'] = 'intraday'
                intraday_recs = recommender.run(predictions=predictions, mode='live', min_k=args.min_k)

                # Persist market-selection metrics for analysis/dashboarding (best-effort).
                try:
                    run_meta["elapsed_sec"] = float(time.time() - t0)
                    from utils.run_markets_metrics import record_run
                    record_run("intraday", run_markets, meta=run_meta, recs=intraday_recs)
                except Exception as e:
                    logger.debug(f"[Metrics] intraday record_run failed: {e}")

                if send_tg:
                    try:
                        from utils.telegram_bot import send_alert, format_short_term_report
                        msg = format_short_term_report(intraday_recs, pump_recs=[], meta=run_meta)
                        send_alert(msg, bypass_dedup=True)
                    except Exception as e:
                        logger.error(f"Failed to send intraday telegram report: {e}")

            try:
                with time_limit(timeout, name="intraday"):
                    _intraday_body()
            except TimeoutError as e:
                logger.error(f"[Watchdog] Intraday timeout: {e}")
                if send_tg:
                    try:
                        from utils.telegram_bot import send_alert
                        send_alert(f"<b>⏱ INTRADAY TIMEOUT</b>\n\n<code>{str(e)}</code>", bypass_dedup=True)
                    except Exception as ex:
                        logger.error(f"Failed to send intraday timeout telegram: {ex}")
                raise SystemExit(3)

        logger.info("Intraday inference run finished.")

    elif args.mode == 'morning-report':
        # Inference-only morning report (08:00 snapshot): no training.
        # Recommended schedule: daily at 08:00 (server local time or cron time).
        config.General.REC_TAG = "morning"
        logger.info("Starting morning report run (no training)...")

        from utils.run_lock import run_lock
        from utils.freshness import get_db_latest_and_lag_hours

        send_tg = (not args.no_telegram)
        if args.send_telegram:
            send_tg = True

        from utils.watchdog import time_limit
        timeout = int(args.timeout_sec)
        if timeout == 0:
            timeout = 45 * 60
        if timeout < 0:
            timeout = 0

        def _morning_body():
            t0 = time.time()
            seed_markets = screener.get_trending_markets(mode='live', limit=args.limit, lookback_days=max(args.lookback_days, 1))
            seed_fallback_index = False
            if not seed_markets:
                seed_markets = list(getattr(config.Data, "MARKET_INDEX_COINS", [])) or ["KRW-BTC", "KRW-ETH"]
                seed_fallback_index = True
                logger.warning(
                    f"No trending markets found for morning report; falling back to index coins: {seed_markets}"
                )

            from utils.market_selection import select_markets_for_scheduled_run
            from utils.netcheck import can_resolve

            budget = int(args.market_budget or 0)
            if budget <= 0:
                budget = 64 if str(config.Device.DEVICE).startswith("cuda") else 40

            max_lag_sel = float(getattr(config.Recommender, "DATA_FRESHNESS_MAX_LAG_HOURS_MORNING", 12))
            run_meta = {
                "mode": "morning",
                "budget": budget,
                "seed_n": len(seed_markets or []),
                "seed_fallback_index": bool(seed_fallback_index),
            }
            run_markets, sel_meta = select_markets_for_scheduled_run(
                mode="morning",
                seed_markets=seed_markets,
                budget=budget,
                tv_hours=int(args.refresh_tv_hours),
                candidate_top=max(200, budget * 5),
                lookback_hours=168,
                exploit_target=int(args.refresh_top_n),
                max_holdings=5,
                max_core=10,
                max_lag_h=max_lag_sel,
                return_meta=True,
            )
            run_meta.update(sel_meta or {})

            try:
                from utils.run_markets_cache import rotate_markets
                rot_meta = {}
                run_markets, rot_meta = rotate_markets(
                    mode="morning",
                    new_markets=run_markets,
                    budget=budget,
                    keep_ratio=float(args.rotation_keep),
                    max_lag_h=max_lag_sel,
                )
                logger.info(f"[RunMarkets] rotation kept={rot_meta.get('kept')} added={rot_meta.get('added')} budget={budget}")
                run_meta["rotation_kept"] = rot_meta.get("kept")
                run_meta["rotation_added"] = rot_meta.get("added")
            except Exception as e:
                logger.warning(f"[RunMarkets] rotation failed (proceeding): {e}")

            auto_refresh_enabled = (not args.no_auto_refresh) or args.refresh_data
            if auto_refresh_enabled:
                try:
                    offline = not can_resolve("api.upbit.com")
                    run_meta["auto_refresh_enabled"] = True
                    run_meta["auto_refresh_skipped_offline"] = bool(offline)
                    if not offline:
                        refresh_markets = run_markets
                        logger.info(
                            f"Refreshing latest data for {len(refresh_markets)} markets (days={args.refresh_days})..."
                        )
                        for market in refresh_markets:
                            collector.collect_market_data(market, days=args.refresh_days)
                            time.sleep(0.15)
                    else:
                        logger.warning("Upbit host resolution failed; skipping auto-refresh (offline mode).")
                except Exception as e:
                    logger.warning(f"Auto-refresh failed (proceeding): {e}")
            else:
                run_meta["auto_refresh_enabled"] = False

            # Freshness gate (same as intraday).
            run_meta["run_markets_selected"] = len(run_markets)
            try:
                max_lag_h = float(getattr(config.Recommender, "DATA_FRESHNESS_MAX_LAG_HOURS_MORNING", 12))
                fail_on_stale = bool(getattr(config.Recommender, "FAIL_ON_STALE_DATA_LIVE", True))
                if fail_on_stale and (not args.allow_stale_data):
                    from data.database import get_latest_db_timestamps_by_market
                    now_utc = datetime.now(timezone.utc)
                    ts_map = get_latest_db_timestamps_by_market(run_markets)
                    kept = []
                    dropped = []
                    kept_lags = []
                    for m in run_markets:
                        ts = ts_map.get(m)
                        if ts is None:
                            dropped.append((m, None))
                            continue
                        lag_h = (now_utc - ts).total_seconds() / 3600.0
                        if lag_h > max_lag_h:
                            dropped.append((m, lag_h))
                            continue
                        kept.append(m)
                        kept_lags.append(lag_h)

                    if dropped:
                        sample = ", ".join([f"{m}({('None' if lag is None else f'{lag:.1f}h')})" for m, lag in dropped[:5]])
                        logger.warning(
                            f"[Freshness] Dropped {len(dropped)} stale/empty markets over {max_lag_h}h: {sample}"
                        )

                    if not kept:
                        fallback_markets = list(getattr(config.Data, "MARKET_INDEX_COINS", [])) or ["KRW-BTC", "KRW-ETH"]
                        ts_map_fb = get_latest_db_timestamps_by_market(fallback_markets)
                        fb_kept = []
                        fb_lags = []
                        for m in fallback_markets:
                            ts = ts_map_fb.get(m)
                            if ts is None:
                                continue
                            lag_h = (now_utc - ts).total_seconds() / 3600.0
                            if lag_h <= max_lag_h:
                                fb_kept.append(m)
                                fb_lags.append(lag_h)
                        if fb_kept:
                            run_markets = fb_kept
                            logger.warning(
                                f"[Freshness] Screener markets stale; falling back to index coins: {fb_kept} "
                                f"(worst_lag={max(fb_lags):.2f}h)"
                            )
                        else:
                            msg = (
                                f"<b>⚠️ DATA STALE (MORNING)</b>\n"
                                f"<code>kept=0 dropped={len(dropped)}</code>\n"
                                f"<code>max_lag_h={max_lag_h}</code>\n"
                                f"\nNo recommendations generated."
                            )
                            if send_tg and bool(getattr(config.Recommender, "STALE_DATA_ALERT_SEND_TELEGRAM", True)):
                                try:
                                    from utils.telegram_bot import send_alert
                                    send_alert(msg, bypass_dedup=True)
                                except Exception as e:
                                    logger.error(f"Failed to send stale-data telegram alert: {e}")
                            raise SystemExit(2)
                    if kept:
                        run_markets = kept
                        logger.info(
                            f"[Freshness] Kept {len(kept)}/{len(kept)+len(dropped)} markets within {max_lag_h}h "
                            f"(worst_lag={max(kept_lags):.2f}h)"
                        )
                    run_meta["freshness_dropped"] = len(dropped)
                    run_meta["run_markets_kept"] = len(run_markets)
                else:
                    db_latest, lag_h = get_db_latest_and_lag_hours(markets=run_markets)
                    if db_latest is not None and lag_h is not None:
                        logger.info(f"[Freshness] DB latest candle (max): {db_latest.isoformat()} (lag={lag_h:.2f}h)")
            except SystemExit:
                raise
            except Exception as e:
                logger.warning(f"[Freshness] Freshness gate check failed (proceeding): {e}")

            run_meta.setdefault("run_markets_kept", len(run_markets))
            run_meta.setdefault("freshness_dropped", 0)

            # Trending recommendations
            trend_preds = predictor.run(markets=run_markets)
            for pred in trend_preds:
                pred['strategy'] = 'trending'
            trending_recs = recommender.run(predictions=trend_preds, mode='live', min_k=args.min_k)

            # Pattern followers recommendations (micro-pattern layer)
            pattern_recs = []
            skip_pattern_followers = bool(args.skip_aux or args.skip_pattern_followers)
            if skip_pattern_followers:
                logger.info("[Aux] Skipping pattern followers section.")
            else:
                try:
                    leader_market = seed_markets[0]
                    all_krw_markets_df = database.load_data("SELECT DISTINCT market FROM crypto_data WHERE market LIKE 'KRW-%'")
                    all_krw_markets = all_krw_markets_df['market'].tolist() if not all_krw_markets_df.empty else []
                    other_markets = [m for m in all_krw_markets if m not in run_markets]
                    top_pattern_followers = find_pattern_followers(leader_market, other_markets)
                    if top_pattern_followers:
                        follower_markets = [c['market'] for c in top_pattern_followers]
                        pattern_preds = predictor.run(markets=follower_markets)
                        for pred in pattern_preds:
                            pred['strategy'] = 'pattern'
                        pattern_recs = recommender.run(predictions=pattern_preds, mode='live', min_k=args.min_k)
                except Exception as e:
                    logger.error(f"Pattern follower section failed: {e}")

            # Pump radar (inference-only)
            pump_recs = []
            skip_pump = bool(args.skip_aux or args.skip_pump_radar)
            if skip_pump:
                logger.info("[Aux] Skipping pump radar section.")
            else:
                try:
                    from inference import pump_predictor
                    pump_recs = pump_predictor.run()
                except Exception as e:
                    logger.error(f"Pump radar failed: {e}")

            # Persist market-selection metrics for analysis/dashboarding (best-effort).
            try:
                run_meta["elapsed_sec"] = float(time.time() - t0)
                from utils.run_markets_metrics import record_run
                record_run("morning", run_markets, meta=run_meta, recs=(trending_recs or []) + (pattern_recs or []))
            except Exception as e:
                logger.debug(f"[Metrics] morning record_run failed: {e}")

            if send_tg:
                try:
                    from utils.telegram_bot import send_alert, format_daily_report
                    msg = format_daily_report(trending_recs, pattern_recs, pump_recs, meta=run_meta)
                    send_alert(msg, bypass_dedup=True)
                except Exception as e:
                    logger.error(f"Failed to send morning telegram report: {e}")

        with run_lock("morning-report"):
            try:
                with time_limit(timeout, name="morning-report"):
                    _morning_body()
            except TimeoutError as e:
                logger.error(f"[Watchdog] Morning-report timeout: {e}")
                if send_tg:
                    try:
                        from utils.telegram_bot import send_alert
                        send_alert(f"<b>⏱ MORNING TIMEOUT</b>\n\n<code>{str(e)}</code>", bypass_dedup=True)
                    except Exception as ex:
                        logger.error(f"Failed to send morning timeout telegram: {ex}")
                raise SystemExit(3)

        logger.info("Morning report run finished.")

    elif args.mode == 'screen':
        screener.get_trending_markets(mode='live')

    elif args.mode == 'explain':
        target_market = args.symbol if args.symbol else "KRW-BTC"
        logger.info(f"Running explainability analysis for {target_market}...")
        result = predictor.run_with_explainability(target_market)
        if not result:
            logger.error(f"Explainability analysis failed for {target_market}.")
            raise SystemExit(1)

        pred = np.array(result.get('prediction', [])).flatten()
        gate_expl = result.get('gate_explanation', {})
        uncertainty = result.get('multi_sample', {}).get('uncertainty')

        logger.info(f"Explainability summary for {target_market}:")
        if pred.size:
            logger.info(f"  Predicted pattern (H+1..): {np.round(pred, 6).tolist()}")
        if uncertainty is not None:
            logger.info(f"  Multi-noise uncertainty: {float(uncertainty):.6f}")
        if gate_expl:
            logger.info(
                f"  Gate: {gate_expl.get('gate_value', 0.5):.3f} "
                f"({gate_expl.get('dominant_path', 'Unknown')})"
            )
            logger.info(f"  Interpretation: {gate_expl.get('interpretation', '')}")

        os.makedirs("analysis", exist_ok=True)
        summary_path = os.path.join("analysis", f"explain_{target_market.replace('-', '_')}.json")
        serializable_summary = {
            "market": target_market,
            "prediction": pred.tolist() if pred.size else [],
            "uncertainty": float(uncertainty) if uncertainty is not None else None,
            "gate_explanation": gate_expl,
            "artifacts": {
                "attention": "static/explainability/latest_attention.png",
                "prototypes": "static/explainability/latest_prototypes.png",
                "distribution": "static/explainability/latest_prediction_dist.png",
            },
            "generated_at": datetime.now().isoformat(),
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(serializable_summary, f, ensure_ascii=False, indent=2)
        logger.info(f"Explainability summary saved to {summary_path}")

    elif args.mode == 'continuous':
        if not bool(getattr(args, "enable_continuous", False)):
            logger.error(
                "Refusing to run deprecated mode=continuous. "
                "Use scheduled jobs: refresh-db + intraday + morning-report. "
                "Pass --enable_continuous to override."
            )
            # Best-effort alert so it is visible in ops.
            send_tg = (not args.no_telegram)
            if args.send_telegram:
                send_tg = True
            if send_tg:
                try:
                    from utils.telegram_bot import send_alert
                    send_alert(
                        "<b>🧯 CONTINUOUS MODE BLOCKED</b>\n\n"
                        "<code>continuous is deprecated</code>\n"
                        "<code>use: refresh-db + intraday + morning-report</code>\n"
                        "<code>override: --enable_continuous</code>",
                        bypass_dedup=True,
                    )
                except Exception as e:
                    logger.error(f"Failed to send continuous blocked telegram: {e}")
            raise SystemExit(2)
        try:
            from utils.smart_portfolio import SmartPortfolioManager
            from utils.telegram_bot import send_alert, format_trade_alert, format_status_report, get_dashboard_url
            from inference import pump_predictor
            
            logger.info("🚀 Starting Phase 2: Continuous Profit Engine (Hybrid-Native 24/7) 🚀")
            
            # [Fix] Send startup notification
            try:
                start_msg = f"<b>🚀 SYSTEM RESTARTED</b>\n\n"
                start_msg += f"Engine: Continuous Profit (Model 5)\n"
                start_msg += f"<a href='{get_dashboard_url()}'>📊 Dashboard (Port 5002)</a>"
                send_alert(start_msg, bypass_dedup=True)
            except Exception as e:
                logger.error(f"Startup notification failed: {e}")
            
            # Using the new model path from args or config
            logger.info(f"Model Path: {config.General.MODEL_PATH}")
            
            pm = SmartPortfolioManager()
            
            # 4-Hour Report Timer
            last_report_time = datetime.now()
            last_daily_report_date = None
            
            while True:
                try:
                    now = datetime.now()
                    logger.info("\n==================================================")
                    logger.info(f"⏰ Cycle Start: {now.strftime('%Y-%m-%d %H:%M:%S')}")
                    logger.info("==================================================")

                    # --- [Periodic Status Report (Every 4 Hours)] ---
                    # Check if 4 hours have passed OR if it's strictly 00, 04, 08... ?
                    # Simple interval check: > 4 hours since last report
                    time_diff = (now - last_report_time).total_seconds()
                    if time_diff >= 4 * 3600: 
                        # Send periodic report
                        logger.info("📢 Sending 4H Status Report...")
                        try:
                            # Calculate equity/pnl roughly
                            total_equity = pm.get_total_equity(current_prices={}) # Need to fetch current prices?
                            # pm.positions usually has latest 'current_price' updated after loop below?
                            # Actually we do the loop first, then report.
                            # Let's move this block to END of loop or use Updated data.
                            pass # Will execute at end of loop
                        except Exception as e:
                            logger.error(f"Status Report Prep Failed: {e}")

                    # 1. Candidate Selection (Scanning)
                    trending_markets = screener.get_trending_markets(mode='live')
                    current_holdings = list(pm.positions.keys())
                    monitor_list = list(set(trending_markets + current_holdings))
                    
                    if not monitor_list:
                        logger.info("No active markets to monitor. Sleeping...")
                        time.sleep(300) # Sleep 5 min if empty
                        continue
                        
                    logger.info(f"Targeting {len(monitor_list)} markets.")
                    
                    # 2. Data Collection
                    failed_markets = []
                    current_prices_map = {}
                    
                    for market in monitor_list:
                        try:
                            collector.collect_market_data(market, days=14)
                            # Get latest price for reporting
                            cp = collector.get_current_price(market)
                            if cp: current_prices_map[market] = cp
                            time.sleep(0.2)
                        except Exception as e:
                            logger.error(f"Failed to collect data for {market}: {e}")
                            failed_markets.append(market)
                    
                    valid_list = [m for m in monitor_list if m not in failed_markets]
                    
                    # [RESTORED] 0. Morning Daily Report (08:00 AM)
                    if now.hour == 8 and now.minute < 30 and (last_daily_report_date != now.date()):
                        try:
                            logger.info("📢 Preparing 08:00 AM Daily Report...")
                            from utils.telegram_bot import format_daily_report
                            
                            # Run full prediction for report
                            daily_preds = predictor.run(markets=valid_list)
                            trending = [p for p in daily_preds if p.get('gate_value', 0) > 0.6]
                            patterns = [p for p in daily_preds if p.get('gate_value', 0) < 0.4]
                            
                            # Pump Scan for report (using XGBoost pump_predictor)
                            potential_pumps_daily = pump_predictor.run()
                            
                            daily_msg = format_daily_report(trending, patterns, potential_pumps_daily)
                            send_alert(daily_msg, bypass_dedup=True)
                            
                            last_daily_report_date = now.date()
                            logger.info("✅ Daily Report Sent.")
                        except Exception as de:
                            logger.error(f"Daily Report Failed: {de}")

                    # Pump Scan & Alert (Every Cycle) — using XGBoost pump_predictor
                    potential_pumps = pump_predictor.run()
                    display_pump_candidates(potential_pumps)
                    
                    if potential_pumps:
                        try:
                            from utils.telegram_bot import format_short_term_report
                            # Send Instant Pump Alert
                            pump_msg = format_short_term_report([], potential_pumps)
                            send_alert(pump_msg)
                            logger.info(f"🚀 Sent Pump Alert for {len(potential_pumps)} items.")
                        except Exception as pe:
                            logger.error(f"Pump Alert Failed: {pe}")
                    
                    # 3. Hybrid Prediction (Main execution)
                    if valid_list:
                        predictions = predictor.run(markets=valid_list)
                        
                        # 4. Continuous Rebalancing
                        for pred in predictions:
                            market = pred['market']
                            current_price = pred['current_price']
                            gate_val = pred.get('gate_value', 0.5)
                            consensus = pred.get('consensus_score', 0.5)
                            predicted_pattern = pred['predicted_pattern']
                            
                            expected_return = np.prod(1 + predicted_pattern) - 1
                            n_pos = np.sum(predicted_pattern > 0)
                            consistency = max(n_pos, len(predicted_pattern)-n_pos) / len(predicted_pattern)
                            
                            base_target = 0.0
                            if expected_return > 0.002 and consistency >= 0.66:
                                base_target = 0.1 # 10% base
                            elif expected_return < -0.005:
                                base_target = 0.0
                            elif market in pm.positions:
                                base_target = 0.0 
                            
                            # EXECUTE REBALANCING
                            report = pm.sync_target_weight(
                                market=market,
                                target_weight=base_target,
                                current_price=current_price,
                                gate_value=gate_val,
                                consensus_score=consensus
                            )
                            
                            # [TELEGRAM INSTANT ALERT]
                            if report['action'] in ['BUY', 'SELL']:
                                action_icon = "🟢" if report['action'] == "BUY" else "🔴"
                                logger.info(f"{action_icon} {report['action']} {market}: {report['reason']}")
                                
                                try:
                                    # Calculate Realized PnL if SELL
                                    pnl = report.get('realized_pnl', None)
                                    alert_msg = format_trade_alert(
                                        action=report['action'],
                                        market=market,
                                        price=report['price'],
                                        reason=report['reason'],
                                        pnl=pnl
                                    )
                                    send_alert(alert_msg)
                                except Exception as te:
                                    logger.error(f"Telegram Alert Failed: {te}")

                    # --- [Periodic Status Report Execution] ---
                    # Check time again
                    time_diff = (datetime.now() - last_report_time).total_seconds()
                    if time_diff >= 4 * 3600: 
                        try:
                            # [Fix] Convert Position objects to dicts for reporting
                            active_pos_dicts = {k: v.to_dict() for k, v in pm.positions.items()}
                            
                            # Update prices in dicts for accurate pnl display
                            for m, data in active_pos_dicts.items():
                                if m in current_prices_map:
                                    data['current_price'] = current_prices_map[m]
                                    
                            status_msg = format_status_report(
                                active_positions=active_pos_dicts,
                                total_equity=pm.get_equity(current_prices_map),
                                profit_loss=0.0 
                            )
                            send_alert(status_msg, bypass_dedup=True)
                            last_report_time = datetime.now() # Reset timer
                            logger.info("Sent 4H Status Report.")
                        except Exception as e:
                            logger.error(f"Failed to send 4H report: {e}")

                    logger.info("Cycle complete. Sleeping for 30 minutes...")
                    time.sleep(1800) 

                except Exception as inner_e:
                    logger.error(f"Error in continuous cycle: {inner_e}")
                    time.sleep(60)

        except KeyboardInterrupt:
            logger.info("Continuous engine stopped by user.")
        except Exception as e:
            logger.error(f"Fatal error in continuous engine: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
