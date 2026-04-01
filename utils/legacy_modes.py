from datetime import datetime
import time
import traceback

from data import collector, database
from inference import predictor
from utils import screener


def run_daily_mode(
    args,
    *,
    config,
    logger,
    trainer,
    collect_recent_market_data,
    run_recommender_for_markets,
    find_pattern_followers,
    display_pump_candidates,
    save_pump_predictions_to_csv,
    open_daily_trades,
    send_morning_report,
    generate_research_report,
    evaluate_previous_daily_recommendations,
):
    config.General.REC_TAG = "daily"

    from training import pump_trainer
    from inference import pump_predictor
    from utils.portfolio_manager import PortfolioManager
    from data.collector import get_current_price

    logger.info("Starting daily run (including model reinforcement)...")
    pm = PortfolioManager()

    logger.info("=== PERFORMANCE TRACKER: Checking Yesterday's Recommendations ===")
    try:
        conn = pm._get_conn()
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT market FROM trades WHERE status='OPEN' AND strategy='daily'")
        open_markets = [row[0] for row in cur.fetchall()]
        conn.close()

        current_prices_map = {}
        if open_markets:
            for market in open_markets:
                price = get_current_price(market)
                if price:
                    current_prices_map[market] = price
            pm.close_open_trades('daily', current_prices_map)
            logger.info("Portfolio Manager: Closed previous Daily trades.")
    except Exception as e:
        logger.error(f"Portfolio Manager Error (Close): {e}")

    try:
        evaluate_previous_daily_recommendations()
    except Exception as e:
        logger.error(f"Performance tracking failed: {e}. Continuing with daily run.")

    logger.info("=== END PERFORMANCE REPORT ===")

    trending_markets = screener.get_trending_markets(mode='live')
    trending_recs = []
    if trending_markets:
        logger.info(f"Collecting latest data for {len(trending_markets)} trending markets...")
        collect_recent_market_data(trending_markets, days=30, sleep_sec=0.5, continue_on_error=False)
        logger.info("Fine-tuning main trend model...")
        trainer.run(markets=trending_markets, epochs=args.daily_epochs)

        logger.info("Generating TRENDING predictions...")
        trending_recs = run_recommender_for_markets(trending_markets, strategy_name="trending", min_k=3)
    else:
        logger.info("No trending markets found today.")

    pattern_recs = []
    if trending_markets:
        leader_market = trending_markets[0]
        logger.info(f"Leader Market for Pattern Matching: {leader_market}")
        all_krw_markets_df = database.load_data("SELECT DISTINCT market FROM crypto_data WHERE market LIKE 'KRW-%'")
        all_krw_markets = all_krw_markets_df['market'].tolist() if not all_krw_markets_df.empty else []
        other_markets = [m for m in all_krw_markets if m not in trending_markets]
        top_pattern_followers = find_pattern_followers(leader_market, other_markets)

        if top_pattern_followers:
            follower_markets = [c['market'] for c in top_pattern_followers]
            logger.info(f"Collecting latest data for {len(follower_markets)} pattern-following coins...")
            collect_recent_market_data(follower_markets, days=30, sleep_sec=0.5, continue_on_error=False)

            logger.info("Generating PATTERN predictions...")
            pattern_recs = run_recommender_for_markets(follower_markets, strategy_name="pattern", min_k=3)

    logger.info("Fine-tuning pump prediction model...")
    pump_trainer.run()
    potential_pumps = pump_predictor.run()
    display_pump_candidates(potential_pumps)
    save_pump_predictions_to_csv(potential_pumps)

    all_final_recs = trending_recs + pattern_recs
    if all_final_recs:
        logger.info(f"Portfolio Manager: Opening {len(all_final_recs)} new 'Daily' trades.")
        open_daily_trades(pm, all_final_recs)

    try:
        send_morning_report(trending_recs, pattern_recs, potential_pumps, run_meta={})
    except Exception as e:
        logger.error(f"Failed to send daily telegram report: {e}")

    logger.info("Generating Daily Research Report...")
    try:
        generate_research_report()
    except Exception as e:
        logger.error(f"Failed to generate research report: {e}")

    logger.info("Daily run finished.")


def run_continuous_mode(
    args,
    *,
    config,
    logger,
    send_continuous_blocked_alert,
    send_continuous_startup_alert,
    collect_monitor_market_data,
    maybe_send_continuous_daily_report,
    display_pump_candidates,
    maybe_send_continuous_pump_alert,
    rebalance_continuous_predictions,
    maybe_send_continuous_status_report,
):
    if not bool(getattr(args, "enable_continuous", False)):
        logger.error(
            "Refusing to run deprecated mode=continuous. "
            "Use scheduled jobs: refresh-db + intraday + morning-report. "
            "Pass --enable_continuous to override."
        )
        send_tg = (not args.no_telegram)
        if args.send_telegram:
            send_tg = True
        if send_tg:
            try:
                send_continuous_blocked_alert()
            except Exception as e:
                logger.error(f"Failed to send continuous blocked telegram: {e}")
        raise SystemExit(2)

    try:
        from utils.smart_portfolio import SmartPortfolioManager
        from utils.telegram_bot import send_alert, format_trade_alert, format_status_report, get_dashboard_url
        from inference import pump_predictor

        logger.info("🚀 Starting Phase 2: Continuous Profit Engine (Hybrid-Native 24/7) 🚀")
        try:
            send_continuous_startup_alert(get_dashboard_url)
        except Exception as e:
            logger.error(f"Startup notification failed: {e}")

        logger.info(f"Model Path: {config.General.MODEL_PATH}")
        pm = SmartPortfolioManager()
        last_report_time = datetime.now()
        last_daily_report_date = None

        while True:
            try:
                now = datetime.now()
                logger.info("\n==================================================")
                logger.info(f"⏰ Cycle Start: {now.strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info("==================================================")

                trending_markets = screener.get_trending_markets(mode='live')
                current_holdings = list(pm.positions.keys())
                monitor_list = list(set(trending_markets + current_holdings))

                if not monitor_list:
                    logger.info("No active markets to monitor. Sleeping...")
                    time.sleep(300)
                    continue

                logger.info(f"Targeting {len(monitor_list)} markets.")
                valid_list, current_prices_map, _failed_markets = collect_monitor_market_data(
                    monitor_list,
                    days=14,
                    sleep_sec=0.2,
                )

                last_daily_report_date = maybe_send_continuous_daily_report(
                    now=now,
                    last_daily_report_date=last_daily_report_date,
                    valid_list=valid_list,
                    pump_predictor=pump_predictor,
                )

                potential_pumps = pump_predictor.run()
                display_pump_candidates(potential_pumps)
                maybe_send_continuous_pump_alert(potential_pumps)

                if valid_list:
                    predictions = predictor.run(markets=valid_list)
                    rebalance_continuous_predictions(
                        pm=pm,
                        predictions=predictions,
                        format_trade_alert=format_trade_alert,
                        send_alert=send_alert,
                    )

                last_report_time = maybe_send_continuous_status_report(
                    pm=pm,
                    current_prices_map=current_prices_map,
                    last_report_time=last_report_time,
                    format_status_report=format_status_report,
                    send_alert=send_alert,
                )

                logger.info("Cycle complete. Sleeping for 30 minutes...")
                time.sleep(1800)
            except Exception as inner_e:
                logger.error(f"Error in continuous cycle: {inner_e}")
                time.sleep(60)

    except KeyboardInterrupt:
        logger.info("Continuous engine stopped by user.")
    except Exception as e:
        logger.error(f"Fatal error in continuous engine: {e}")
        traceback.print_exc()
