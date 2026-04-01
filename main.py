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


def collect_recent_market_data(markets, days: int, sleep_sec: float = 0.5, continue_on_error: bool = False):
    for market in markets or []:
        try:
            collector.collect_market_data(market, days=days)
        except Exception as e:
            logger.error(f"collect_market_data failed for {market}: {e}")
            if not continue_on_error:
                raise
        if sleep_sec > 0:
            time.sleep(float(sleep_sec))

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


def collect_refresh_batch(markets, days: int, sleep_sec: float = 0.15):
    results = []
    for market in markets:
        try:
            result = collector.collect_market_data(market, days=days)
        except Exception as e:
            logger.error(f"collect_market_data failed for {market}: {e}")
            result = {
                "market": market,
                "status": "failed",
                "saved_records": 0,
                "pages": 0,
                "request_attempts": 0,
                "error": str(e),
            }
        results.append(result or {"market": market, "status": "unknown", "saved_records": 0})
        if sleep_sec > 0:
            time.sleep(float(sleep_sec))
    return results


def summarize_refresh_results(results):
    summary = {
        "attempted": len(results or []),
        "ok": 0,
        "up_to_date": 0,
        "no_data": 0,
        "failed": 0,
        "saved_records": 0,
        "index_failed": [],
    }
    index_coins = set(getattr(config.Data, "MARKET_INDEX_COINS", []) or [])
    for item in results or []:
        status = str((item or {}).get("status", "unknown") or "unknown")
        summary["saved_records"] += int((item or {}).get("saved_records", 0) or 0)
        if status in ("ok", "up_to_date", "no_data", "failed"):
            summary[status] += 1
        if status == "failed" and (item or {}).get("market") in index_coins:
            summary["index_failed"].append((item or {}).get("market"))
    return summary


def alert_index_refresh_failures(send_tg: bool, context_label: str, refresh_summary: dict):
    failed = list((refresh_summary or {}).get("index_failed") or [])
    if not failed:
        return
    failed_txt = ", ".join(failed)
    logger.error(f"[Refresh] Critical index refresh failure during {context_label}: {failed_txt}")
    if send_tg:
        try:
            from utils.telegram_bot import send_alert
            send_alert(
                f"<b>⚠️ INDEX REFRESH FAILED</b>\n"
                f"<code>stage={context_label}</code>\n"
                f"<code>markets={failed_txt}</code>",
                bypass_dedup=True,
            )
        except Exception as e:
            logger.error(f"Failed to send index refresh failure alert: {e}")


def persist_run_outputs(mode: str, markets, run_meta: dict, recs: list, include_pump_preds: bool = False):
    from utils.run_markets_metrics import record_run
    from utils.output_contract import build_output_manifest, write_output_manifest

    try:
        diag_path = recommender.get_last_run_diagnostics_path()
        if diag_path and mode != "refresh-db":
            run_meta = dict(run_meta or {})
            run_meta["recommender_diagnostics_path"] = diag_path
    except Exception:
        pass

    payload = record_run(mode, markets, meta=run_meta, recs=recs)
    metrics_path = os.path.join("analysis", f"run_markets_metrics_{mode}.json")
    include_recommendation = mode != "refresh-db"
    include_prediction = mode != "refresh-db"
    manifest = build_output_manifest(
        mode=mode,
        rec_tag=str(getattr(config.General, "REC_TAG", mode) or mode),
        run_metrics_path=metrics_path,
        include_pump_preds=include_pump_preds,
        include_recommendation=include_recommendation,
        include_prediction=include_prediction,
    )
    manifest["run_metrics_payload"] = payload
    manifest_path = write_output_manifest(mode, manifest)
    return payload, manifest_path


def apply_refresh_summary(run_meta: dict, refresh_summary: dict):
    run_meta["refresh_attempted"] = int((refresh_summary or {}).get("attempted", 0) or 0)
    run_meta["refresh_ok"] = int((refresh_summary or {}).get("ok", 0) or 0)
    run_meta["refresh_up_to_date"] = int((refresh_summary or {}).get("up_to_date", 0) or 0)
    run_meta["refresh_no_data"] = int((refresh_summary or {}).get("no_data", 0) or 0)
    run_meta["refresh_failed"] = int((refresh_summary or {}).get("failed", 0) or 0)
    run_meta["refresh_saved_records"] = int((refresh_summary or {}).get("saved_records", 0) or 0)
    run_meta["refresh_index_failed"] = list((refresh_summary or {}).get("index_failed") or [])


def maybe_auto_refresh_markets(
    run_markets,
    refresh_days: int,
    run_meta: dict,
    send_tg: bool,
    context_label: str,
    sleep_sec: float = 0.15,
):
    from utils.netcheck import resolution_status

    dns_attempts = int(getattr(config.Recommender, "DNS_RESOLVE_RETRIES", 3) or 3)
    dns_delay_sec = float(getattr(config.Recommender, "DNS_RESOLVE_RETRY_DELAY_SEC", 0.25) or 0.25)
    dns_status = resolution_status("api.upbit.com", attempts=dns_attempts, delay_sec=dns_delay_sec)
    offline = not bool(dns_status.get("ok"))
    run_meta["auto_refresh_enabled"] = True
    run_meta["auto_refresh_skipped_offline"] = bool(offline)
    run_meta["auto_refresh_dns_attempts"] = int(dns_status.get("attempts") or dns_attempts)
    run_meta["auto_refresh_dns_error"] = dns_status.get("error")
    run_meta["auto_refresh_dns_ips"] = list(dns_status.get("ips") or [])
    if offline:
        logger.warning(
            "Upbit host resolution failed; skipping auto-refresh "
            f"(attempts={run_meta['auto_refresh_dns_attempts']}, error={run_meta['auto_refresh_dns_error'] or 'unknown'})."
        )
        return None

    logger.info(f"Refreshing latest data for {len(run_markets)} markets (days={refresh_days})...")
    refresh_results = collect_refresh_batch(run_markets, days=refresh_days, sleep_sec=sleep_sec)
    refresh_summary = summarize_refresh_results(refresh_results)
    apply_refresh_summary(run_meta, refresh_summary)
    alert_index_refresh_failures(send_tg, context_label, refresh_summary)
    return refresh_summary


def enforce_freshness_gate(
    mode_label: str,
    run_markets,
    run_meta: dict,
    send_tg: bool,
    max_lag_h: float,
    allow_stale_data: bool = False,
):
    from utils.freshness import evaluate_market_freshness, get_db_latest_and_lag_hours

    run_meta["run_markets_selected"] = len(run_markets)
    fail_on_stale = bool(getattr(config.Recommender, "FAIL_ON_STALE_DATA_LIVE", True))
    if fail_on_stale and (not allow_stale_data):
        fallback_markets = list(getattr(config.Data, "MARKET_INDEX_COINS", [])) or ["KRW-BTC", "KRW-ETH"]
        freshness = evaluate_market_freshness(
            run_markets,
            max_lag_h=max_lag_h,
            fallback_markets=fallback_markets,
        )
        dropped = freshness["dropped"]
        if dropped:
            sample = ", ".join([f"{m}({('None' if lag is None else f'{lag:.1f}h')})" for m, lag in dropped[:5]])
            logger.warning(
                f"[Freshness] Dropped {len(dropped)} stale/empty markets over {max_lag_h}h: {sample}"
            )

        if freshness["status"] == "stale_abort":
            mode_txt = str(mode_label).upper()
            msg = (
                f"<b>⚠️ DATA STALE ({mode_txt})</b>\n"
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

        run_markets = freshness["kept_markets"]
        if freshness["used_fallback"]:
            logger.warning(
                f"[Freshness] Screener markets stale; falling back to index coins: {run_markets} "
                f"(worst_lag={float(freshness['worst_lag_h'] or 0.0):.2f}h)"
            )
        else:
            logger.info(
                f"[Freshness] Kept {len(run_markets)}/{len(run_markets)+len(dropped)} markets within {max_lag_h}h "
                f"(worst_lag={float(freshness['worst_lag_h'] or 0.0):.2f}h)"
            )
        run_meta["freshness_used_fallback"] = bool(freshness["used_fallback"])
        run_meta["freshness_fallback_markets"] = list(freshness["fallback_markets"])
        run_meta["freshness_dropped"] = len(dropped)
        run_meta["run_markets_kept"] = len(run_markets)
    else:
        db_latest, lag_h = get_db_latest_and_lag_hours(markets=run_markets)
        if db_latest is not None and lag_h is not None:
            logger.info(f"[Freshness] DB latest candle (max): {db_latest.isoformat()} (lag={lag_h:.2f}h)")

    run_meta.setdefault("run_markets_kept", len(run_markets))
    run_meta.setdefault("freshness_dropped", 0)
    run_meta.setdefault("freshness_used_fallback", False)
    run_meta.setdefault("freshness_fallback_markets", [])
    return run_markets


def attach_strategy(predictions, strategy_name: str):
    for pred in predictions or []:
        pred["strategy"] = strategy_name
    return predictions or []


def run_recommender_for_markets(markets, strategy_name: str, min_k: int):
    logger.info(
        f"[OpsTrace] run_recommender_for_markets start strategy={strategy_name} "
        f"markets_n={len(markets or [])} min_k={int(min_k)}"
    )
    try:
        predictions = predictor.run(markets=markets)
        logger.info(
            f"[OpsTrace] predictor.run finished strategy={strategy_name} "
            f"predictions_n={len(predictions or [])}"
        )
    except Exception:
        logger.exception(f"[OpsTrace] predictor.run failed strategy={strategy_name}")
        raise

    attach_strategy(predictions, strategy_name)

    try:
        recs = recommender.run(predictions=predictions, mode='live', min_k=min_k)
        logger.info(
            f"[OpsTrace] recommender.run finished strategy={strategy_name} "
            f"recs_n={len(recs or [])}"
        )
        return recs
    except Exception:
        logger.exception(f"[OpsTrace] recommender.run failed strategy={strategy_name}")
        raise


def run_pattern_followers_section(seed_markets, run_markets, min_k: int):
    if not seed_markets:
        return []

    leader_market = seed_markets[0]
    all_krw_markets_df = database.load_data("SELECT DISTINCT market FROM crypto_data WHERE market LIKE 'KRW-%'")
    all_krw_markets = all_krw_markets_df['market'].tolist() if not all_krw_markets_df.empty else []
    other_markets = [m for m in all_krw_markets if m not in run_markets]
    top_pattern_followers = find_pattern_followers(leader_market, other_markets)
    if not top_pattern_followers:
        return []

    follower_markets = [c['market'] for c in top_pattern_followers]
    return run_recommender_for_markets(follower_markets, strategy_name="pattern", min_k=min_k)


def run_pump_radar_section():
    from inference import pump_predictor
    return pump_predictor.run()


def send_intraday_report(intraday_recs, run_meta: dict):
    from utils.telegram_bot import send_alert, format_short_term_report, should_send_signal_alert
    actionable = [
        item for item in (intraday_recs or [])
        if str(item.get("status", "") or "").startswith(("Recommended", "Forced"))
        and not str(item.get("status", "") or "").startswith("Watch")
    ]
    if not actionable:
        logger.info("Skipping intraday telegram: no actionable entries.")
        return
    if not should_send_signal_alert("intraday", actionable):
        logger.info("Skipping intraday telegram: actionable signal set unchanged.")
        return
    msg = format_short_term_report(actionable, pump_recs=[], meta=run_meta)
    send_alert(msg, bypass_dedup=True)


def send_morning_report(trending_recs, pattern_recs, pump_recs, run_meta: dict):
    from utils.telegram_bot import send_alert, format_daily_report, should_send_signal_alert
    actionable_trending = [
        item for item in (trending_recs or [])
        if str(item.get("status", "") or "").startswith(("Recommended", "Forced"))
        and not str(item.get("status", "") or "").startswith("Watch")
    ]
    actionable_pattern = [
        item for item in (pattern_recs or [])
        if str(item.get("status", "") or "").startswith(("Recommended", "Forced"))
        and not str(item.get("status", "") or "").startswith("Watch")
    ]
    if not actionable_trending and not actionable_pattern:
        logger.info("Skipping morning telegram: no actionable entries.")
        return
    actionable = list(actionable_trending) + list(actionable_pattern)
    if not should_send_signal_alert("morning", actionable):
        logger.info("Skipping morning telegram: actionable signal set unchanged.")
        return
    msg = format_daily_report(actionable_trending, actionable_pattern, pump_recs, meta=run_meta)
    send_alert(msg, bypass_dedup=True)


def send_timeout_alert(mode_label: str, error_text: str):
    from utils.telegram_bot import send_alert
    send_alert(f"<b>⏱ {str(mode_label).upper()} TIMEOUT</b>\n\n<code>{error_text}</code>", bypass_dedup=True)


def send_refresh_done_alert(markets_count: int, refresh_days: int):
    from utils.telegram_bot import send_alert
    msg = (
        "<b>✅ REFRESH-DB DONE</b>\n"
        f"<code>markets={markets_count}</code>\n"
        f"<code>days={refresh_days}</code>\n"
    )
    send_alert(msg, bypass_dedup=True)


def open_daily_trades(pm, recommendations):
    for rec in recommendations or []:
        pm.add_trade(
            market=rec['market'],
            strategy='daily',
            signal=rec['signal'],
            entry_price=rec['current_price']
        )


def generate_research_report():
    research_reporter.run()


def evaluate_previous_daily_recommendations():
    from utils.model_tracker import get_tracker
    from data.collector import get_current_price
    import glob

    perf_tracker = get_tracker(n_models=config.Gan.N_ENSEMBLE_MODELS)
    strategy_to_model = {
        'trending': 0,       # Scalper
        'mean_reversion': 1, # Swing Trader
        'continuous': 2,     # Trend Follower
        'pattern': 3,        # Regime Sentinel
    }

    rec_files = sorted(glob.glob(os.path.join("recommendations", "recs_*.csv")), reverse=True)
    if not rec_files:
        logger.info("No previous recommendation files found. (This is normal for first run.)")
        return

    latest_rec_file = rec_files[0]
    logger.info(f"Found previous recommendations: {os.path.basename(latest_rec_file)}")
    past_recs = pd.read_csv(latest_rec_file)
    if past_recs.empty:
        logger.info("No past recommendations to evaluate.")
        return

    results = []
    tracker_updates = []
    for _, row in past_recs.iterrows():
        market = row['market']
        predicted_return = row['expected_return']
        entry_price = row['current_price']
        signal = row.get('signal', 'Unknown')
        strategy = str(row.get('strategy', 'daily'))

        current_price = get_current_price(market)
        if not current_price or current_price <= 0:
            continue

        actual_return = (current_price - entry_price) / entry_price
        if signal == 'Short':
            actual_return = -actual_return

        error = abs(predicted_return - actual_return)
        direction_correct = (predicted_return * actual_return) > 0
        results.append({
            'market': market,
            'signal': signal,
            'predicted_%': predicted_return,
            'actual_%': actual_return,
            'error_%': error,
            'direction_ok': direction_correct
        })

        model_id = 4
        for key, mid in strategy_to_model.items():
            if key in strategy.lower() or key in os.path.basename(latest_rec_file).lower():
                model_id = mid
                break
        tracker_updates.append((model_id, direction_correct))

    if not results:
        logger.info("No past recommendations to evaluate.")
        return

    results_df = pd.DataFrame(results)
    accuracy = results_df['direction_ok'].mean() * 100
    avg_error = results_df['error_%'].mean() * 100

    logger.info("  PERFORMANCE REPORT:")
    logger.info(f"   - Total Tested: {len(results)} recommendations")
    logger.info(f"   - Direction Accuracy: {accuracy:.1f}%")
    logger.info(f"   - Average Prediction Error: {avg_error:.2f}%")
    for res in results:
        status = "[OK]" if res['direction_ok'] else "[X]"
        logger.info(f"   {status} {res['market']} ({res['signal']}): Predicted {res['predicted_%']:+.2%}, Actual {res['actual_%']:+.2%}")

    if tracker_updates:
        perf_tracker.update_batch(tracker_updates)
        perf_tracker.save()
        weights = perf_tracker.get_weights()
        logger.info(f"  📊 ModelPerformanceTracker 업데이트: {len(tracker_updates)}건 반영")
        logger.info(f"  📊 새 가중치: {[f'{w:.3f}' for w in weights]}")


def send_continuous_blocked_alert():
    from utils.telegram_bot import send_alert
    send_alert(
        "<b>🧯 CONTINUOUS MODE BLOCKED</b>\n\n"
        "<code>continuous is deprecated</code>\n"
        "<code>use: refresh-db + intraday + morning-report</code>\n"
        "<code>override: --enable_continuous</code>",
        bypass_dedup=True,
    )


def send_continuous_startup_alert(get_dashboard_url):
    from utils.telegram_bot import send_alert
    start_msg = (
        "<b>🚀 SYSTEM RESTARTED</b>\n\n"
        "Engine: Continuous Profit (Model 5)\n"
        f"<a href='{get_dashboard_url()}'>📊 Dashboard (Port 5002)</a>"
    )
    send_alert(start_msg, bypass_dedup=True)


def collect_monitor_market_data(monitor_list, days: int = 14, sleep_sec: float = 0.2):
    failed_markets = []
    current_prices_map = {}
    for market in monitor_list:
        try:
            collector.collect_market_data(market, days=days)
            cp = collector.get_current_price(market)
            if cp:
                current_prices_map[market] = cp
            if sleep_sec > 0:
                time.sleep(float(sleep_sec))
        except Exception as e:
            logger.error(f"Failed to collect data for {market}: {e}")
            failed_markets.append(market)
    valid_list = [m for m in monitor_list if m not in failed_markets]
    return valid_list, current_prices_map, failed_markets


def maybe_send_continuous_daily_report(now, last_daily_report_date, valid_list, pump_predictor):
    if not valid_list:
        return last_daily_report_date
    if not (now.hour == 8 and now.minute < 30 and (last_daily_report_date != now.date())):
        return last_daily_report_date

    try:
        from utils.telegram_bot import send_alert, format_daily_report
        logger.info("📢 Preparing 08:00 AM Daily Report...")
        daily_preds = predictor.run(markets=valid_list)
        trending = [p for p in daily_preds if p.get('gate_value', 0) > 0.6]
        patterns = [p for p in daily_preds if p.get('gate_value', 0) < 0.4]
        potential_pumps_daily = pump_predictor.run()
        daily_msg = format_daily_report(trending, patterns, potential_pumps_daily)
        send_alert(daily_msg, bypass_dedup=True)
        logger.info("✅ Daily Report Sent.")
        return now.date()
    except Exception as e:
        logger.error(f"Daily Report Failed: {e}")
        return last_daily_report_date


def maybe_send_continuous_pump_alert(potential_pumps):
    if not potential_pumps:
        return
    try:
        from utils.telegram_bot import send_alert, format_short_term_report
        pump_msg = format_short_term_report([], potential_pumps)
        send_alert(pump_msg)
        logger.info(f"🚀 Sent Pump Alert for {len(potential_pumps)} items.")
    except Exception as e:
        logger.error(f"Pump Alert Failed: {e}")


def rebalance_continuous_predictions(pm, predictions, format_trade_alert, send_alert):
    for pred in predictions or []:
        market = pred['market']
        current_price = pred['current_price']
        gate_val = pred.get('gate_value', 0.5)
        consensus = pred.get('consensus_score', 0.5)
        predicted_pattern = pred['predicted_pattern']

        expected_return = np.prod(1 + predicted_pattern) - 1
        n_pos = np.sum(predicted_pattern > 0)
        consistency = max(n_pos, len(predicted_pattern) - n_pos) / len(predicted_pattern)

        base_target = 0.0
        if expected_return > 0.002 and consistency >= 0.66:
            base_target = 0.1
        elif expected_return < -0.005:
            base_target = 0.0
        elif market in pm.positions:
            base_target = 0.0

        report = pm.sync_target_weight(
            market=market,
            target_weight=base_target,
            current_price=current_price,
            gate_value=gate_val,
            consensus_score=consensus
        )

        if report['action'] in ['BUY', 'SELL']:
            action_icon = "🟢" if report['action'] == "BUY" else "🔴"
            logger.info(f"{action_icon} {report['action']} {market}: {report['reason']}")
            try:
                pnl = report.get('realized_pnl', None)
                alert_msg = format_trade_alert(
                    action=report['action'],
                    market=market,
                    price=report['price'],
                    reason=report['reason'],
                    pnl=pnl
                )
                send_alert(alert_msg)
            except Exception as e:
                logger.error(f"Telegram Alert Failed: {e}")


def maybe_send_continuous_status_report(pm, current_prices_map, last_report_time, format_status_report, send_alert):
    time_diff = (datetime.now() - last_report_time).total_seconds()
    if time_diff < 4 * 3600:
        return last_report_time
    try:
        active_pos_dicts = {k: v.to_dict() for k, v in pm.positions.items()}
        for market, data in active_pos_dicts.items():
            if market in current_prices_map:
                data['current_price'] = current_prices_map[market]
        status_msg = format_status_report(
            active_positions=active_pos_dicts,
            total_equity=pm.get_equity(current_prices_map),
            profit_loss=0.0
        )
        send_alert(status_msg, bypass_dedup=True)
        logger.info("Sent 4H Status Report.")
        return datetime.now()
    except Exception as e:
        logger.error(f"Failed to send 4H report: {e}")
        return last_report_time

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

    def handle_init_db():
        database.init_db()

    def handle_collect_all():
        collector.run_all(days=args.days)

    def handle_refresh_db():
        from utils.scheduled_modes import run_refresh_db_mode

        run_refresh_db_mode(
            args,
            config=config,
            logger=logger,
            collect_refresh_batch=collect_refresh_batch,
            summarize_refresh_results=summarize_refresh_results,
            alert_index_refresh_failures=alert_index_refresh_failures,
            persist_run_outputs=persist_run_outputs,
            send_refresh_done_alert=send_refresh_done_alert,
        )

    def handle_train():
        from utils.command_modes import run_train_mode

        run_train_mode(args, config=config, logger=logger, trainer=trainer)

    def handle_train_pump():
        from utils.command_modes import run_train_pump_mode

        run_train_pump_mode(args)

    def handle_find_pumps():
        from utils.command_modes import run_find_pumps_mode

        run_find_pumps_mode(
            display_pump_candidates=display_pump_candidates,
            save_pump_predictions_to_csv=save_pump_predictions_to_csv,
        )

    def handle_backtest():
        from utils.command_modes import run_backtest_mode

        run_backtest_mode(args)

    def handle_daily():
        from utils.legacy_modes import run_daily_mode

        run_daily_mode(
            args,
            config=config,
            logger=logger,
            trainer=trainer,
            collect_recent_market_data=collect_recent_market_data,
            run_recommender_for_markets=run_recommender_for_markets,
            find_pattern_followers=find_pattern_followers,
            display_pump_candidates=display_pump_candidates,
            save_pump_predictions_to_csv=save_pump_predictions_to_csv,
            open_daily_trades=open_daily_trades,
            send_morning_report=send_morning_report,
            generate_research_report=generate_research_report,
            evaluate_previous_daily_recommendations=evaluate_previous_daily_recommendations,
        )

    def handle_quick_recommend():
        from utils.command_modes import run_quick_recommend_mode

        run_quick_recommend_mode(
            logger=logger,
            collect_recent_market_data=collect_recent_market_data,
            attach_strategy=attach_strategy,
        )

    def handle_intraday():
        from utils.scheduled_modes import run_intraday_mode

        run_intraday_mode(
            args,
            config=config,
            logger=logger,
            maybe_auto_refresh_markets=maybe_auto_refresh_markets,
            enforce_freshness_gate=enforce_freshness_gate,
            run_recommender_for_markets=run_recommender_for_markets,
            persist_run_outputs=persist_run_outputs,
            send_intraday_report=send_intraday_report,
            send_timeout_alert=send_timeout_alert,
        )

    def handle_morning_report():
        from utils.scheduled_modes import run_morning_report_mode

        run_morning_report_mode(
            args,
            config=config,
            logger=logger,
            maybe_auto_refresh_markets=maybe_auto_refresh_markets,
            enforce_freshness_gate=enforce_freshness_gate,
            run_recommender_for_markets=run_recommender_for_markets,
            run_pattern_followers_section=run_pattern_followers_section,
            run_pump_radar_section=run_pump_radar_section,
            persist_run_outputs=persist_run_outputs,
            send_morning_report=send_morning_report,
            send_timeout_alert=send_timeout_alert,
        )

    def handle_screen():
        screener.get_trending_markets(mode='live')

    def handle_explain():
        from utils.command_modes import run_explain_mode

        run_explain_mode(args, logger=logger)

    def handle_continuous():
        from utils.legacy_modes import run_continuous_mode

        run_continuous_mode(
            args,
            config=config,
            logger=logger,
            send_continuous_blocked_alert=send_continuous_blocked_alert,
            send_continuous_startup_alert=send_continuous_startup_alert,
            collect_monitor_market_data=collect_monitor_market_data,
            maybe_send_continuous_daily_report=maybe_send_continuous_daily_report,
            display_pump_candidates=display_pump_candidates,
            maybe_send_continuous_pump_alert=maybe_send_continuous_pump_alert,
            rebalance_continuous_predictions=rebalance_continuous_predictions,
            maybe_send_continuous_status_report=maybe_send_continuous_status_report,
        )

    mode_handlers = {
        'init_db': handle_init_db,
        'collect-all': handle_collect_all,
        'refresh-db': handle_refresh_db,
        'train': handle_train,
        'train-pump': handle_train_pump,
        'find-pumps': handle_find_pumps,
        'backtest': handle_backtest,
        'daily': handle_daily,
        'quick-recommend': handle_quick_recommend,
        'intraday': handle_intraday,
        'morning-report': handle_morning_report,
        'screen': handle_screen,
        'explain': handle_explain,
        'continuous': handle_continuous,
    }
    mode_handlers[args.mode]()

if __name__ == "__main__":
    main()
