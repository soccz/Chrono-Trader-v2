import json
import os
import time
from datetime import datetime

import numpy as np

from data import collector
from data.database import get_data_period
from inference import predictor, recommender
from utils import screener


def run_train_mode(args, *, config, logger, trainer):
    logger.info("Starting full model training...")
    days_available = get_data_period()
    if str(config.Device.DEVICE).startswith("cuda"):
        min_days, max_days = 180, 720
    else:
        min_days, max_days = 120, 365

    training_days = min(max(days_available, min_days), max_days)
    logger.info(f"Data available for {days_available} days. Training will use data from the last {training_days} days.")

    from data.database import get_top_markets_by_trading_value
    exclude = tuple(t.upper() for t in getattr(config.Data, "DYNAMIC_UNIVERSE_EXCLUDE", []))
    top_n = int(getattr(config.Data, "DYNAMIC_UNIVERSE_TOP_N", 100))
    dynamic = get_top_markets_by_trading_value(limit=top_n, hours=24)
    dynamic = [m for m in dynamic if not any(tok in m.upper() for tok in exclude)]
    target_markets = dynamic or getattr(config.Data, "TRAIN_COINS_FALLBACK", config.Data.MARKET_INDEX_COINS)
    logger.info(f"Dynamic universe: {len(target_markets)} markets for training")

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
                logger.error(f"collect_market_data failed for {market}: {e}")
                if args.offline_ok:
                    logger.warning("--offline_ok enabled: continuing despite collection failure.")
                    continue
                raise
            time.sleep(0.5)

    trainer.run(tune=args.tune, epochs=args.epochs)


def run_train_pump_mode(args):
    from training import pump_trainer
    pump_trainer.run(tune=args.tune)


def run_find_pumps_mode(*, display_pump_candidates, save_pump_predictions_to_csv):
    from inference import pump_predictor
    potential_pumps = pump_predictor.run()
    display_pump_candidates(potential_pumps)
    save_pump_predictions_to_csv(potential_pumps)


def run_backtest_mode(args):
    from training import evaluator
    stride = int(os.getenv("AETHER_BACKTEST_STRIDE_HOURS", "1") or 1)
    summary_tag = str(os.getenv("AETHER_BACKTEST_SUMMARY_TAG", "main") or "main").strip() or "main"
    success = evaluator.run(days_to_backtest=args.days, stride_hours=stride, summary_tag=summary_tag)
    if not success:
        raise SystemExit(1)


def run_quick_recommend_mode(*, logger, collect_recent_market_data, attach_strategy):
    logger.info("Starting quick recommendation run (no training)...")
    trending_markets = screener.get_trending_markets(mode='live')
    if trending_markets:
        logger.info(f"Collecting latest data for {len(trending_markets)} trending markets...")
        collect_recent_market_data(trending_markets, days=30, sleep_sec=0.5, continue_on_error=False)
        logger.info("Making predictions with the existing model...")
        recommender.run(
            predictions=attach_strategy(predictor.run(markets=trending_markets), "trending"),
            mode='live'
        )
    else:
        logger.info("No trending markets found today.")
    logger.info("Quick recommendation run finished.")


def run_explain_mode(args, *, logger):
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
