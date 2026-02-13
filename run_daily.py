from utils.logger import logger
from utils.auto_retrain import get_recent_accuracy, check_and_trigger_retrain, ACCURACY_THRESHOLD
from data import collector, database
from training import trainer
from inference import recommender, predictor
from utils.config import config


def run_daily_pipeline():
    """
    Executes the complete daily pipeline:
    0. [NEW] Check model performance - trigger Full Retrain if accuracy is too low
    1. Initialize DB (if not exists)
    2. Collect recent data
    3. Incrementally train/update the model (OR Full Retrain if triggered)
    4. Generate predictions
    5. Generate and save new recommendations
    """
    logger.info("=========================================")
    logger.info("=== STARTING FULL DAILY PIPELINE RUN ====")
    logger.info("=========================================")

    try:
        # Step 0: Performance Check - Decide Full Retrain vs Fine-tuning
        logger.info("--- Step 0: Checking Model Performance ---")
        current_accuracy = get_recent_accuracy()
        needs_full_retrain = False
        
        if current_accuracy is not None:
            logger.info(f"📊 Recent Model Accuracy: {current_accuracy:.1f}%")
            if current_accuracy < ACCURACY_THRESHOLD:
                logger.warning(f"⚠️ Accuracy {current_accuracy:.1f}% is below threshold {ACCURACY_THRESHOLD}%!")
                needs_full_retrain = check_and_trigger_retrain(current_accuracy)
            else:
                logger.info(f"✅ Accuracy is healthy. Proceeding with Fine-tuning.")
        else:
            logger.info("ℹ️ Could not calculate recent accuracy (no past data). Proceeding with Fine-tuning.")

        # If Full Retrain was triggered, skip the rest (it runs in background)
        if needs_full_retrain:
            logger.info("🚀 Full Retrain triggered in background. Skipping rest of daily pipeline.")
            logger.info("   Next daily run will use newly trained model.")
            return

        # Step 1: Initialize Database (safe to run daily)
        logger.info("--- Step 1: Initializing Database ---")
        database.init_db()

        # Step 2: Collect recent data (e.g., last 2 days to be safe)
        logger.info("--- Step 2: Collecting Recent Data ---")
        collector.run(days=2)

        # Step 3: Update model with new data (Fine-tuning)
        logger.info("--- Step 3: Training/Updating Model (Fine-tuning) ---")
        trainer.run(markets=config.Data.TRAIN_COINS)

        # Step 4: Generate predictions
        logger.info("--- Step 4: Generating Predictions ---")
        predictions = predictor.run(markets=config.Data.TRAIN_COINS)

        # Step 5: Generate new recommendations
        logger.info("--- Step 5: Generating Recommendations ---")
        recommender.run(predictions=predictions)

        logger.info("=======================================")
        logger.info("=== DAILY PIPELINE COMPLETED SUCCESSFULLY ===")
        logger.info("=======================================")

    except Exception as e:
        logger.critical(f"An error occurred during the daily pipeline: {e}", exc_info=True)
        logger.info("=======================================")
        logger.info("====== DAILY PIPELINE FAILED ========")
        logger.info("=======================================")

if __name__ == "__main__":
    run_daily_pipeline()
