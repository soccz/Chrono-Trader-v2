from utils.logger import logger
from data import collector, database
from training import trainer
from inference import recommender, predictor
from utils.config import config


def run_daily_pipeline():
    """
    Executes the complete daily pipeline:
    1. Initialize DB (if not exists)
    2. Collect recent data
    3. Incrementally train/update the model
    4. Generate predictions
    5. Generate and save new recommendations
    """
    logger.info("=========================================")
    logger.info("=== STARTING FULL DAILY PIPELINE RUN ====")
    logger.info("=========================================")

    try:
        # Step 1: Initialize Database (safe to run daily)
        logger.info("--- Step 1: Initializing Database ---")
        database.init_db()

        # Step 2: Collect recent data (e.g., last 2 days to be safe)
        logger.info("--- Step 2: Collecting Recent Data ---")
        collector.run(days=2)

        # Step 3: Update model with new data
        logger.info("--- Step 3: Training/Updating Model ---")
        trainer.run()

        # Step 4: Generate predictions
        logger.info("--- Step 4: Generating Predictions ---")
        predictions = predictor.run(markets=config.TARGET_MARKETS)

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