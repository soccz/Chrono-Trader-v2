
import logging
import os
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime
from .config import config

def setup_logger():
    """Set up the logger for the project."""
    if not os.path.exists(config.General.LOG_DIR):
        os.makedirs(config.General.LOG_DIR)

    log_file = os.path.join(config.General.LOG_DIR, f"crypto_predictor_{datetime.now().strftime('%Y-%m-%d')}.log")

    logger = logging.getLogger("CryptoPredictor")
    logger.setLevel(logging.INFO)

    # Avoid adding handlers multiple times
    if logger.hasHandlers():
        return logger

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Console Handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler (rotates daily)
    fh = TimedRotatingFileHandler(log_file, when="midnight", interval=1, backupCount=30)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

logger = setup_logger()
