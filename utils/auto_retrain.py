"""
Auto-Retraining Module
Monitors model performance and triggers retraining when accuracy drops below threshold.
"""
import os
import subprocess
from datetime import datetime, timedelta
from utils.logger import logger

# Configuration
ACCURACY_THRESHOLD = 25.0  # Retrain if accuracy drops below 25%
CHECK_INTERVAL_HOURS = 24  # Check accuracy daily
LAST_RETRAIN_FILE = "logs/last_retrain.log"
MIN_RETRAIN_INTERVAL_HOURS = 48  # Don't retrain more than once every 48 hours


def get_last_retrain_time():
    """Get the timestamp of the last retraining."""
    try:
        if os.path.exists(LAST_RETRAIN_FILE):
            with open(LAST_RETRAIN_FILE, 'r') as f:
                return datetime.fromisoformat(f.read().strip())
    except Exception as e:
        logger.error(f"Error reading last retrain time: {e}")
    return None


def save_retrain_time():
    """Save the current time as the last retraining time."""
    try:
        os.makedirs(os.path.dirname(LAST_RETRAIN_FILE), exist_ok=True)
        with open(LAST_RETRAIN_FILE, 'w') as f:
            f.write(datetime.now().isoformat())
    except Exception as e:
        logger.error(f"Error saving retrain time: {e}")


def check_and_trigger_retrain(current_accuracy: float, force: bool = False):
    """
    Check if retraining is needed based on accuracy and time constraints.
    
    Args:
        current_accuracy: Recent model accuracy (0-100%)
        force: Force retraining regardless of conditions
    
    Returns:
        bool: True if retraining was triggered
    """
    logger.info(f"📊 Checking retrain conditions: Accuracy={current_accuracy:.1f}%")
    
    # Check accuracy threshold
    if current_accuracy >= ACCURACY_THRESHOLD and not force:
        logger.info(f"✅ Accuracy {current_accuracy:.1f}% is above threshold {ACCURACY_THRESHOLD}%. No retrain needed.")
        return False
    
    # Check minimum interval since last retrain
    last_retrain = get_last_retrain_time()
    if last_retrain:
        hours_since = (datetime.now() - last_retrain).total_seconds() / 3600
        if hours_since < MIN_RETRAIN_INTERVAL_HOURS and not force:
            logger.info(f"⏰ Only {hours_since:.1f}h since last retrain. Waiting for {MIN_RETRAIN_INTERVAL_HOURS}h minimum.")
            return False
    
    # Trigger retraining
    logger.warning(f"⚠️ Accuracy {current_accuracy:.1f}% below threshold {ACCURACY_THRESHOLD}%. Triggering retraining...")
    
    try:
        # Run training in background
        cmd = "python main.py --mode train --epochs 50"
        logger.info(f"🚀 Starting retrain: {cmd}")
        
        process = subprocess.Popen(
            cmd.split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        
        save_retrain_time()
        logger.info(f"✅ Retraining triggered successfully (PID: {process.pid})")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to trigger retraining: {e}")
        return False


def get_recent_accuracy():
    """
    Get the recent model accuracy from performance data.
    Returns accuracy as percentage (0-100).
    """
    try:
        from web_utils.data_loader import DataLoader
        loader = DataLoader()
        recs_df = loader.get_latest_recommendations()
        
        if recs_df is None or recs_df.empty:
            return None
        
        # Calculate accuracy based on direction correctness
        from data.collector import get_current_price
        correct = 0
        total = 0
        
        for _, row in recs_df.iterrows():
            market = row.get('market')
            signal = row.get('signal', 'Long')
            entry_price = row.get('current_price', 0)
            
            current = get_current_price(market)
            if current and entry_price > 0:
                actual_return = (current - entry_price) / entry_price
                predicted_positive = signal == 'Long'
                actual_positive = actual_return > 0
                
                if predicted_positive == actual_positive:
                    correct += 1
                total += 1
        
        if total > 0:
            return (correct / total) * 100
        return None
        
    except Exception as e:
        logger.error(f"Error calculating recent accuracy: {e}")
        return None


if __name__ == "__main__":
    # Test the module
    accuracy = get_recent_accuracy()
    if accuracy is not None:
        print(f"Recent accuracy: {accuracy:.1f}%")
        check_and_trigger_retrain(accuracy)
    else:
        print("Could not calculate accuracy")
