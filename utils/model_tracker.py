"""
Model Performance Tracker for Weighted Ensemble Voting.
Tracks rolling accuracy of each ensemble model to weight predictions dynamically.
"""
import numpy as np
from collections import deque
import json
import os
from utils.logger import logger


class ModelPerformanceTracker:
    """
    Tracks the rolling prediction accuracy of each ensemble model.
    Used to weight model predictions based on recent performance.
    
    Example:
        tracker = ModelPerformanceTracker(n_models=5, window=50)
        
        # After each prediction is validated
        tracker.update(model_id=2, was_correct=True)
        
        # Get weights for weighted average
        weights = tracker.get_weights()  # [0.18, 0.22, 0.25, 0.18, 0.17]
    """
    
    def __init__(self, n_models: int = 5, window: int = 50, save_path: str = "models/model_performance.json"):
        """
        Args:
            n_models: Number of ensemble models to track
            window: Size of rolling window for accuracy calculation
            save_path: Path to persist performance data
        """
        self.n_models = n_models
        self.window = window
        self.save_path = save_path
        
        # Initialize accuracy tracking for each model
        self.accuracies = {i: deque(maxlen=window) for i in range(n_models)}
        
        # Load existing data if available
        self._load()
    
    def update(self, model_id: int, was_correct: bool):
        """
        Record a prediction outcome for a specific model.
        
        Args:
            model_id: Index of the model (0 to n_models-1)
            was_correct: Whether the model's direction prediction was correct
        """
        if 0 <= model_id < self.n_models:
            self.accuracies[model_id].append(1 if was_correct else 0)
        else:
            logger.warning(f"ModelPerformanceTracker: Invalid model_id {model_id}")
    
    def update_batch(self, model_results: list):
        """
        Update multiple models at once.
        
        Args:
            model_results: List of (model_id, was_correct) tuples
        """
        for model_id, was_correct in model_results:
            self.update(model_id, was_correct)
    
    def get_weights(self, min_samples: int = 10, default_weight: float = 0.5) -> np.ndarray:
        """
        Calculate normalized weights based on recent accuracy.
        
        Args:
            min_samples: Minimum samples required to use actual accuracy
            default_weight: Default weight if insufficient samples
            
        Returns:
            Normalized weight array summing to 1.0
        """
        weights = []
        for i in range(self.n_models):
            if len(self.accuracies[i]) >= min_samples:
                accuracy = np.mean(self.accuracies[i])
            else:
                accuracy = default_weight  # Use default if not enough data
            weights.append(accuracy)
        
        weights = np.array(weights)
        
        # Handle edge case where all weights are zero
        if weights.sum() == 0:
            return np.ones(self.n_models) / self.n_models
        
        # Normalize to sum to 1.0
        return weights / weights.sum()
    
    def get_stats(self) -> dict:
        """
        Get detailed statistics for logging/debugging.
        """
        stats = {}
        for i in range(self.n_models):
            n_samples = len(self.accuracies[i])
            accuracy = np.mean(self.accuracies[i]) if n_samples > 0 else 0.0
            stats[f"model_{i}"] = {
                "n_samples": n_samples,
                "accuracy": round(accuracy, 4),
                "recent_10": list(self.accuracies[i])[-10:] if n_samples > 0 else []
            }
        return stats

    def get_model_metadata(self) -> dict:
        """
        Get static metadata including specialization and file modification time.
        """
        from datetime import datetime
        
        roles = [
            "Trend Following (추세 추종)",
            "Mean Reversion (평균 회귀)",
            "Volatility Breakout (변동성)",
            "Pattern Recognition (패턴)",
            "Market Neutral (시장 중립)"
        ]
        
        metadata = {}
        for i in range(self.n_models):
            # 1. Determine Role
            role = roles[i] if i < len(roles) else f"Ensemble Node {i+1}"
            
            # 2. Get File Modification Time
            # Use absolute path to ensure detection
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(project_root, "models", f"model_{i+1}.pth")
            
            if os.path.exists(model_path):
                timestamp = os.path.getmtime(model_path)
                dt = datetime.fromtimestamp(timestamp)
                # Format: YYYY-MM-DD HH:MM
                last_updated = dt.strftime("%Y-%m-%d %H:%M")
            else:
                last_updated = "Not Trained"
                # Check for backup/alternative names if primary missing?
                if i == 0 and os.path.exists(os.path.join(project_root, "models", "model_1.pth")):
                     pass # Already checked above
                
            metadata[f"model_{i}"] = {
                "role": role,
                "last_updated": last_updated
            }
        return metadata

    def save(self):
        """Persist tracker state to disk."""
        data = {
            "n_models": self.n_models,
            "window": self.window,
            "accuracies": {str(k): list(v) for k, v in self.accuracies.items()}
        }
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        with open(self.save_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"ModelPerformanceTracker saved to {self.save_path}")
    
    def _load(self):
        """Load tracker state from disk if exists."""
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, 'r') as f:
                    data = json.load(f)
                for k, v in data.get("accuracies", {}).items():
                    model_id = int(k)
                    if model_id < self.n_models:
                        self.accuracies[model_id] = deque(v, maxlen=self.window)
                logger.info(f"ModelPerformanceTracker loaded from {self.save_path}")
            except Exception as e:
                logger.warning(f"Failed to load ModelPerformanceTracker: {e}")
    
    def reset(self):
        """Clear all tracking data."""
        for i in range(self.n_models):
            self.accuracies[i].clear()
        logger.info("ModelPerformanceTracker reset.")


# Global singleton instance
_TRACKER_INSTANCE = None

def get_tracker(n_models: int = 5) -> ModelPerformanceTracker:
    """Get or create the global tracker instance."""
    global _TRACKER_INSTANCE
    if _TRACKER_INSTANCE is None:
        _TRACKER_INSTANCE = ModelPerformanceTracker(n_models=n_models)
    return _TRACKER_INSTANCE
