
import torch
import numpy as np
import pandas as pd
import argparse
import os
import sys
from pathlib import Path
from scipy.stats import spearmanr
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import config
from utils.logger import logger
from data.preprocessor import get_processed_data_for_training, get_market_index
from utils.metrics import calculate_ece

def set_dropout_p(model, p):
    """Recursively set the dropout probability for all nn.Dropout layers in a model."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = p

def tune_mc_dropout(model_path: str):
    logger.info("=== MC Dropout Tuning Script Start ===")

    # 1. Load Model
    try:
        model = torch.load(model_path, map_location=config.DEVICE, weights_only=False)
        logger.info(f"Successfully loaded model from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # 2. Load a fixed validation dataset (using a subset for speed)
    logger.info("Loading validation data...")
    market_index_df = get_market_index()
    X, y, _ = get_processed_data_for_training(config.TARGET_MARKETS[0], market_index_df)
    if X is None:
        logger.error("Failed to load data.")
        return
    
    # Use last 20% as a fixed validation set
    val_split_idx = int(len(X) * 0.8)
    X_val, y_val = X[val_split_idx:], y[val_split_idx:]
    logger.info(f"Using a fixed validation set of {len(X_val)} samples.")

    # 3. Define search grid
    dropout_p_values = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    n_inferences_values = [20, 30, 50, 80, 100]
    results = []

    # 4. Run Grid Search
    for p in tqdm(dropout_p_values, desc="Dropout Probabilities"):
        logger.info(f"--- Setting Dropout p = {p} ---")
        set_dropout_p(model, p)
        
        for n in tqdm(n_inferences_values, desc=f"N_Inferences (p={p})", leave=False):
            model.train() # Ensure dropout is active
            # Set BatchNorm layers to eval mode
            for module in model.modules():
                if isinstance(module, torch.nn.BatchNorm1d):
                    module.eval()

            all_errors = []
            all_uncertainties = []
            all_confidences = []
            y_true_binary = []

            with torch.no_grad():
                for i in range(len(X_val)):
                    x_sample = torch.FloatTensor(X_val[i:i+1]).to(config.DEVICE)
                    y_true = y_val[i]
                    
                    mc_preds = [model(x_sample)[0].cpu().numpy() for _ in range(n)]
                    mc_preds = np.array(mc_preds)
                    
                    mean_pred = mc_preds.mean(axis=0)
                    std_pred = mc_preds.std(axis=0)
                    
                    error = np.mean(np.abs(mean_pred - y_true))
                    uncertainty = np.sum(std_pred)
                    confidence = 1 / (1 + uncertainty)

                    all_errors.append(error)
                    all_uncertainties.append(uncertainty)
                    all_confidences.append(confidence)
                    y_true_binary.append(1 if np.sum(y_true) > 0 else 0)

            # Calculate overall metrics for this combination
            correlation, _ = spearmanr(all_errors, all_uncertainties)
            correlation = 0 if np.isnan(correlation) else correlation
            
            ece = calculate_ece(np.array(y_true_binary), np.array(all_confidences), n_bins=15)
            avg_uncertainty = np.mean(all_uncertainties)

            results.append({
                'dropout_p': p,
                'n_inferences': n,
                'spearman_corr': correlation,
                'ece': ece,
                'avg_uncertainty': avg_uncertainty
            })

    # 5. Save and Report Results
    if not results:
        logger.error("Tuning produced no results.")
        return

    results_df = pd.DataFrame(results)
    
    # Create results directory if it doesn't exist
    output_dir = "analysis/results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "mc_dropout_tuning.csv")
    
    results_df.to_csv(output_path, index=False)
    logger.info(f"Tuning results saved to {output_path}")

    # Log summary of top 5 combinations
    top_5 = results_df.sort_values(by='spearman_corr', ascending=False).head(5)
    logger.info("--- Top 5 MC Dropout Combinations (by Spearman Correlation) ---")
    logger.info(top_5.to_string())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune MC Dropout parameters.")
    parser.add_argument("--model_path", type=str, default=config.MODEL_PATH, help="Path to the trained model file.")
    args = parser.parse_args()
    
    tune_mc_dropout(model_path=args.model_path)
