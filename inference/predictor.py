import torch
import numpy as np
import pandas as pd
import glob
import os
import json
import torch.serialization
import torch.nn as nn
import models.hybrid_model
import models.transformer_encoder
from models.hybrid_model import build_model
from scipy.stats import zscore

from utils.config import config
from utils.logger import logger
from data import database
from data.preprocessor import get_intermediate_data, create_final_sequences_and_scale, get_market_index
from utils.metrics import calculate_ece
from sklearn.calibration import calibration_curve

from tslearn.metrics import soft_dtw

def get_pattern_similarity(pattern1: np.ndarray, pattern2: np.ndarray) -> float:
    """
    Calculates the similarity between two price change patterns using Soft Dynamic Time Warping (soft-DTW)
    with z-score normalization.
    """
    if len(pattern1) == 0 or len(pattern2) == 0:
        return float('inf')

    # Z-score normalization to compare shapes
    pattern1_norm = zscore(pattern1)
    pattern2_norm = zscore(pattern2)

    # Reshape for tslearn: (sz, d)
    p1_reshaped = pattern1_norm.reshape(-1, 1)
    p2_reshaped = pattern2_norm.reshape(-1, 1)

    # Use gamma=0.1 as a starting point
    distance = soft_dtw(p1_reshaped, p2_reshaped, gamma=0.1)
    
    return distance



def run(markets: list, market_index_df: pd.DataFrame = None, historical_df: pd.DataFrame = None):
    """
    Makes ensembled, probabilistic predictions for a given list of markets.
    Can accept pre-loaded dataframes for efficient backtesting.
    """
    logger.info(f"--- Making ensembled predictions for {len(markets)} markets ---")

    model_paths = glob.glob(os.path.join("models", "model_*.pth"))
    if not model_paths:
        logger.error("No trained models found in /models directory. Please run training first.")
        return []

    logger.info(f"Found {len(model_paths)} ensemble models.")
    models = []
    for path in model_paths:
        try:
            loaded_obj = torch.load(path, map_location=config.DEVICE, weights_only=False)

            if isinstance(loaded_obj, nn.Module):
                model = loaded_obj
            else:
                state_dict = loaded_obj
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']

                encoder_weight_key = None
                if 'transformer_encoder.encoder_layer.weight' in state_dict:
                    encoder_weight_key = 'transformer_encoder.encoder_layer.weight'
                else:
                    for key in state_dict.keys():
                        if key.endswith('encoder_layer.weight'):
                            encoder_weight_key = key
                            break
                if encoder_weight_key is None:
                    raise KeyError('encoder_layer.weight')

                encoder_weight = state_dict[encoder_weight_key]
                input_dim = encoder_weight.shape[1]
                d_model_loaded = encoder_weight.shape[0]

                decoder_weight_key = None
                for key in reversed(state_dict.keys()):
                    if key.startswith('decoder') and key.endswith('weight'):
                        decoder_weight_key = key
                        break
                if decoder_weight_key is None:
                    raise KeyError('decoder weight')
                output_dim = state_dict[decoder_weight_key].shape[0]

                noise_dim = getattr(config, 'GAN_NOISE_DIM', 32)
                n_layers = getattr(config, 'N_LAYERS', 3)
                n_heads = getattr(config, 'N_HEADS', 8)
                dropout_p = getattr(config, 'DROPOUT_P', 0.1)

                model = build_model(
                    d_model=d_model_loaded,
                    n_heads=n_heads,
                    n_layers=n_layers,
                    input_dim=input_dim,
                    noise_dim=noise_dim,
                    output_dim=output_dim,
                    dropout_p=dropout_p
                )
                model.load_state_dict(state_dict)

            model = model.to(config.DEVICE)
            model.eval()
            models.append(model)
            logger.info(f"Successfully loaded model from {path}")
        except Exception as e:
            logger.error(f"Failed to load model from {path}. Error: {e}")
            return []

    if not models:
        logger.error("No models were successfully loaded. Aborting prediction.")
        return []

    if market_index_df is None:
        logger.info("Market index not provided, calculating fresh...")
        market_index_df = get_market_index()
    
    # --- Refactoring for Shrunk Beta ---
    intermediate_data = {}
    for market in markets:
        df, scaler = get_intermediate_data(market, market_index_df, historical_df=historical_df)
        if df is not None:
            intermediate_data[market] = {'df': df, 'scaler': scaler}

    if not intermediate_data:
        logger.warning("No markets had sufficient data for pre-computation.")
        return []

    beta_series_list = [data['df']['beta'].rename(market) for market, data in intermediate_data.items()]
    all_betas_df = pd.concat(beta_series_list, axis=1)
    cs_beta_mean = all_betas_df.mean(axis=1)
    logger.info("Calculated cross-sectional mean beta for shrinkage.")

    all_predictions = []
    for market, data in intermediate_data.items():
        df = data['df']
        scaler = data['scaler']

        df['beta'] = 0.5 * df['beta'] + 0.5 * cs_beta_mean
        df.loc[:, 'beta'] = df['beta'].fillna(0)

        X, y, scaler = create_final_sequences_and_scale(df, scaler)
        
        if X is None:
            continue
        
        last_sequence = X[-1]
        sequence_tensor = torch.FloatTensor([last_sequence]).to(config.DEVICE)

        with torch.no_grad():
            individual_patterns = []
            for model in models:
                model.eval()
                predicted_pattern = model(sequence_tensor)[0].cpu().numpy()
                individual_patterns.append(predicted_pattern)
        
        individual_patterns = np.array(individual_patterns)
        final_pattern = np.mean(individual_patterns, axis=0)
        final_uncertainty = np.sum(np.std(individual_patterns, axis=0))

        current_price = df.iloc[-1]['close']

        all_predictions.append({
            "market": market,
            "predicted_pattern": final_pattern,
            "uncertainty": final_uncertainty,
            "current_price": current_price,
            "individual_patterns": individual_patterns 
        })
        logger.info(f"Ensemble prediction for {market} generated (Uncertainty: {final_uncertainty:.6f}).")

    return all_predictions
