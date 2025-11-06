import pandas as pd
import joblib
import os

from data.database import load_data
from data.preprocessor import calculate_technical_indicators, create_pump_features, get_market_index
from utils.logger import logger

def run():
    """Loads the trained multi-class pump classifier and predicts pump distribution."""
    logger.info("--- Finding Potential Pump Candidates (Multi-class) ---")

    # 1. Load the trained model
    model_path = os.path.join("models", "pump_classifier.joblib")
    if not os.path.exists(model_path):
        logger.error(f"Pump classifier model not found at {model_path}. Please train it first using --mode train-pump.")
        return []
    
    try:
        model = joblib.load(model_path)
    except Exception as e:
        logger.error(f"Failed to load the pump classifier model: {e}")
        return []

    # 2. Get market data
    all_markets_df = load_data("SELECT DISTINCT market FROM crypto_data WHERE market LIKE 'KRW-%'")
    all_markets = all_markets_df['market'].tolist() if not all_markets_df.empty else []
    market_index_df = get_market_index() # Get market index

    if not all_markets:
        logger.warning("No markets found in the database.")
        return []

    potential_pumps = []

    # 3. Iterate through each market to get the latest features and predict
    for market in all_markets:
        query = f"SELECT * FROM crypto_data WHERE market = '{market}' ORDER BY timestamp DESC LIMIT 100"
        df = load_data(query)

        if df.empty or len(df) < 100:
            continue

        df = df.iloc[::-1].reset_index(drop=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df.index = df.index.tz_localize('UTC') # Make index tz-aware (UTC)

        # 4. Create features
        df = df.join(market_index_df, how='left') # Join market index
        df['market_index_return'] = df['market_index_return'].fillna(0)
        df = calculate_technical_indicators(df)
        df = create_pump_features(df)
        
        latest_features = df.tail(1)

        if latest_features.empty:
            continue

        # 5. Select features
        feature_cols = [
            'close', 'volume', 'rsi', 'macd', 'macdsignal', 'macdhist', 'adx', 'obv',
            'bb_upper', 'bb_middle', 'bb_lower', 'volume_ma',
            'market_index_return', # Added market trend as a feature
            'volume_spike_score', 'squeeze_on', 'roc_3', 'roc_6'
        ]
        feature_cols = [col for col in feature_cols if col in latest_features.columns]
        X_latest = latest_features[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

        # 6. Predict probabilities for all classes
        try:
            probabilities = model.predict_proba(X_latest)[0]
            total_pump_prob = probabilities[1:].sum()

            if total_pump_prob > 0.2: # Example threshold
                potential_pumps.append({
                    'market': market,
                    'current_price': latest_features['close'].iloc[0],
                    'probabilities': probabilities,
                    'total_pump_prob': total_pump_prob
                })
        except Exception as e:
            logger.error(f"Prediction failed for {market}: {e}")
            continue

    # 7. Sort by total pump probability and return
    potential_pumps.sort(key=lambda x: x['total_pump_prob'], reverse=True)
    
    logger.info(f"--- Found {len(potential_pumps)} potential pump candidates ---")
    return potential_pumps

if __name__ == '__main__':
    # For direct testing of this script
    from data.database import init_db
    init_db()
    results = run()
    print(results)
