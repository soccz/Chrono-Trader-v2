import pandas as pd
import numpy as np
import pandas_ta_classic as ta
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import os

from utils.config import config
from utils.logger import logger
from data.database import load_data, get_db_connection

FUTURE_WINDOW_SIZE = 6
MAJOR_COINS = ['KRW-BTC', 'KRW-ETH'] # Coins to build the market index

def get_market_weights(start_date=None, end_date=None) -> dict:
    """
    Dynamically calculates market weights for major coins based on trading value in a given period.
    """
    logger.info("Dynamically calculating market weights...")
    weights = {}
    total_value = 0
    
    try:
        # Define the time window for the query
        if end_date and start_date:
             # Use the provided date range
             date_filter = f"AND timestamp BETWEEN '{start_date}' AND '{end_date}'"
        else:
             # Default to last 30 days if no range is provided
             date_filter = "AND timestamp >= date('now', '-30 days')"

        for coin in MAJOR_COINS:
            query = f"""
                SELECT AVG(close * volume) 
                FROM crypto_data 
                WHERE market = '{coin}' 
                {date_filter}
            """
            avg_value_df = load_data(query)
            if avg_value_df.empty or avg_value_df.iloc[0, 0] is None:
                logger.warning(f"Could not calculate average trading value for {coin}. It will be excluded from dynamic weighting.")
                avg_value = 0
            else:
                avg_value = avg_value_df.iloc[0, 0]
            
            weights[coin] = avg_value
            total_value += avg_value

        if total_value > 0:
            for coin in weights:
                weights[coin] = weights[coin] / total_value
            logger.info(f"Calculated dynamic weights: {weights}")
            return weights
            
    except Exception as e:
        logger.error(f"Failed to calculate dynamic market weights: {e}")

    # Fallback to default weights if calculation fails
    logger.warning("Using default market cap weights (70/30).")
    return {'KRW-BTC': 0.7, 'KRW-ETH': 0.3}

def get_market_index(start_date=None, end_date=None) -> pd.DataFrame:
    """
    Calculates a market-cap weighted market index based on major coins for a given period.
    """
    logger.info(f"Calculating market index from {MAJOR_COINS}...")
    
    # Pass date range to get weights for the specific period
    market_weights = get_market_weights(start_date=start_date, end_date=end_date)
    index_df = pd.DataFrame()
    
    try:
        # Define the time window for the query
        if end_date and start_date:
             date_filter = f"AND timestamp BETWEEN '{start_date}' AND '{end_date}'"
        else:
             date_filter = "" # No filter, get all data

        for coin in MAJOR_COINS:
            query = f"SELECT timestamp, close FROM crypto_data WHERE market = '{coin}' {date_filter} ORDER BY timestamp ASC"
            df = load_data(query)
            if df.empty:
                logger.warning(f"No data for {coin} to calculate market index for the given period.")
                continue
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize('UTC')
            df.set_index('timestamp', inplace=True)
            df[f'{coin}_pct_change'] = df['close'].pct_change()
            
            if index_df.empty:
                index_df = df[[f'{coin}_pct_change']]
            else:
                index_df = index_df.join(df[[f'{coin}_pct_change']], how='outer')
        
        if index_df.empty or not all(f'{c}_pct_change' in index_df.columns for c in MAJOR_COINS if c in market_weights):
            logger.warning("Could not calculate market index, data missing for major coins with calculated weights.")
            return pd.DataFrame()

        # Fill NaNs before calculation
        index_df.fillna(0, inplace=True)

        # Calculate weighted average using dynamic weights
        index_df['market_index_return'] = 0.0
        for coin, weight in market_weights.items():
             if f'{coin}_pct_change' in index_df.columns:
                  index_df['market_index_return'] += index_df[f'{coin}_pct_change'] * weight
        
        logger.info("Market-cap weighted index calculated successfully.")
        return index_df[['market_index_return']]

    except Exception as e:
        logger.error(f"Failed to calculate market index: {e}")
        return pd.DataFrame()

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates technical indicators for the given data using pandas-ta."""
    # Calculate all indicators and store them temporarily
    rsi = ta.rsi(df['close'])
    macd = ta.macd(df['close'])
    adx = ta.adx(df['high'], df['low'], df['close'])
    obv = ta.obv(df['close'], df['volume'])
    bbands = ta.bbands(df['close'], length=20)
    volume_ma = ta.sma(df['volume'], length=20)

    # Assign to the dataframe with the exact column names the project expects
    df['rsi'] = rsi
    if macd is not None and not macd.empty:
        df['macd'] = macd['MACD_12_26_9']
        df['macdsignal'] = macd['MACDs_12_26_9']
        df['macdhist'] = macd['MACDh_12_26_9']
    else:
        df['macd'] = 0
        df['macdsignal'] = 0
        df['macdhist'] = 0

    if adx is not None and not adx.empty:
        df['adx'] = adx['ADX_14']
    else:
        df['adx'] = 0

    df['obv'] = obv
    if bbands is not None and not bbands.empty and bbands.shape[1] >= 3:
        # Access by position for robustness against naming changes
        df['bb_lower'] = bbands.iloc[:, 0]  # Lower band
        df['bb_middle'] = bbands.iloc[:, 1] # Middle band
        df['bb_upper'] = bbands.iloc[:, 2]  # Upper band
    else:
        df['bb_lower'] = 0
        df['bb_middle'] = 0
        df['bb_upper'] = 0

    df['volume_ma'] = volume_ma

    df.fillna(0, inplace=True)
    return df

def create_sequences(data, sequence_length, future_window):
    xs, ys = [], []
    for i in range(len(data) - sequence_length - future_window):
        x = data[i:(i + sequence_length)]
        y = data[(i + sequence_length):(i + sequence_length + future_window), 0]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def get_intermediate_data(market: str, market_index_df: pd.DataFrame, historical_df: pd.DataFrame = None):
    """
    Loads and preprocesses data for a market up to the point of feature calculation.
    If historical_df is provided, it uses that dataframe; otherwise, it queries the database.
    Ensures all timezone information is consistent (UTC) before processing.
    """
    logger.info(f"Processing intermediate data for market: {market}")
    
    if historical_df is not None:
        df = historical_df[historical_df['market'] == market].copy()
        # The historical_df from the evaluator is already tz-aware and indexed.
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
    else:
        query = f"SELECT * FROM crypto_data WHERE market = '{market}' ORDER BY timestamp ASC"
        df = load_data(query)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df.index = df.index.tz_localize('UTC') # Always localize after loading

    if df.empty or len(df) < 100:
        logger.warning(f"Not enough data for market {market} to process (< 100 records). Skipping.")
        return None, None

    # Ensure market_index_df is also tz-aware before joining
    if market_index_df.index.tz is None:
        market_index_df.index = market_index_df.index.tz_localize('UTC')

    df = calculate_technical_indicators(df)

    # Now the join should be safe as both indexes are tz-aware
    df = df.join(market_index_df, how='left')
    df['market_index_return'] = df['market_index_return'].fillna(0)

    df['coin_return'] = df['close'].pct_change().fillna(0)
    beta_span = 720
    ewm_cov = df['coin_return'].ewm(span=beta_span, adjust=False).cov(df['market_index_return'])
    ewm_var = df['market_index_return'].ewm(span=beta_span, adjust=False).var()
    df['beta'] = (ewm_cov / ewm_var).fillna(0).clip(-3, 3)
    df['alpha'] = (df['coin_return'] - (df['beta'] * df['market_index_return'])).fillna(0)

    df['future_pct_change'] = (df['close'].shift(-1) - df['close']) / df['close']
    df.fillna(0, inplace=True)
    
    scaler = MinMaxScaler()

    return df, scaler

def create_final_sequences_and_scale(df: pd.DataFrame, scaler: MinMaxScaler):
    """
    Takes a dataframe with all features, scales them, and creates sequences.
    """
    features_to_scale = [
        'close', 'volume', 'rsi', 'macd', 'macdsignal', 'macdhist', 'adx', 'obv', 
        'market_index_return', 'bb_upper', 'bb_middle', 'bb_lower', 'volume_ma',
        'alpha', 'beta'
    ]
    
    features_to_scale = [f for f in features_to_scale if f in df.columns]

    scaled_features = scaler.fit_transform(df[features_to_scale])

    data_for_sequences = np.c_[df[['future_pct_change']], scaled_features]

    X, y = create_sequences(data_for_sequences, config.SEQUENCE_LENGTH, FUTURE_WINDOW_SIZE)
    
    if X.shape[0] == 0:
        logger.warning(f"Not enough data to create any sequences after processing. Skipping.")
        return None, None, None

    X = X[:, :, 1:]

    logger.info(f"Data scaling and sequencing complete. Shape of X: {X.shape}, Shape of y: {y.shape}")
    
    return X, y, scaler

def get_processed_data_for_training(market: str, market_index_df: pd.DataFrame):
    """Wrapper function for training that calls the refactored processing steps."""
    # Pass historical_df=None to ensure it queries the DB for fresh training data
    df, scaler = get_intermediate_data(market, market_index_df, historical_df=None)
    if df is None:
        return None, None, None
    
    # For training, we don't apply shrinkage, so we pass the df directly
    return create_final_sequences_and_scale(df, scaler)

def get_recent_pattern(market: str, current_time: datetime, hours: int = 24) -> np.ndarray:
    """
    Loads recent historical data for a market and returns its price change pattern.
    Returns a 1D numpy array of percentage changes for the last 'hours' period.
    """
    # Need hours + 1 data points to calculate 'hours' percentage changes
    query = f"SELECT timestamp, close FROM crypto_data WHERE market = '{market}' AND timestamp <= '{current_time}' ORDER BY timestamp DESC LIMIT {hours + 1}"
    df = load_data(query)
    
    if df.empty or len(df) < (hours + 1):
        # logger.warning(f"Not enough recent data for {market} to form a {hours}-hour pattern.")
        return np.array([]) # Return empty array if not enough data points

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='timestamp', ascending=True) # Ensure chronological order

    pattern = df['close'].pct_change().dropna().values
    
    if len(pattern) != hours:
        # This should ideally not happen if len(df) >= hours + 1 and no NaNs
        return np.array([])

    return pattern # Return the 'hours' percentage changes

def get_historical_success_patterns(
    cache_path="data/success_patterns.npy",
    recalculate=False,
    window_size=6,
    min_return=0.15
):
    """
    Finds and caches historical price patterns that led to significant returns.
    A "success pattern" is defined as a sequence of `window_size` hourly returns
    that resulted in a total return of at least `min_return`.
    """
    if os.path.exists(cache_path) and not recalculate:
        logger.info(f"Loading cached success patterns from {cache_path}")
        return np.load(cache_path)

    logger.info("Calculating and caching historical success patterns...")
    all_markets_df = load_data("SELECT DISTINCT market FROM crypto_data WHERE market LIKE 'KRW-%'")
    all_markets = all_markets_df['market'].tolist() if not all_markets_df.empty else []
    
    success_patterns = []

    for market in all_markets:
        query = f"SELECT timestamp, close FROM crypto_data WHERE market = '{market}' ORDER BY timestamp ASC"
        df = load_data(query)
        
        if len(df) < window_size:
            continue

        df['coin_return'] = df['close'].pct_change()
        df['future_window_return'] = (df['close'].shift(-window_size) / df['close']) - 1
        
        success_indices = df[df['future_window_return'] >= min_return].index
        
        for index in success_indices:
            try:
                start_idx = df.index.get_loc(index)
                pattern = df.iloc[start_idx : start_idx + window_size]['coin_return'].values
                if len(pattern) == window_size and not np.isnan(pattern).any():
                    success_patterns.append(pattern)
            except KeyError:
                continue

    if not success_patterns:
        logger.warning("No historical success patterns found. DTW filter will be ineffective.")
        return np.array([])

    patterns_array = np.array(success_patterns)
    np.save(cache_path, patterns_array)
    logger.info(f"Found and cached {len(patterns_array)} success patterns to {cache_path}")
    
    return patterns_array

# --- Functions for Pump Prediction Dataset ---

def create_pump_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates features that might indicate a future pump."""
    # Volume Spikes (volume change vs. rolling average)
    df['volume_pct_change'] = df['volume'].pct_change()
    df['volume_spike_score'] = df['volume_pct_change'] / (df['volume_pct_change'].rolling(window=24).mean() + 1e-9)

    # Bollinger Band Squeeze
    if 'bb_middle' in df.columns and df['bb_middle'].notna().any():
        bb_bandwidth = (df['bb_upper'] - df['bb_lower']) / (df['bb_middle'] + 1e-9)
        # A squeeze is when the current bandwidth is at a 24-hour low
        df['squeeze_on'] = (bb_bandwidth.rolling(window=24).min() == bb_bandwidth).astype(int)
    else:
        df['squeeze_on'] = 0

    # Price momentum (rate of change over different periods)
    df['roc_3'] = ta.roc(df['close'], length=3)
    df['roc_6'] = ta.roc(df['close'], length=6)

    return df

def create_pump_labels(df: pd.DataFrame, time_horizon: int = 6) -> pd.DataFrame:
    """Creates multi-class ground truth labels for pump events."""
    # Find the max high price in the next `time_horizon` hours
    future_max_high = df['high'].rolling(window=time_horizon, min_periods=1).max().shift(-time_horizon)
    
    # Calculate the maximum percentage rise from the current close
    max_future_rise = (future_max_high / df['close']) - 1
    
    # Define conditions for each class
    # Label 0: < 10% (No Pump)
    # Label 1: 10% to 15%
    # Label 2: 15% to 20%
    # Label 3: >= 20%
    conditions = [
        (max_future_rise < 0.10),
        (max_future_rise >= 0.10) & (max_future_rise < 0.15),
        (max_future_rise >= 0.15) & (max_future_rise < 0.20),
        (max_future_rise >= 0.20)
    ]
    choices = [0, 1, 2, 3]
    
    # Use numpy.select for conditional labeling
    df['pump_label'] = np.select(conditions, choices, default=0)
    
    return df

def get_pump_dataset(days: int = None):
    """
    Generates a complete dataset for training the pump prediction model.
    If 'days' is specified, it only processes data from the last N days.
    """
    if days:
        logger.info(f"--- Starting Generation of Pump Prediction Dataset for last {days} days ---")
    else:
        logger.info("--- Starting Generation of Full Pump Prediction Dataset ---")
    
    # First, get the market index that will be joined to all individual dataframes
    market_index_df = get_market_index()

    all_markets_df = load_data("SELECT DISTINCT market FROM crypto_data WHERE market LIKE 'KRW-%'")
    all_markets = all_markets_df['market'].tolist() if not all_markets_df.empty else []

    full_dataset = []

    for market in all_markets:
        
        if days:
            query = f"SELECT * FROM crypto_data WHERE market = '{market}' AND timestamp >= date('now', '-{days} days') ORDER BY timestamp ASC"
        else:
            query = f"SELECT * FROM crypto_data WHERE market = '{market}' ORDER BY timestamp ASC"
            
        df = load_data(query)
        
        if df.empty or len(df) < 100: # Need enough data for rolling windows
            continue

        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize('UTC')
        df.set_index('timestamp', inplace=True)

        # Join the market index here
        # Both dataframes must be tz-aware for the join.
        df = df.join(market_index_df, how='left')
        df['market_index_return'] = df['market_index_return'].fillna(0)

        # 1. Calculate standard and new pump-specific features
        df = calculate_technical_indicators(df)
        df = create_pump_features(df)

        # 2. Create the pump labels
        df = create_pump_labels(df, time_horizon=6)

        # 3. Select features and labels
        feature_cols = [
            'close', 'volume', 'rsi', 'macd', 'macdsignal', 'macdhist', 'adx', 'obv',
            'bb_upper', 'bb_middle', 'bb_lower', 'volume_ma',
            'market_index_return', # Add market trend as a feature
            'volume_spike_score', 'squeeze_on', 'roc_3', 'roc_6'
        ]
        # Ensure all columns exist before trying to select them
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        df_subset = df[feature_cols + ['pump_label']].copy()
        df_subset['market'] = market # Add market identifier
        
        full_dataset.append(df_subset)

    if not full_dataset:
        logger.error("Could not generate any data for the pump dataset.")
        return

    # Combine all dataframes and save to a single file
    final_df = pd.concat(full_dataset, ignore_index=True)
    final_df.dropna(inplace=True) # Drop rows with NaNs from rolling calculations
    
    output_path = "data/pump_dataset.csv"
    final_df.to_csv(output_path, index=False)
    logger.info(f"Pump prediction dataset created successfully with {len(final_df)} samples.")
    logger.info(f"Saved to {output_path}")

    return final_df
