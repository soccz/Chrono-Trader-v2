import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas_ta_classic as ta
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import os
from scipy.stats import zscore

from utils.config import config
from utils.logger import logger
from data.database import load_data, get_db_connection

def get_or_create_pattern_memory_bank(market_index_df: pd.DataFrame, pattern_length: int = None):
    """
    Creates or retrieves a cached memory bank of historical market index patterns.
    Each pattern is a sequence of `pattern_length` market index returns.
    Returns a tuple of (patterns_array, timestamps_array).
    
    This implements the "Retrieval-Augmented" concept: we store past patterns
    so we can later find which historical period is most similar to "now".
    """
    if pattern_length is None:
        pattern_length = config.Data.SEQUENCE_LENGTH
    
    logger.info(f"Creating pattern memory bank with pattern_length={pattern_length}...")
    
    if market_index_df.empty or 'market_index_return' not in market_index_df.columns:
        logger.warning("Cannot create pattern memory bank: market_index_df is empty or missing column.")
        return np.array([]), np.array([])
    
    returns = market_index_df['market_index_return'].values
    timestamps = market_index_df.index.values
    
    patterns = []
    pattern_timestamps = []
    
    # Create overlapping patterns from the index series
    for i in range(len(returns) - pattern_length + 1):
        pattern = returns[i:i + pattern_length]
        if not np.isnan(pattern).any():
            patterns.append(pattern)
            pattern_timestamps.append(timestamps[i + pattern_length - 1])  # End timestamp of pattern
    
    if not patterns:
        logger.warning("No valid patterns could be created for memory bank.")
        return np.array([]), np.array([])
    
    memory_bank = np.array(patterns)
    memory_bank_timestamps = np.array(pattern_timestamps)
    
    logger.info(f"Pattern memory bank created with {len(patterns)} patterns.")
    return memory_bank, memory_bank_timestamps


def calculate_historical_similarity_batch(
    market_values: np.ndarray,
    memory_bank: np.ndarray,
    pattern_length: int,
    market_timestamps: np.ndarray | None = None,
    memory_bank_timestamps: np.ndarray | None = None,
    causal: bool = True,
) -> np.ndarray:
    """
    Calculates historical similarity for the entire time series in one go using batched matrix operations.
    
    Args:
        market_values (np.ndarray): 1D array of market index returns.
        memory_bank (np.ndarray): 2D array of historical patterns (n_patterns, pattern_length).
        pattern_length (int): Length of each pattern window.
        market_timestamps (np.ndarray | None): Optional timestamps aligned to market_values.
        memory_bank_timestamps (np.ndarray | None): Optional end timestamps aligned to memory_bank rows.
        causal (bool): If True, only allow matches to strictly earlier windows.
        
    Returns:
        np.ndarray: 1D array of similarity scores aligned with market_values. 
                   First (pattern_length-1) elements will be 0.
    """
    n_samples = len(market_values)
    if n_samples < pattern_length or memory_bank.size == 0:
        return np.zeros(n_samples)
        
    try:
        # Build lightweight float32 views to avoid a giant O(n_windows * n_patterns)
        # similarity matrix, which can exceed tens of GB on live data.
        market_values = np.asarray(market_values, dtype=np.float32)
        memory_bank = np.asarray(memory_bank, dtype=np.float32)

        # 1. Create sliding windows for the current market data
        windows = sliding_window_view(market_values, window_shape=pattern_length)

        # 2. Normalize windows statistics once; normalize chunks lazily below.
        w_mean = np.mean(windows, axis=1, keepdims=True, dtype=np.float32)
        w_std = np.std(windows, axis=1, keepdims=True, dtype=np.float32)
        w_valid_mask = (w_std.squeeze(-1) > 1e-8)
        w_std[w_std < 1e-8] = 1.0

        # 3. Normalize valid memory-bank patterns once.
        b_mean = np.mean(memory_bank, axis=1, keepdims=True, dtype=np.float32)
        b_std = np.std(memory_bank, axis=1, keepdims=True, dtype=np.float32)
        b_valid_mask = (b_std.squeeze(-1) > 1e-8)

        valid_bank = memory_bank[b_valid_mask]
        if valid_bank.size == 0:
            return np.zeros(n_samples)

        valid_b_mean = b_mean[b_valid_mask]
        valid_b_std = b_std[b_valid_mask]
        valid_bank_indices = np.flatnonzero(b_valid_mask)
        valid_bank_timestamps = None
        if memory_bank_timestamps is not None:
            valid_bank_timestamps = np.asarray(memory_bank_timestamps)[b_valid_mask]

        bank_norm = (valid_bank - valid_b_mean) / valid_b_std
        bank_norm = np.asarray(bank_norm, dtype=np.float32)
        b_norms = np.linalg.norm(bank_norm, axis=1).astype(np.float32)
        b_norms[b_norms < 1e-8] = 1.0

        if causal and market_timestamps is not None and valid_bank_timestamps is not None:
            query_end_times = np.asarray(market_timestamps)[pattern_length - 1:]
        else:
            query_end_times = None

        chunk_size = int(getattr(config.Data, "HIST_SIM_CHUNK_SIZE", 256) or 256)
        max_similarities = np.zeros(len(windows), dtype=np.float32)
        total_chunks = max(1, (len(windows) + chunk_size - 1) // chunk_size)

        for chunk_idx, start in enumerate(range(0, len(windows), chunk_size), start=1):
            end = min(start + chunk_size, len(windows))
            window_chunk = np.asarray(windows[start:end], dtype=np.float32)
            mean_chunk = w_mean[start:end]
            std_chunk = w_std[start:end]
            windows_norm_chunk = (window_chunk - mean_chunk) / std_chunk

            dot_products = np.dot(windows_norm_chunk, bank_norm.T)
            w_norms_chunk = np.linalg.norm(windows_norm_chunk, axis=1).astype(np.float32)
            w_norms_chunk[w_norms_chunk < 1e-8] = 1.0
            norm_products = w_norms_chunk[:, None] * b_norms[None, :]
            similarity_chunk = dot_products / norm_products

            if causal:
                if query_end_times is not None:
                    causal_mask = valid_bank_timestamps[None, :] < query_end_times[start:end, None]
                else:
                    query_indices = np.arange(start, end, dtype=int)[:, None]
                    causal_mask = valid_bank_indices[None, :] < query_indices
                similarity_chunk = np.where(causal_mask, similarity_chunk, -np.inf)

            chunk_max = np.max(similarity_chunk, axis=1) if similarity_chunk.shape[1] > 0 else np.zeros(end - start, dtype=np.float32)
            chunk_max = (chunk_max + 1.0) / 2.0
            chunk_max[~np.isfinite(chunk_max)] = 0.0
            max_similarities[start:end] = chunk_max

            if total_chunks <= 4 or chunk_idx == 1 or chunk_idx == total_chunks or (chunk_idx % 50 == 0):
                logger.info(
                    f"[HistSim] chunk {chunk_idx}/{total_chunks} processed "
                    f"rows={start}:{end} bank_n={bank_norm.shape[0]}"
                )

        max_similarities[~w_valid_mask] = 0.0

        padding = np.zeros(pattern_length - 1, dtype=np.float32)
        final_similarities = np.concatenate([padding, max_similarities])
        return final_similarities

    except Exception as e:
        logger.error(f"Error in batch similarity calculation: {e}", exc_info=True)
        return np.zeros(n_samples)

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

        for coin in config.Data.MARKET_INDEX_COINS:
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
    logger.info(f"Calculating market index from {config.Data.MARKET_INDEX_COINS}...")
    
    # Pass date range to get weights for the specific period
    market_weights = get_market_weights(start_date=start_date, end_date=end_date)
    index_df = pd.DataFrame()
    
    try:
        # Define the time window for the query
        if end_date and start_date:
             date_filter = f"AND timestamp BETWEEN '{start_date}' AND '{end_date}'"
        else:
             date_filter = "" # No filter, get all data

        for coin in config.Data.MARKET_INDEX_COINS:
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
        
        if index_df.empty or not all(f'{c}_pct_change' in index_df.columns for c in config.Data.MARKET_INDEX_COINS if c in market_weights):
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

def get_crypto_factors(start_date=None, end_date=None):
    """
    Calculates cross-sectional crypto factor returns adapted from Fama-French,
    plus pattern-transfer features for detecting cross-market momentum propagation.

    Factors: SIZE, MOM, VOL, LIQ (long-short portfolios).
    Transfer (market-wide): breadth_ratio, top_n_return_1h, top_n_return_4h, volume_breadth.
    Transfer (per-coin): rank_return_4h (returned separately as a pivoted DataFrame).

    Returns:
        tuple(factor_df, rank_4h_df)
        - factor_df: DataFrame indexed by timestamp with market-wide columns.
        - rank_4h_df: DataFrame indexed by timestamp, columns = market names,
          values = percentile rank of 4h return (0-1). None when data is insufficient.
    """
    logger.info("Calculating crypto factor returns (SIZE, MOM, VOL, LIQ) + pattern transfer features...")

    try:
        date_filter = ""
        if start_date and end_date:
            date_filter = f"AND timestamp BETWEEN '{start_date}' AND '{end_date}'"

        # Load all KRW coin data
        query = f"SELECT market, timestamp, close, volume FROM crypto_data WHERE market LIKE 'KRW-%%' {date_filter} ORDER BY timestamp ASC"
        all_data = load_data(query)

        if all_data.empty:
            logger.warning("No data available for crypto factor calculation.")
            return pd.DataFrame(), None

        all_data['timestamp'] = pd.to_datetime(all_data['timestamp']).dt.tz_localize('UTC')

        # Pivot to wide format: each column = one coin's close price
        price_pivot = all_data.pivot_table(index='timestamp', columns='market', values='close')
        volume_pivot = all_data.pivot_table(index='timestamp', columns='market', values='volume')

        # Need at least 6 coins to form meaningful groups
        valid_coins = price_pivot.columns[price_pivot.notna().sum() > 168]  # At least 7 days of data
        if len(valid_coins) < 6:
            logger.warning(f"Only {len(valid_coins)} coins with sufficient data. Need >= 6 for factor calculation.")
            return pd.DataFrame(), None

        price_pivot = price_pivot[valid_coins]
        volume_pivot = volume_pivot[valid_coins].reindex(price_pivot.index)

        # Calculate returns at different horizons
        returns = price_pivot.pct_change(fill_method=None).fillna(0)
        returns_4h = price_pivot.pct_change(periods=4, fill_method=None).fillna(0)

        # --- Factor Calculations (per timestamp, cross-sectional) ---
        factor_df = pd.DataFrame(index=price_pivot.index)

        # SIZE factor: volume * price as market cap proxy
        # Long small, short large (SMB style)
        market_cap_proxy = price_pivot * volume_pivot.fillna(0)  # vol × price ≈ market cap
        median_cap = market_cap_proxy.median(axis=1)
        small_mask = market_cap_proxy.lt(median_cap, axis=0)
        large_mask = ~small_mask
        factor_df['factor_size'] = (
            returns.where(small_mask).mean(axis=1) - returns.where(large_mask).mean(axis=1)
        ).fillna(0)

        # MOM factor: 7-day momentum (168 hours)
        # Long winners, short losers (WML style)
        rolling_return = price_pivot.pct_change(periods=168, fill_method=None).fillna(0)
        median_mom = rolling_return.median(axis=1)
        winner_mask = rolling_return.gt(median_mom, axis=0)
        loser_mask = ~winner_mask
        factor_df['factor_mom'] = (
            returns.where(winner_mask).mean(axis=1) - returns.where(loser_mask).mean(axis=1)
        ).fillna(0)

        # VOL factor: 24-hour realized volatility
        # Long low-vol, short high-vol (defensive strategy)
        rolling_vol = returns.rolling(window=24).std().fillna(0)
        median_vol = rolling_vol.median(axis=1)
        low_vol_mask = rolling_vol.lt(median_vol, axis=0)
        high_vol_mask = ~low_vol_mask
        factor_df['factor_vol'] = (
            returns.where(low_vol_mask).mean(axis=1) - returns.where(high_vol_mask).mean(axis=1)
        ).fillna(0)

        # LIQ factor: volume-based liquidity
        # Long liquid, short illiquid
        volume_ma_24h = volume_pivot.rolling(window=24).mean().fillna(0)
        median_liq = volume_ma_24h.median(axis=1)
        liquid_mask = volume_ma_24h.gt(median_liq, axis=0)
        illiquid_mask = ~liquid_mask
        factor_df['factor_liq'] = (
            returns.where(liquid_mask).mean(axis=1) - returns.where(illiquid_mask).mean(axis=1)
        ).fillna(0)

        # Clip extreme values for original factors
        for col in ['factor_size', 'factor_mom', 'factor_vol', 'factor_liq']:
            factor_df[col] = factor_df[col].clip(-0.1, 0.1)

        # ===== Pattern Transfer Features (cross-sectional, market-wide) =====

        # 1. breadth_ratio: fraction of coins with positive 4h return
        n_positive = (returns_4h > 0).sum(axis=1)
        n_total = returns_4h.notna().sum(axis=1).clip(lower=1)
        factor_df['breadth_ratio'] = (n_positive / n_total).clip(0, 1)
        factor_df['breadth_ratio'] = factor_df['breadth_ratio'].ffill().fillna(
            factor_df['breadth_ratio'].median() if factor_df['breadth_ratio'].notna().any() else 0.5
        )

        # 2. top_n_return_1h: mean 1h return of top-5 performers
        # At each row, rank coins by 1h return (= returns), take top-5 mean
        top_n = 5
        # Use np for speed: sort each row descending, take top_n mean
        ret_vals = returns.values  # (T, N_coins)
        top_n_1h = np.full(ret_vals.shape[0], np.nan)
        for i in range(ret_vals.shape[0]):
            row = ret_vals[i]
            valid = row[~np.isnan(row)]
            if len(valid) >= top_n:
                top_n_1h[i] = np.partition(valid, -top_n)[-top_n:].mean()
            elif len(valid) > 0:
                top_n_1h[i] = valid.mean()
        factor_df['top_n_return_1h'] = top_n_1h
        factor_df['top_n_return_1h'] = factor_df['top_n_return_1h'].ffill().fillna(
            factor_df['top_n_return_1h'].median() if factor_df['top_n_return_1h'].notna().any() else 0.0
        )
        factor_df['top_n_return_1h'] = factor_df['top_n_return_1h'].clip(-0.3, 0.3)

        # 3. top_n_return_4h: mean 4h return of top-5 performers
        ret4_vals = returns_4h.values
        top_n_4h = np.full(ret4_vals.shape[0], np.nan)
        for i in range(ret4_vals.shape[0]):
            row = ret4_vals[i]
            valid = row[~np.isnan(row)]
            if len(valid) >= top_n:
                top_n_4h[i] = np.partition(valid, -top_n)[-top_n:].mean()
            elif len(valid) > 0:
                top_n_4h[i] = valid.mean()
        factor_df['top_n_return_4h'] = top_n_4h
        factor_df['top_n_return_4h'] = factor_df['top_n_return_4h'].ffill().fillna(
            factor_df['top_n_return_4h'].median() if factor_df['top_n_return_4h'].notna().any() else 0.0
        )
        factor_df['top_n_return_4h'] = factor_df['top_n_return_4h'].clip(-0.5, 0.5)

        # 4. volume_breadth: fraction of coins where current volume > 2x their 24h MA
        vol_activated = volume_pivot > (2.0 * volume_ma_24h)
        n_vol_active = vol_activated.sum(axis=1)
        n_vol_total = vol_activated.notna().sum(axis=1).clip(lower=1)
        factor_df['volume_breadth'] = (n_vol_active / n_vol_total).clip(0, 1)
        factor_df['volume_breadth'] = factor_df['volume_breadth'].ffill().fillna(
            factor_df['volume_breadth'].median() if factor_df['volume_breadth'].notna().any() else 0.0
        )

        # 5. rank_return_4h (per-coin): percentile rank across all coins at each timestamp
        # Returns a pivoted DataFrame (index=timestamp, columns=market, values=0..1)
        rank_4h_df = returns_4h.rank(axis=1, pct=True)
        rank_4h_df = rank_4h_df.ffill().fillna(0.5)  # 0.5 = median rank when unknown

        logger.info(f"Crypto factors + transfer features calculated. Shape: {factor_df.shape}, Coins: {len(valid_coins)}")
        return factor_df, rank_4h_df

    except Exception as e:
        logger.error(f"Failed to calculate crypto factors: {e}")
        return pd.DataFrame(), None

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

    # Normalized Bollinger Band position: (close - lower) / (upper - lower), clipped to [0, 1]
    df['bb_position'] = ((df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)).clip(0, 1)

    df['volume_ma'] = volume_ma

    # --- Volatility Features (NEW) ---
    # ATR proxy using close price only (true ATR needs high/low which may not always be reliable)
    df['volatility_24h'] = df['close'].pct_change().rolling(window=24).std()
    df['volatility_7d'] = df['close'].pct_change().rolling(window=168).std()  # 7 days * 24 hours
    
    # Volume volatility (sudden volume spikes indicate potential moves)
    df['volume_volatility'] = df['volume'].pct_change().rolling(window=24).std()

    # --- New Features (Phase 2 Enhancement) ---
    # Price position within 20-period range (0=at low, 1=at high)
    rolling_high = df['close'].rolling(window=480).max()  # 20 days * 24h
    rolling_low = df['close'].rolling(window=480).min()
    df['price_position'] = ((df['close'] - rolling_low) / (rolling_high - rolling_low + 1e-10)).clip(0, 1)

    # Volume ratio: current volume vs 20-period MA (spike detection)
    df['volume_ratio'] = (df['volume'] / (volume_ma + 1e-10)).clip(0, 10)

    # Return skewness over 24h (asymmetry → potential breakout signal)
    df['return_skew_24h'] = df['close'].pct_change().rolling(window=24).skew()

    # Forward-fill indicators that should persist (RSI, MACD, ADX, etc.)
    indicator_cols = ['rsi', 'macd', 'macdsignal', 'macdhist', 'adx', 'obv',
                      'bb_upper', 'bb_middle', 'bb_lower', 'bb_position', 'volume_ma',
                      'volatility_24h', 'volatility_7d', 'volume_volatility',
                      'price_position', 'volume_ratio', 'return_skew_24h']
    for col in indicator_cols:
        if col in df.columns:
            df[col] = df[col].ffill().fillna(df[col].median() if df[col].notna().any() else 0)
    return df

def create_sequences(data, sequence_length, future_window):
    xs, ys = [], []
    for i in range(len(data) - sequence_length - future_window):
        x = data[i:(i + sequence_length)]
        y = data[(i + sequence_length):(i + sequence_length + future_window), 0]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def get_intermediate_data(market: str, market_index_df: pd.DataFrame, historical_df: pd.DataFrame = None, crypto_factors_df: pd.DataFrame = None, rank_4h_df: pd.DataFrame = None):
    """
    Loads and preprocesses data for a market up to the point of feature calculation.
    If historical_df is provided, it uses that dataframe; otherwise, it queries the database.
    Ensures all timezone information is consistent (UTC) before processing.

    Args:
        crypto_factors_df: Optional pre-calculated crypto factor returns (from get_crypto_factors).
        rank_4h_df: Optional per-coin 4h return percentile ranks (from get_crypto_factors).
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
    df['market_index_return'] = df['market_index_return'].ffill().bfill()

    # Detect candle gaps: if timestamp diff > 1.5h, mark as gap
    if isinstance(df.index, pd.DatetimeIndex):
        time_diff = df.index.to_series().diff()
        gap_mask = time_diff > pd.Timedelta(hours=1.5)
        n_gaps = gap_mask.sum()
        if n_gaps > 0:
            logger.info(f"[Preprocessor] Detected {n_gaps} candle gaps, interpolating")
            # Reindex to full hourly grid and interpolate
            full_idx = pd.date_range(df.index.min(), df.index.max(), freq='h', tz=df.index.tz)
            df = df.reindex(full_idx)
            df['close'] = df['close'].interpolate(method='linear')
            df['volume'] = df['volume'].interpolate(method='linear').fillna(0)
            # Forward-fill other columns
            df = df.ffill().bfill()

    df['coin_return'] = df['close'].pct_change().fillna(0)
    # --- Rolling closed-form beta (vectorized, no OLS) ---
    W = getattr(config.Data, 'BETA_ROLLING_WINDOW', 168)
    eps = 1e-8
    coin_ret = df['coin_return']
    mkt_ret  = df['market_index_return']
    # clip extreme returns before beta estimation
    roll_std = coin_ret.rolling(W).std().ffill().fillna(coin_ret.expanding().std().fillna(1e-4))
    coin_ret_c = coin_ret.clip(-5 * roll_std, 5 * roll_std)
    mean_x  = mkt_ret.rolling(W).mean()
    mean_y  = coin_ret_c.rolling(W).mean()
    cov_xy  = (coin_ret_c * mkt_ret).rolling(W).mean() - mean_y * mean_x
    var_x   = mkt_ret.rolling(W).var().clip(lower=eps)
    beta_raw = (cov_xy / var_x).clip(-3, 3)
    # hold last valid beta when var_x is too small
    df['beta'] = beta_raw.ffill().bfill()
    alpha_intercept = mean_y - df['beta'] * mean_x
    df['alpha'] = (coin_ret_c - alpha_intercept - df['beta'] * mkt_ret).fillna(0)  # idiosyncratic resid

    # --- BTC 4-state regime features (computed from market_index_df BTC price) ---
    # market_index_df has market_index_return; reconstruct cumulative price proxy
    if 'market_index_return' in df.columns:
        mkt_r = df['market_index_return']
        rv = mkt_r.rolling(168).std() * (8760 ** 0.5)  # annualised RV
        rv_lookback = getattr(config.Data, 'REGIME_RV_LOOKBACK', 4320)
        rv_q40 = rv.rolling(rv_lookback, min_periods=168).quantile(0.4)
        rv_q60 = rv.rolling(rv_lookback, min_periods=168).quantile(0.6)
        rv_low  = (rv < rv_q40).astype(float)
        rv_high = (rv > rv_q60).astype(float)
        # MA distance as trend signal
        price_proxy = (1 + mkt_r).cumprod()
        ma200 = price_proxy.rolling(200, min_periods=1).mean()
        delta = getattr(config.Data, 'REGIME_HYSTERESIS', 0.02)
        trend_up   = (price_proxy > ma200 * (1 + delta)).astype(float)
        trend_down = (price_proxy < ma200 * (1 - delta)).astype(float)
        # 4-state: 0=bull_quiet,1=bull_vol,2=bear_quiet,3=bear_vol
        regime = pd.Series(np.nan, index=df.index)
        regime[trend_up.astype(bool)  & rv_low.astype(bool)]  = 0
        regime[trend_up.astype(bool)  & rv_high.astype(bool)] = 1
        regime[trend_down.astype(bool) & rv_low.astype(bool)] = 2
        regime[trend_down.astype(bool) & rv_high.astype(bool)]= 3
        regime = regime.ffill().fillna(0)
        df['btc_regime']     = regime
        df['btc_regime_rv']  = (rv / (rv_q60 + 1e-8)).clip(0, 5).fillna(0)
        df['btc_ma_distance'] = ((price_proxy / (ma200 + 1e-8)) - 1).clip(-0.5, 0.5).fillna(0)
    else:
        df['btc_regime'] = 0.0
        df['btc_regime_rv'] = 0.0
        df['btc_ma_distance'] = 0.0

    # --- Historical Similarity (Retrieval-Augmented PE) ---
    # Get or create the pattern memory bank from market index
    pattern_length = config.Data.SEQUENCE_LENGTH
    memory_bank, memory_bank_timestamps = get_or_create_pattern_memory_bank(market_index_df, pattern_length)
    
    # Calculate historical similarity using batch processing
    # This replaces the slow loop with optimized matrix operations
    market_index_values = df['market_index_return'].values
    historical_similarities = calculate_historical_similarity_batch(
        market_index_values,
        memory_bank,
        pattern_length,
        market_timestamps=df.index.values,
        memory_bank_timestamps=memory_bank_timestamps,
    )
    
    df['historical_similarity'] = historical_similarities
    logger.info(f"Historical similarity calculated. Mean: {df['historical_similarity'].mean():.4f}")

    # Cross-correlation with market index (rolling 24h)
    df['cross_corr_btc'] = df['coin_return'].rolling(window=24).corr(df['market_index_return']).ffill().fillna(0).clip(-1, 1)

    # --- Join Crypto Factor Returns (FF-adapted) + Pattern Transfer Features ---
    cross_sectional_cols = ['factor_size', 'factor_mom', 'factor_vol', 'factor_liq',
                            'breadth_ratio', 'top_n_return_1h', 'top_n_return_4h', 'volume_breadth']
    if crypto_factors_df is not None and not crypto_factors_df.empty:
        if crypto_factors_df.index.tz is None:
            crypto_factors_df.index = crypto_factors_df.index.tz_localize('UTC')
        df = df.join(crypto_factors_df, how='left')
        for col in cross_sectional_cols:
            if col not in df.columns:
                df[col] = 0.0
            median_val = df[col].median() if df[col].notna().any() else 0.0
            df[col] = df[col].ffill().fillna(median_val)
    else:
        for col in cross_sectional_cols:
            df[col] = 0.0

    # --- Per-coin: rank_return_4h (percentile rank of 4h return across all coins) ---
    if rank_4h_df is not None and market in rank_4h_df.columns:
        rank_series = rank_4h_df[market]
        if rank_series.index.tz is None:
            rank_series.index = rank_series.index.tz_localize('UTC')
        df['rank_return_4h'] = rank_series.reindex(df.index)
        df['rank_return_4h'] = df['rank_return_4h'].ffill().fillna(0.5)  # 0.5 = median rank fallback
    else:
        df['rank_return_4h'] = 0.5

    # --- Per-coin: net_volume_flow (signed volume proxy) ---
    # rolling_sum(volume * sign(return), window=6h) / rolling_mean(volume, 24h)
    ret_sign = np.sign(df['coin_return'])
    signed_vol = df['volume'] * ret_sign
    net_vol_raw = signed_vol.rolling(window=6, min_periods=1).sum()
    vol_ma_norm = df['volume'].rolling(window=24, min_periods=1).mean().clip(lower=1e-8)
    df['net_volume_flow'] = (net_vol_raw / vol_ma_norm).clip(-10, 10)
    df['net_volume_flow'] = df['net_volume_flow'].ffill().fillna(
        df['net_volume_flow'].median() if df['net_volume_flow'].notna().any() else 0.0
    )

    # Ablation: keep model input dims fixed, but zero-out selected feature groups.
    if str(os.getenv("AETHER_ABLATE_FACTORS", "0")).strip().lower() in ("1", "true", "yes", "y"):
        for col in ["factor_size", "factor_mom", "factor_vol", "factor_liq"]:
            df[col] = 0.0
    if str(os.getenv("AETHER_ABLATE_CONTEXT", "0")).strip().lower() in ("1", "true", "yes", "y"):
        # Macro/context features
        df["market_index_return"] = 0.0
        df["historical_similarity"] = 0.0

    # --- FF5 × regime interaction features ---
    # Compute these after factor join / ablation so they don't silently stay at zero.
    bull_flag = (df['btc_regime'].isin([0, 1])).astype(float)
    bear_flag = (df['btc_regime'].isin([2, 3])).astype(float)
    df['factor_mom_x_bull'] = df.get('factor_mom', pd.Series(0.0, index=df.index)).fillna(0.0) * bull_flag
    df['factor_liq_x_bear'] = df.get('factor_liq', pd.Series(0.0, index=df.index)).fillna(0.0) * bear_flag

    # ── Binance Cross-Market Features ─────────────────────────────────
    try:
        binance_df = load_data(
            f"SELECT timestamp, close as binance_close, volume as binance_volume "
            f"FROM binance_data WHERE market = '{market}' ORDER BY timestamp ASC"
        )
        if binance_df is not None and len(binance_df) > 0:
            binance_df['timestamp'] = pd.to_datetime(binance_df['timestamp'], utc=True)
            binance_df = binance_df.set_index('timestamp')
            # Align timezone: strip tz from both sides for safe join
            if binance_df.index.tz is not None:
                binance_df.index = binance_df.index.tz_localize(None)
            df_had_tz = df.index.tz is not None
            if df_had_tz:
                df.index = df.index.tz_localize(None)

            # Merge on timestamp
            df = df.join(binance_df, how='left')

            # Restore tz if needed
            if df_had_tz:
                df.index = df.index.tz_localize('UTC')

            # 1. Kimchi Premium: upbit_price / binance_price ratio (proxy, no FX needed)
            #    Ratio changes = premium changes. Absolute level doesn't matter for returns.
            df['kimchi_premium'] = (df['close'] / (df['binance_close'] + 1e-8))
            df['kimchi_premium'] = df['kimchi_premium'] / df['kimchi_premium'].rolling(24).mean()  # normalize to 24h MA
            df['kimchi_premium'] = (df['kimchi_premium'] - 1.0).clip(-0.1, 0.1)  # centered, clipped

            # 2. Binance Lead Return (1h): what Binance did in the last hour
            binance_return = df['binance_close'].pct_change()
            df['binance_lead_return_1h'] = binance_return.shift(0)  # current hour binance return
            # The "lead" signal comes from the fact that global markets react faster

            # 3. Global Volume Ratio: upbit share of total volume
            total_vol = df['volume'] + df['binance_volume'].fillna(0) + 1e-8
            df['global_volume_ratio'] = df['volume'] / total_vol
            df['global_volume_ratio'] = df['global_volume_ratio'].clip(0, 1)

            # Clean up temp columns
            df.drop(columns=['binance_close', 'binance_volume'], inplace=True, errors='ignore')
        else:
            # No Binance data for this market — fill with neutral values
            df['kimchi_premium'] = 0.0
            df['binance_lead_return_1h'] = 0.0
            df['global_volume_ratio'] = 0.5
    except Exception as e:
        logger.warning(f"Binance cross-market features failed for {market}: {e}")
        df['kimchi_premium'] = 0.0
        df['binance_lead_return_1h'] = 0.0
        df['global_volume_ratio'] = 0.5

    # Fill NaN in new features
    for col in ['kimchi_premium', 'binance_lead_return_1h', 'global_volume_ratio']:
        df[col] = df[col].ffill().fillna(0.0 if col != 'global_volume_ratio' else 0.5)

    df['future_pct_change'] = (df['close'].shift(-1) - df['close']) / df['close']
    # Forward-fill then use column median for remaining NaN (avoids false zero signals)
    df = df.ffill()
    for col in df.select_dtypes(include='number').columns:
        if df[col].isna().any():
            fill_val = df[col].median() if df[col].notna().any() else 0.0
            df[col] = df[col].fillna(fill_val)
    
    scaler = MinMaxScaler()

    return df, scaler

WARMUP_ROWS = 480  # max(price_position window=480, beta window=168)

def create_final_sequences_and_scale(df: pd.DataFrame, scaler: MinMaxScaler):
    """
    Takes a dataframe with all features, scales them, and creates sequences.
    """
    # Drop warmup rows where rolling indicators (price_position=480, beta=168) are unreliable
    if len(df) > WARMUP_ROWS:
        df = df.iloc[WARMUP_ROWS:].copy()
        logger.info(f"Dropped first {WARMUP_ROWS} warmup rows. Remaining rows: {len(df)}")

    features_to_scale = [
        'close', 'volume', 'rsi', 'macd', 'macdsignal', 'macdhist', 'adx', 'obv',
        'market_index_return', 'historical_similarity',
        'bb_upper', 'bb_middle', 'bb_lower', 'volume_ma',
        'volatility_24h', 'volatility_7d', 'volume_volatility',
        'alpha', 'beta',
        'price_position', 'volume_ratio', 'return_skew_24h', 'cross_corr_btc',
        'factor_size', 'factor_mom', 'factor_vol', 'factor_liq',
        'btc_regime', 'btc_regime_rv', 'btc_ma_distance',
        'factor_mom_x_bull', 'factor_liq_x_bear',
        'breadth_ratio', 'top_n_return_1h', 'top_n_return_4h',
        'rank_return_4h', 'net_volume_flow', 'volume_breadth',
    ]
    
    # Validate no impossible values remain after warmup drop
    if 'rsi' in df.columns and df['rsi'].notna().any():
        invalid_rsi = ((df['rsi'] < 0) | (df['rsi'] > 100)).sum()
        if invalid_rsi > 0:
            logger.warning(f"RSI has {invalid_rsi} values outside [0, 100] after warmup drop")
    if 'beta' in df.columns:
        nan_beta = df['beta'].isna().sum()
        if nan_beta > 0:
            logger.warning(f"Beta has {nan_beta} NaN values after warmup drop")

    features_df, _ = _prepare_feature_frame(df, scaler, fit_scaler=True)
    data_for_sequences = np.c_[df[['future_pct_change']], features_df.to_numpy(dtype=float)]

    X, y = create_sequences(data_for_sequences, config.Data.SEQUENCE_LENGTH, config.Data.FUTURE_WINDOW_SIZE)
    
    if X.shape[0] == 0:
        logger.warning(f"Not enough data to create any sequences after processing. Skipping.")
        return None, None, None

    X = X[:, :, 1:]

    logger.info(f"Data scaling and sequencing complete. Shape of X: {X.shape}, Shape of y: {y.shape}")
    
    return X, y, scaler


def _prepare_feature_frame(df: pd.DataFrame, scaler: MinMaxScaler, fit_scaler: bool):
    feature_columns = [f for f in config.Data.FEATURE_COLUMNS if f in df.columns]
    non_scaled_cols = [
        c for c in getattr(config.Data, 'NON_SCALED_FEATURE_COLUMNS', [])
        if c in feature_columns
    ]
    scale_cols = [f for f in feature_columns if f not in non_scaled_cols]

    # Normalize dtypes up front so pandas/scikit-learn round-trips do not fail
    # on mixed float32/float64 columns (seen after chunked HistSim outputs).
    features_df = df[feature_columns].copy().astype(np.float64)
    if scale_cols:
        if fit_scaler:
            features_df.loc[:, scale_cols] = scaler.fit_transform(features_df[scale_cols])
        else:
            features_df.loc[:, scale_cols] = scaler.transform(features_df[scale_cols])
    return features_df, feature_columns


def create_train_val_sequences_and_scale(
    df: pd.DataFrame,
    scaler: MinMaxScaler,
    train_ratio: float,
):
    """
    Chronological train/validation sequence builder.
    Fits the scaler on the train slice only, then applies the same scaler to
    validation data while preserving enough lookback context for the first
    validation window.
    """
    # Drop warmup rows where rolling indicators (price_position=480, beta=168) are unreliable
    if len(df) > WARMUP_ROWS:
        df = df.iloc[WARMUP_ROWS:].copy()
        logger.info(f"[train_val] Dropped first {WARMUP_ROWS} warmup rows. Remaining rows: {len(df)}")

    seq_len = config.Data.SEQUENCE_LENGTH
    horizon = config.Data.FUTURE_WINDOW_SIZE
    min_rows = seq_len + horizon + 8
    if len(df) < min_rows:
        logger.warning("Not enough rows to build chronological train/val sequences.")
        return None, None, None, None, scaler

    split_idx = int(len(df) * train_ratio)
    split_idx = max(min_rows, split_idx)
    split_idx = min(split_idx, len(df) - (horizon + 2))

    train_df = df.iloc[:split_idx].copy()
    val_start = max(0, split_idx - seq_len)
    val_df = df.iloc[val_start:].copy()

    train_features_df, _ = _prepare_feature_frame(train_df, scaler, fit_scaler=True)
    val_features_df, _ = _prepare_feature_frame(val_df, scaler, fit_scaler=False)

    train_data = np.c_[train_df[['future_pct_change']], train_features_df.to_numpy(dtype=float)]
    val_data = np.c_[val_df[['future_pct_change']], val_features_df.to_numpy(dtype=float)]

    X_train, y_train = create_sequences(train_data, seq_len, horizon)
    X_val, y_val = create_sequences(val_data, seq_len, horizon)

    if X_train.shape[0] == 0 or X_val.shape[0] == 0:
        logger.warning("Chronological split produced empty train/val sequences.")
        return None, None, None, None, scaler

    X_train = X_train[:, :, 1:]
    X_val = X_val[:, :, 1:]

    # Validation sequences include `seq_len` warmup rows from train for context.
    # Drop any sequence whose target starts before the true split point.
    offset = max(0, split_idx - val_start - seq_len)
    if offset > 0:
        X_val = X_val[offset:]
        y_val = y_val[offset:]

    if X_val.shape[0] == 0:
        logger.warning("Validation offset removed all validation sequences.")
        return None, None, None, None, scaler

    return X_train, y_train, X_val, y_val, scaler

def get_processed_data_for_training(market: str, market_index_df: pd.DataFrame):
    """Wrapper function for training that calls the refactored processing steps."""
    # Calculate crypto factor returns + pattern transfer features for training context
    crypto_factors_df, rank_4h_df = get_crypto_factors()
    # Pass historical_df=None to ensure it queries the DB for fresh training data
    df, scaler = get_intermediate_data(market, market_index_df, historical_df=None, crypto_factors_df=crypto_factors_df, rank_4h_df=rank_4h_df)
    if df is None:
        return None, None, None
    
    # For training, we don't apply shrinkage, so we pass the df directly
    return create_final_sequences_and_scale(df, scaler)

def get_recent_pattern(market: str, current_time: datetime, hours: int = config.Pattern.LENGTH) -> np.ndarray:
    """
    Loads recent historical data for a market and returns its price change pattern.
    Returns a 1D numpy array of percentage changes for the last 'hours' period.
    """
    # Need hours + 1 data points to calculate 'hours' percentage changes
    # DB timestamps are stored as ISO strings "YYYY-MM-DDTHH:MM:SS"
    if isinstance(current_time, pd.Timestamp):
        current_time = current_time.to_pydatetime()
    if isinstance(current_time, datetime):
        current_time_s = current_time.strftime('%Y-%m-%dT%H:%M:%S')
    else:
        current_time_s = str(current_time)
    query = (
        "SELECT timestamp, close FROM crypto_data "
        f"WHERE market = '{market}' AND timestamp <= '{current_time_s}' "
        f"ORDER BY timestamp DESC LIMIT {hours + 1}"
    )
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
    recalculate=False
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
    
    window_size = config.Data.FUTURE_WINDOW_SIZE
    min_return = config.Recommender.SUCCESS_PATTERN_MIN_RETURN

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
    df['volume_spike_score'] = df['volume_pct_change'] / (df['volume_pct_change'].rolling(window=config.Pattern.LENGTH).mean() + 1e-9)

    # Bollinger Band Squeeze
    if 'bb_middle' in df.columns and df['bb_middle'].notna().any():
        bb_bandwidth = (df['bb_upper'] - df['bb_lower']) / (df['bb_middle'] + 1e-9)
        # A squeeze is when the current bandwidth is at a 24-hour low
        df['squeeze_on'] = (bb_bandwidth.rolling(window=config.Pattern.LENGTH).min() == bb_bandwidth).astype(int)
    else:
        df['squeeze_on'] = 0

    # Price momentum (rate of change over different periods)
    df['roc_3'] = ta.roc(df['close'], length=3)
    df['roc_6'] = ta.roc(df['close'], length=6)

    return df

def create_pump_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Creates multi-class ground truth labels for pump events."""
    time_horizon = config.Data.FUTURE_WINDOW_SIZE
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
        df['market_index_return'] = df['market_index_return'].ffill().bfill()

        # 1. Calculate standard and new pump-specific features
        df = calculate_technical_indicators(df)
        df = create_pump_features(df)

        # 2. Create the pump labels
        df = create_pump_labels(df)

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

def create_sequences_from_index(index_df: pd.DataFrame, length: int):
    """Creates overlapping sequences from the market index return series."""
    if index_df.empty or 'market_index_return' not in index_df.columns:
        logger.error("Market index dataframe is empty or missing 'market_index_return' column.")
        return np.array([])

    returns = index_df['market_index_return'].values
    sequences = []
    for i in range(len(returns) - length + 1):
        sequences.append(returns[i:i+length])
    
    return np.array(sequences)
