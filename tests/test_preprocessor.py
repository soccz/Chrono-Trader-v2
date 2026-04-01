import pandas as pd
import numpy as np
import sys
import os
from sklearn.preprocessing import MinMaxScaler

# Add project root to allow imports from other directories
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.preprocessor import (
    create_sequences_from_index,
    create_final_sequences_and_scale,
    calculate_historical_similarity_batch,
)
from utils.config import config

def test_create_sequences_from_index_basic():
    """
    Tests the basic functionality of sequence creation.
    """
    # 1. Prepare sample data
    data = {'market_index_return': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}
    df = pd.DataFrame(data)
    length = 4

    # 2. Call the function
    sequences = create_sequences_from_index(df, length)

    # 3. Assertions
    assert isinstance(sequences, np.ndarray), "Output should be a numpy array"
    
    expected_num_sequences = len(data['market_index_return']) - length + 1
    assert sequences.shape == (expected_num_sequences, length), f"Shape should be ({expected_num_sequences}, {length})"
    
    expected_first_sequence = np.array([0.1, 0.2, 0.3, 0.4])
    assert np.array_equal(sequences[0], expected_first_sequence), "First sequence content is incorrect"

    expected_last_sequence = np.array([0.3, 0.4, 0.5, 0.6])
    assert np.array_equal(sequences[-1], expected_last_sequence), "Last sequence content is incorrect"

def test_create_sequences_from_index_empty_input():
    """
    Tests behavior with an empty DataFrame.
    """
    df = pd.DataFrame({'market_index_return': []})
    sequences = create_sequences_from_index(df, 5)
    assert sequences.shape == (0,), "Should return an empty array for empty input"

def test_create_sequences_from_index_length_too_long():
    """
    Tests behavior when the requested sequence length is longer than the data.
    """
    data = {'market_index_return': [0.1, 0.2, 0.3]}
    df = pd.DataFrame(data)
    sequences = create_sequences_from_index(df, 4)
    assert sequences.shape[0] == 0, "Should return no sequences if length is greater than data size"


def test_create_final_sequences_and_scale_preserves_regime_states():
    rows = 220
    df = pd.DataFrame({
        'close': np.linspace(100, 200, rows),
        'volume': np.linspace(1000, 2000, rows),
        'rsi': np.linspace(30, 70, rows),
        'macd': np.linspace(-1, 1, rows),
        'macdsignal': np.linspace(-0.5, 0.5, rows),
        'macdhist': np.linspace(-0.2, 0.2, rows),
        'adx': np.linspace(10, 40, rows),
        'obv': np.linspace(10000, 20000, rows),
        'market_index_return': np.linspace(-0.01, 0.01, rows),
        'historical_similarity': np.linspace(0, 1, rows),
        'bb_upper': np.linspace(110, 210, rows),
        'bb_middle': np.linspace(100, 200, rows),
        'bb_lower': np.linspace(90, 190, rows),
        'bb_position': np.linspace(0.0, 1.0, rows),
        'volume_ma': np.linspace(900, 1900, rows),
        'volatility_24h': np.linspace(0.01, 0.03, rows),
        'volatility_7d': np.linspace(0.02, 0.04, rows),
        'volume_volatility': np.linspace(0.1, 0.3, rows),
        'alpha': np.linspace(-0.02, 0.02, rows),
        'beta': np.linspace(0.5, 1.5, rows),
        'price_position': np.linspace(0, 1, rows),
        'volume_ratio': np.linspace(0.5, 2.0, rows),
        'return_skew_24h': np.linspace(-1, 1, rows),
        'cross_corr_btc': np.linspace(-0.5, 0.5, rows),
        'factor_size': np.linspace(-0.05, 0.05, rows),
        'factor_mom': np.linspace(-0.04, 0.04, rows),
        'factor_vol': np.linspace(-0.03, 0.03, rows),
        'factor_liq': np.linspace(-0.02, 0.02, rows),
        'btc_regime': np.tile([0, 1, 2, 3], rows // 4 + 1)[:rows],
        'btc_regime_rv': np.linspace(0.2, 1.8, rows),
        'btc_ma_distance': np.linspace(-0.1, 0.1, rows),
        'factor_mom_x_bull': np.linspace(-0.04, 0.04, rows),
        'factor_liq_x_bear': np.linspace(-0.02, 0.02, rows),
        'breadth_ratio': np.linspace(0.3, 0.8, rows),
        'top_n_return_1h': np.linspace(-0.05, 0.1, rows),
        'top_n_return_4h': np.linspace(-0.1, 0.2, rows),
        'rank_return_4h': np.linspace(0.0, 1.0, rows),
        'net_volume_flow': np.linspace(-3, 3, rows),
        'volume_breadth': np.linspace(0.0, 0.5, rows),
        'future_pct_change': np.linspace(-0.01, 0.01, rows),
    })

    X, _, _ = create_final_sequences_and_scale(df, MinMaxScaler())
    regime_idx = config.Data.FEATURE_COLUMNS.index('btc_regime')
    observed_values = set(np.unique(X[..., regime_idx]).tolist())
    assert observed_values.issubset({0.0, 1.0, 2.0, 3.0})


def test_calculate_historical_similarity_batch_is_causal():
    timestamps = pd.date_range("2024-01-01", periods=9, freq="h", tz="UTC")
    market_values = np.array([1.0, 2.0, 3.0, 9.0, 9.0, 9.0, 1.0, 2.0, 3.0])
    pattern_length = 3
    memory_bank = np.array([
        market_values[i : i + pattern_length]
        for i in range(len(market_values) - pattern_length + 1)
    ])
    memory_bank_timestamps = timestamps[pattern_length - 1 :].to_numpy()

    similarities = calculate_historical_similarity_batch(
        market_values=market_values,
        memory_bank=memory_bank,
        pattern_length=pattern_length,
        market_timestamps=timestamps.to_numpy(),
        memory_bank_timestamps=memory_bank_timestamps,
    )

    assert similarities.shape == market_values.shape
    assert np.isclose(similarities[pattern_length - 1], 0.0)
    assert similarities[-1] > 0.99


def test_calculate_historical_similarity_batch_chunked_matches_small_case():
    timestamps = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
    market_values = np.array([1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 8.0, 5.0, 3.0, 2.0])
    pattern_length = 4
    memory_bank = np.array([
        market_values[i : i + pattern_length]
        for i in range(len(market_values) - pattern_length + 1)
    ])
    memory_bank_timestamps = timestamps[pattern_length - 1 :].to_numpy()

    prev_chunk_size = getattr(config.Data, "HIST_SIM_CHUNK_SIZE", None)
    config.Data.HIST_SIM_CHUNK_SIZE = 2
    try:
        similarities = calculate_historical_similarity_batch(
            market_values=market_values,
            memory_bank=memory_bank,
            pattern_length=pattern_length,
            market_timestamps=timestamps.to_numpy(),
            memory_bank_timestamps=memory_bank_timestamps,
        )
    finally:
        if prev_chunk_size is None:
            try:
                delattr(config.Data, "HIST_SIM_CHUNK_SIZE")
            except AttributeError:
                pass
        else:
            config.Data.HIST_SIM_CHUNK_SIZE = prev_chunk_size

    assert similarities.shape == market_values.shape
    assert np.all(np.isfinite(similarities))
    assert np.isclose(similarities[pattern_length - 1], 0.0)
    assert similarities[-1] >= 0.0
