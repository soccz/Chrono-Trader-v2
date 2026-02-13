import pandas as pd
import numpy as np
import sys
import os

# Add project root to allow imports from other directories
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.preprocessor import create_sequences_from_index

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
