
import unittest
import pandas as pd
import numpy as np
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference import recommender
from utils.config import config

# Mock logger to avoid clutter
logging.basicConfig(level=logging.DEBUG)

class TestRecommender(unittest.TestCase):
    def test_lead_lag_logic(self):
        print("\nTesting Lead-Lag Analysis Logic...")
        
        # 1. Create Mock Predictions
        predictions = [{
            'market': 'KRW-XRP',
            'predicted_pattern': np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06]),
            'uncertainty': 0.1,
            'current_price': 1000,
            'strategy': 'trending'
        }]

        # 2. Create Mock Historical Data (BTC and XRP)
        # Create 100 hours of data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='h')
        
        # BTC moves up
        btc_close = np.linspace(100, 200, 100) 
        # XRP follows BTC with 2 hour lag
        xrp_close = np.roll(btc_close, 2)
        
        df_btc = pd.DataFrame({'timestamp': dates, 'close': btc_close, 'market': 'KRW-BTC'})
        df_xrp = pd.DataFrame({'timestamp': dates, 'close': xrp_close, 'market': 'KRW-XRP'})
        
        historical_data = pd.concat([df_btc, df_xrp])
        historical_data.set_index('timestamp', inplace=True)
        
        # 3. Mock external dependencies to make the test deterministic.
        original_get_market_index = recommender.get_market_index
        original_get_tradeable_markets = recommender.get_tradeable_markets
        original_get_trading_values = recommender.get_trading_values_for_markets
        original_get_historical_success_patterns = recommender.get_historical_success_patterns

        recommender.get_market_index = lambda: pd.DataFrame()
        recommender.get_tradeable_markets = lambda: {'KRW-XRP', 'KRW-BTC'}
        recommender.get_trading_values_for_markets = lambda markets, end_time, hours: {'KRW-XRP': 2_000_000_000}
        recommender.get_historical_success_patterns = lambda: np.array([])
        
        try:
             recommendations = recommender.run(predictions, historical_data=historical_data, mode='live')
        finally:
            recommender.get_market_index = original_get_market_index
            recommender.get_tradeable_markets = original_get_tradeable_markets
            recommender.get_trading_values_for_markets = original_get_trading_values
            recommender.get_historical_success_patterns = original_get_historical_success_patterns

        self.assertTrue(len(recommendations) > 0)

    def test_short_signal_not_counter_when_regime_unknown(self):
        """
        If market regime is unknown (empty index), short signals must not be auto-labeled Counter.
        This prevents unintended stricter uncertainty thresholding.
        """
        predictions = [{
            'market': 'KRW-XRP',
            'predicted_pattern': np.array([-0.01, -0.02, -0.01, -0.015, 0.0, -0.005]),
            'uncertainty': 450.0,  # Below base(500), above old counter-adjusted(350)
            'current_price': 1000,
            'strategy': 'trending',
            'consensus_score': 0.8
        }]

        dates = pd.date_range(start='2024-01-01', periods=100, freq='h')
        btc_close = np.linspace(100, 200, 100)
        xrp_close = np.roll(btc_close, 2)
        df_btc = pd.DataFrame({'timestamp': dates, 'close': btc_close, 'market': 'KRW-BTC'})
        df_xrp = pd.DataFrame({'timestamp': dates, 'close': xrp_close, 'market': 'KRW-XRP'})
        historical_data = pd.concat([df_btc, df_xrp])
        historical_data.set_index('timestamp', inplace=True)

        original_get_market_index = recommender.get_market_index
        original_get_tradeable_markets = recommender.get_tradeable_markets
        original_get_trading_values = recommender.get_trading_values_for_markets
        original_get_historical_success_patterns = recommender.get_historical_success_patterns

        recommender.get_market_index = lambda: pd.DataFrame()  # regime unknown
        recommender.get_tradeable_markets = lambda: {'KRW-XRP', 'KRW-BTC'}
        recommender.get_trading_values_for_markets = lambda markets, end_time, hours: {'KRW-XRP': 2_000_000_000}
        recommender.get_historical_success_patterns = lambda: np.array([])

        try:
            recommendations = recommender.run(predictions, historical_data=historical_data, mode='live')
        finally:
            recommender.get_market_index = original_get_market_index
            recommender.get_tradeable_markets = original_get_tradeable_markets
            recommender.get_trading_values_for_markets = original_get_trading_values
            recommender.get_historical_success_patterns = original_get_historical_success_patterns

        self.assertTrue(len(recommendations) > 0)
        self.assertNotIn("Counter", recommendations[0].get("status", ""))

if __name__ == '__main__':
    unittest.main()
