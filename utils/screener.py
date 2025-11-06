
import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from utils.logger import logger
from utils.config import config

UPBIT_MARKET_API_URL = "https://api.upbit.com/v1/market/all"
UPBIT_CANDLE_API_URL = "https://api.upbit.com/v1/candles/days"

def get_trending_markets(historical_df: pd.DataFrame = None, current_time: datetime = None, limit: int = 5, lookback_days: int = 3, mode: str = 'live'):
    """ 
    Finds top trending markets based on a specified mode.
    - 'live': Queries the Upbit API for top traded markets.
    - 'backtest': Calculates volatility on the provided hourly data.
    """
    logger.info(f"--- Starting Market Screening (mode: {mode}, last {lookback_days} days) ---")
    
    # Use the mode-specific threshold from config, with a fallback to 'live'
    threshold = config.LIQUIDITY_THRESHOLDS.get(mode, config.LIQUIDITY_THRESHOLDS['live'])
    logger.info(f"Applying liquidity threshold: {threshold:,.0f} KRW")

    if historical_df is not None:
        # --- Backtest Mode ---
        try:
            if current_time is None:
                current_time = historical_df.index.max()

            lookback_start_time = current_time - timedelta(days=lookback_days)
            recent_data = historical_df[historical_df.index >= lookback_start_time]
            
            if recent_data.empty:
                logger.warning("No historical data in the lookback period for screening.")
                return []

            market_stats = recent_data.groupby('market').agg(
                volatility=('close', lambda x: x.std() / (x.mean() if x.mean() != 0 else 1e-9)),
                total_volume=('volume', 'sum')
            ).dropna()

            eligible_markets = market_stats[market_stats['total_volume'] > threshold]
            
            logger.info(f"Screening results: {len(eligible_markets)} candidates kept, {len(market_stats) - len(eligible_markets)} dropped.")

            if eligible_markets.empty:
                logger.warning("No markets met the minimum volume criteria in backtest screening.")
                return []

            top_markets = eligible_markets.sort_values(by='volatility', ascending=False).head(limit).index.tolist()
            logger.info(f"Screening complete (Backtest). Top {len(top_markets)} trending markets: {top_markets}")
            return top_markets
        except Exception as e:
            logger.error(f"Market screening failed during backtest: {e}")
            return []

    else:
        # --- Live Mode (using Upbit API) ---
        try:
            res = requests.get(UPBIT_MARKET_API_URL, params={"isWarning": "false"})
            res.raise_for_status()
            all_markets = res.json()
            krw_markets = [m['market'] for m in all_markets if m['market'].startswith('KRW')]
            logger.info(f"Found {len(krw_markets)} KRW markets.")
            time.sleep(0.2)

            market_volatility = []
            for market in krw_markets:
                params = {'market': market, 'count': lookback_days + 1}
                candle_res = requests.get(UPBIT_CANDLE_API_URL, params=params)
                candle_res.raise_for_status()
                candles = candle_res.json()
                time.sleep(0.2)

                if len(candles) < lookback_days + 1:
                    continue

                start_price = candles[lookback_days]['opening_price']
                end_price = candles[0]['trade_price']
                if start_price == 0:
                    continue

                change_pct = (end_price - start_price) / start_price
                trade_volume = candles[0]['candle_acc_trade_price']

                if trade_volume > threshold:
                    market_volatility.append({'market': market, 'abs_change': abs(change_pct)})
            
            logger.info(f"Screening results: {len(market_volatility)} candidates kept, {len(krw_markets) - len(market_volatility)} dropped.")

            market_volatility.sort(key=lambda x: x['abs_change'], reverse=True)
            top_markets = [m['market'] for m in market_volatility[:limit]]
            logger.info(f"Screening complete (Live). Top {len(top_markets)} trending markets: {top_markets}")
            return top_markets

        except Exception as e:
            logger.error(f"Market screening failed: {e}")
            return []

