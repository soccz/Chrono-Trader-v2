"""
Price Cache Module - TTL-based batch price fetching for Upbit API
Solves N+1 query problem by fetching all prices in a single API call
"""
import requests
import time
import sqlite3
from typing import Dict, List, Optional
from threading import Lock
from utils.config import config

# Cache storage
_price_cache: Dict[str, float] = {}
_cache_timestamp: float = 0
_dashboard_cache: Dict[str, float] = {}
_dashboard_cache_timestamp: float = 0
_cache_lock = Lock()

# Configuration - 용도별 TTL 분리
CACHE_TTL_REALTIME = 10   # 실시간 포지션용 (10초)
CACHE_TTL_DASHBOARD = 60  # 대시보드 표시용 (60초)
UPBIT_TICKER_URL = "https://api.upbit.com/v1/ticker"


def _get_latest_prices_from_db(markets: List[str]) -> Dict[str, float]:
    """Fallback: read latest close prices from local SQLite market data."""
    if not markets:
        return {}

    placeholders = ",".join("?" for _ in markets)
    query = f"""
        SELECT t.market, t.close
        FROM crypto_data t
        INNER JOIN (
            SELECT market, MAX(timestamp) AS max_ts
            FROM crypto_data
            WHERE market IN ({placeholders})
            GROUP BY market
        ) latest
        ON t.market = latest.market AND t.timestamp = latest.max_ts
    """

    try:
        conn = sqlite3.connect(config.General.DB_PATH)
        cursor = conn.cursor()
        cursor.execute(query, markets)
        rows = cursor.fetchall()
        return {
            market: float(close)
            for market, close in rows
            if close is not None and float(close) > 0
        }
    except Exception as e:
        print(f"[PriceCache] Local DB fallback price fetch error: {e}")
        return {}
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _get_tradeable_markets_from_db() -> set:
    """Fallback: infer tradeable KRW markets from local SQLite market data."""
    try:
        conn = sqlite3.connect(config.General.DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT market FROM crypto_data WHERE market LIKE 'KRW-%'")
        rows = cursor.fetchall()
        return {row[0] for row in rows if row and row[0]}
    except Exception as e:
        print(f"[PriceCache] Local DB fallback tradeable markets error: {e}")
        return set()
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _apply_db_price_fallback(markets: List[str], result: Dict[str, float], current_time: float) -> Dict[str, float]:
    """Merge local DB fallback prices and refresh in-memory cache if found."""
    global _price_cache, _cache_timestamp
    fallback_prices = _get_latest_prices_from_db(markets)
    if not fallback_prices:
        return result

    with _cache_lock:
        _price_cache.update(fallback_prices)
        _cache_timestamp = current_time
    result.update(fallback_prices)
    return result


def get_prices_batch(markets: List[str]) -> Dict[str, float]:
    """
    Fetch prices for multiple markets in a single API call.
    Uses TTL-based caching to reduce API calls.
    
    Args:
        markets: List of market symbols (e.g., ['KRW-BTC', 'KRW-ETH'])
    
    Returns:
        Dict mapping market -> price
    """
    global _price_cache, _cache_timestamp
    
    if not markets:
        return {}
    
    current_time = time.time()
    
    # Check if cache is still valid
    with _cache_lock:
        if current_time - _cache_timestamp < CACHE_TTL_REALTIME:
            # Return cached prices for requested markets
            result = {}
            missing_markets = []
            for market in markets:
                if market in _price_cache:
                    result[market] = _price_cache[market]
                else:
                    missing_markets.append(market)
            
            # If all markets are cached, return immediately
            if not missing_markets:
                return result
            
            # Otherwise, fetch missing markets
            markets_to_fetch = missing_markets
        else:
            # Cache expired, fetch all requested markets
            markets_to_fetch = markets
            result = {}
    
    # Fetch prices from Upbit API
    try:
        markets_param = ",".join(markets_to_fetch)
        response = requests.get(
            UPBIT_TICKER_URL,
            params={"markets": markets_param},
            timeout=5
        )
        response.raise_for_status()
        data = response.json()
        
        fetched_prices = {}
        for item in data:
            market = item.get('market')
            price = item.get('trade_price')
            if market and price:
                fetched_prices[market] = float(price)
        
        # Update cache
        with _cache_lock:
            _price_cache.update(fetched_prices)
            _cache_timestamp = current_time
        
        # Merge with previously cached results
        result.update(fetched_prices)
        return result
        
    except requests.exceptions.HTTPError as e:
        # 404 error - some markets don't exist. Fallback to individual fetches.
        if e.response.status_code == 404 and len(markets_to_fetch) > 1:
            print(f"[PriceCache] Batch failed (404). Falling back to individual fetches...")
            fetched_prices = {}
            for market in markets_to_fetch:
                try:
                    resp = requests.get(UPBIT_TICKER_URL, params={"markets": market}, timeout=3)
                    if resp.ok:
                        data = resp.json()
                        if data and len(data) > 0:
                            fetched_prices[market] = float(data[0].get('trade_price', 0))
                except Exception:
                    pass  # Skip invalid markets
            
            # Update cache with fetched prices
            with _cache_lock:
                _price_cache.update(fetched_prices)
                _cache_timestamp = current_time
            
            result.update(fetched_prices)
            missing_markets = [m for m in markets_to_fetch if m not in fetched_prices]
            if missing_markets:
                result = _apply_db_price_fallback(missing_markets, result, current_time)
            return result
        else:
            print(f"[PriceCache] Batch price fetch error: {e}")
            return _apply_db_price_fallback(markets_to_fetch, result, current_time)
    except requests.exceptions.RequestException as e:
        print(f"[PriceCache] Batch price fetch error: {e}")
        return _apply_db_price_fallback(markets_to_fetch, result, current_time)
    except Exception as e:
        print(f"[PriceCache] Unexpected error: {e}")
        return _apply_db_price_fallback(markets_to_fetch, result, current_time)


def get_cached_price(market: str) -> Optional[float]:
    """
    Get a single market price from cache or fetch it.
    
    Args:
        market: Market symbol (e.g., 'KRW-BTC')
    
    Returns:
        Price as float or None if unavailable
    """
    prices = get_prices_batch([market])
    return prices.get(market)


def invalidate_cache():
    """Force invalidate the cache (useful for testing)"""
    global _price_cache, _cache_timestamp
    with _cache_lock:
        _price_cache = {}
        _cache_timestamp = 0


def get_all_krw_prices() -> Dict[str, float]:
    """
    Fetch all KRW market prices in a single call.
    Useful for market overview and initial load.
    
    Returns:
        Dict mapping market -> price for all KRW markets
    """
    global _price_cache, _cache_timestamp
    
    try:
        response = requests.get(
            "https://api.upbit.com/v1/ticker/all",
            params={"quoteCurrencies": "KRW"},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        
        prices = {}
        for item in data:
            market = item.get('market')
            price = item.get('trade_price')
            if market and price:
                prices[market] = float(price)
        
        # Update cache with all fetched prices
        with _cache_lock:
            _price_cache.update(prices)
            _cache_timestamp = time.time()
        
        return prices
        
    except Exception as e:
        print(f"[PriceCache] All KRW prices fetch error: {e}")
        return {}


def get_prices_batch_dashboard(markets: List[str]) -> Dict[str, float]:
    """
    대시보드 표시용 가격 조회 (60초 캐시).
    실시간 정확도보다 API 호출 최소화가 중요한 경우 사용.
    """
    global _dashboard_cache, _dashboard_cache_timestamp
    
    if not markets:
        return {}
    
    current_time = time.time()
    
    with _cache_lock:
        if current_time - _dashboard_cache_timestamp < CACHE_TTL_DASHBOARD:
            # 캐시된 가격 반환
            result = {m: _dashboard_cache[m] for m in markets if m in _dashboard_cache}
            if len(result) == len(markets):
                return result
    
    # 캐시 만료 또는 누락 - 전체 조회
    prices = get_all_krw_prices()
    
    with _cache_lock:
        _dashboard_cache.update(prices)
        _dashboard_cache_timestamp = current_time
    
    return {m: prices.get(m, 0) for m in markets}


def prefetch_dashboard_prices():
    """
    대시보드 로드 시 한 번 호출하여 모든 KRW 가격을 미리 캐싱.
    API 호출 횟수를 최소화합니다.
    """
    global _dashboard_cache, _dashboard_cache_timestamp
    
    prices = get_all_krw_prices()
    
    with _cache_lock:
        _dashboard_cache = prices
        _dashboard_cache_timestamp = time.time()
    
    print(f"[PriceCache] Prefetched {len(prices)} KRW market prices for dashboard")
    return prices


# --- Tradeable Markets Validation ---
_tradeable_markets_cache: set = set()
_tradeable_markets_timestamp: float = 0
TRADEABLE_CACHE_TTL = 3600  # 1 hour - market list doesn't change often


def get_tradeable_markets() -> set:
    """
    Fetch the list of currently tradeable KRW markets from Upbit.
    Used to filter out delisted or non-existent coins from recommendations.
    
    Returns:
        Set of market symbols (e.g., {'KRW-BTC', 'KRW-ETH', ...})
    """
    global _tradeable_markets_cache, _tradeable_markets_timestamp
    
    current_time = time.time()
    
    # Check cache validity
    with _cache_lock:
        if current_time - _tradeable_markets_timestamp < TRADEABLE_CACHE_TTL:
            if _tradeable_markets_cache:
                return _tradeable_markets_cache
    
    try:
        response = requests.get(
            "https://api.upbit.com/v1/market/all",
            params={"isDetails": "false"},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        
        # Filter only KRW markets
        tradeable = {item['market'] for item in data if item['market'].startswith('KRW-')}
        
        with _cache_lock:
            _tradeable_markets_cache = tradeable
            _tradeable_markets_timestamp = current_time
        
        print(f"[PriceCache] Fetched {len(tradeable)} tradeable KRW markets")
        return tradeable
        
    except Exception as e:
        print(f"[PriceCache] Failed to fetch tradeable markets: {e}")
        # Fallback to local DB markets when network is unavailable.
        db_tradeable = _get_tradeable_markets_from_db()
        if db_tradeable:
            with _cache_lock:
                _tradeable_markets_cache = db_tradeable
                _tradeable_markets_timestamp = current_time
            print(f"[PriceCache] Using local DB fallback for {len(db_tradeable)} tradeable KRW markets")
            return db_tradeable

        # Return cached data if available, else empty set.
        return _tradeable_markets_cache if _tradeable_markets_cache else set()


def is_market_tradeable(market: str) -> bool:
    """
    Check if a specific market is currently tradeable.
    
    Args:
        market: Market symbol (e.g., 'KRW-BTC')
    
    Returns:
        True if market is tradeable, False otherwise
    """
    tradeable_markets = get_tradeable_markets()
    return market in tradeable_markets
