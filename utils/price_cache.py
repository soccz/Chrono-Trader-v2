"""
Price Cache Module - TTL-based batch price fetching for Upbit API
Solves N+1 query problem by fetching all prices in a single API call
"""
import requests
import time
from typing import Dict, List, Optional
from threading import Lock

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
            return result
        else:
            print(f"[PriceCache] Batch price fetch error: {e}")
            return result
    except requests.exceptions.RequestException as e:
        print(f"[PriceCache] Batch price fetch error: {e}")
        return result  # Return whatever we had from cache
    except Exception as e:
        print(f"[PriceCache] Unexpected error: {e}")
        return result


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
