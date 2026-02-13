"""
Test suite for price_cache module
Tests caching behavior, TTL, and fallback on invalid markets
"""
import unittest
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.price_cache import (
    get_prices_batch, 
    get_prices_batch_dashboard,
    invalidate_cache,
    CACHE_TTL_REALTIME,
    CACHE_TTL_DASHBOARD
)


class TestPriceCache(unittest.TestCase):
    
    def setUp(self):
        """각 테스트 전에 캐시 초기화"""
        invalidate_cache()
    
    def test_basic_price_fetch(self):
        """정상적인 마켓 코드로 가격 조회"""
        prices = get_prices_batch(["KRW-BTC", "KRW-ETH"])
        
        self.assertIn("KRW-BTC", prices)
        self.assertIn("KRW-ETH", prices)
        self.assertGreater(prices["KRW-BTC"], 0)
        self.assertGreater(prices["KRW-ETH"], 0)
    
    def test_fallback_on_invalid_market(self):
        """잘못된 마켓 코드가 포함되어도 다른 코인 가격은 정상 조회"""
        prices = get_prices_batch(["KRW-BTC", "KRW-INVALID-COIN-12345", "KRW-ETH"])
        
        # 유효한 코인은 가격이 있어야 함
        self.assertIn("KRW-BTC", prices)
        self.assertIn("KRW-ETH", prices)
        self.assertGreater(prices["KRW-BTC"], 0)
        
        # 잘못된 코인은 없거나 0이어야 함
        self.assertNotIn("KRW-INVALID-COIN-12345", prices)
    
    def test_cache_returns_same_value_within_ttl(self):
        """TTL 내에서는 동일한 캐시 값 반환"""
        # 첫 번째 호출
        prices1 = get_prices_batch(["KRW-BTC"])
        
        # 즉시 두 번째 호출 (캐시 사용해야 함)
        prices2 = get_prices_batch(["KRW-BTC"])
        
        # 같은 값이어야 함
        self.assertEqual(prices1["KRW-BTC"], prices2["KRW-BTC"])
    
    def test_empty_markets_list(self):
        """빈 리스트 입력 시 빈 딕셔너리 반환"""
        prices = get_prices_batch([])
        self.assertEqual(prices, {})
    
    def test_dashboard_cache_longer_ttl(self):
        """대시보드 캐시는 더 긴 TTL 사용"""
        self.assertGreater(CACHE_TTL_DASHBOARD, CACHE_TTL_REALTIME)
        self.assertEqual(CACHE_TTL_REALTIME, 10)
        self.assertEqual(CACHE_TTL_DASHBOARD, 60)


class TestPriceCacheDashboard(unittest.TestCase):
    
    def setUp(self):
        invalidate_cache()
    
    def test_dashboard_price_fetch(self):
        """대시보드용 가격 조회 정상 동작"""
        prices = get_prices_batch_dashboard(["KRW-BTC", "KRW-ETH"])
        
        self.assertIn("KRW-BTC", prices)
        self.assertIn("KRW-ETH", prices)


if __name__ == '__main__':
    unittest.main()
