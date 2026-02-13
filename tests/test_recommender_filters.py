"""
Test suite for recommender filtering logic
Tests uncertainty filter, minimum return filter, and liquidity filter
"""
import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import config
from inference import recommender


class TestRecommenderFilters(unittest.TestCase):
    """추천 엔진의 필터 로직 테스트"""
    
    def test_uncertainty_threshold_config(self):
        """불확실성 임계값 설정 확인"""
        threshold = config.Recommender.UNCERTAINTY_THRESHOLD
        self.assertIsInstance(threshold, (int, float))
        self.assertGreater(threshold, 0)
    
    def test_min_signal_return_config(self):
        """최소 수익률 임계값 설정 확인"""
        self.assertEqual(config.Recommender.MIN_SIGNAL_RETURN, 0.001)  # 0.1%
    
    def test_liquidity_thresholds_config(self):
        """유동성 임계값 설정 확인"""
        self.assertEqual(config.Recommender.LIQUIDITY_THRESHOLDS['live'], 1_000_000_000)
        self.assertEqual(config.Recommender.LIQUIDITY_THRESHOLDS['backtest'], 50_000_000)
    
    def test_uncertainty_filter_logic(self):
        """불확실성 필터 로직 검증"""
        threshold = float(config.Recommender.UNCERTAINTY_THRESHOLD)
        predictions = [
            {"market": "KRW-BTC", "uncertainty": threshold * 0.5},   # Pass (< threshold)
            {"market": "KRW-ETH", "uncertainty": threshold * 0.99},  # Pass (< threshold)
            {"market": "KRW-XRP", "uncertainty": threshold},         # Fail (= threshold)
            {"market": "KRW-SOL", "uncertainty": threshold * 1.2},   # Fail (> threshold)
        ]
        
        filtered = [p for p in predictions if p["uncertainty"] < threshold]
        
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered[0]["market"], "KRW-BTC")
        self.assertEqual(filtered[1]["market"], "KRW-ETH")
    
    def test_min_return_filter_logic(self):
        """최소 수익률 필터 로직 검증"""
        predictions = [
            {"market": "KRW-BTC", "expected_return": 0.05},    # Pass (5%)
            {"market": "KRW-ETH", "expected_return": 0.001},   # Pass (0.1%)
            {"market": "KRW-XRP", "expected_return": 0.0005},  # Fail (0.05%)
            {"market": "KRW-SOL", "expected_return": -0.02},   # Pass (abs > 0.1%)
        ]
        
        threshold = config.Recommender.MIN_SIGNAL_RETURN
        filtered = [p for p in predictions if abs(p["expected_return"]) >= threshold]
        
        self.assertEqual(len(filtered), 3)
    
    def test_max_positions_config(self):
        """최대 동시 포지션 수 설정 확인"""
        self.assertEqual(config.Recommender.MAX_POSITIONS, 5)
    
    def test_monte_carlo_inferences_config(self):
        """Monte Carlo 추론 횟수 설정 확인"""
        self.assertEqual(config.Recommender.MC_N_INFERENCES, 20)

    def test_forced_topk_default_disabled(self):
        """실사용 안전성을 위해 강제 Top-K는 기본 비활성"""
        self.assertFalse(config.Recommender.FORCED_TOPK_ENABLED)
        self.assertFalse(config.Recommender.FORCED_TOPK_BACKTEST_ENABLED)

    def test_forced_topk_exclusion_rules(self):
        """강제 Top-K를 쓰더라도 핵심 위험 실패는 제외"""
        excluded = config.Recommender.FORCED_TOPK_EXCLUDE_FAILED_REASONS
        self.assertIn("High Uncertainty", excluded)
        self.assertIn("Low Liquidity", excluded)

    def test_dynamic_uncertainty_config_ranges(self):
        """동적 불확실성 임계값 설정 범위 검증"""
        self.assertTrue(config.Recommender.ENABLE_DYNAMIC_UNCERTAINTY_THRESHOLD)
        self.assertGreaterEqual(config.Recommender.DYNAMIC_UNCERTAINTY_QUANTILE, 0.1)
        self.assertLessEqual(config.Recommender.DYNAMIC_UNCERTAINTY_QUANTILE, 0.95)
        self.assertGreater(config.Recommender.DYNAMIC_UNCERTAINTY_MIN_MULTIPLIER, 0)
        self.assertGreaterEqual(
            config.Recommender.DYNAMIC_UNCERTAINTY_MAX_MULTIPLIER,
            config.Recommender.DYNAMIC_UNCERTAINTY_MIN_MULTIPLIER
        )

    def test_dynamic_uncertainty_threshold_computation(self):
        """동적 임계값 계산이 설정된 클램프 범위를 지키는지 확인"""
        base = float(config.Recommender.UNCERTAINTY_THRESHOLD)
        min_mult = float(config.Recommender.DYNAMIC_UNCERTAINTY_MIN_MULTIPLIER)
        max_mult = float(config.Recommender.DYNAMIC_UNCERTAINTY_MAX_MULTIPLIER)
        funnel_data = [
            {"status": "Initial Candidate", "uncertainty": 650.0},
            {"status": "Initial Candidate", "uncertainty": 900.0},
            {"status": "Initial Candidate", "uncertainty": 1400.0},
            {"status": "Initial Candidate", "uncertainty": 2200.0},
        ]
        th = recommender._compute_uncertainty_threshold(funnel_data, base)
        self.assertGreaterEqual(th, base * min_mult)
        self.assertLessEqual(th, base * max_mult)


class TestConfigIntegrity(unittest.TestCase):
    """설정 파일 무결성 테스트"""
    
    def test_model_architecture_config(self):
        """모델 아키텍처 설정 확인"""
        self.assertEqual(config.Gan.D_MODEL, 128)
        self.assertEqual(config.Gan.N_HEADS, 8)
        self.assertEqual(config.Gan.N_LAYERS, 3)
    
    def test_sequence_length_config(self):
        """시퀀스 길이 설정 확인"""
        self.assertEqual(config.Data.SEQUENCE_LENGTH, 168)  # 7일
        self.assertEqual(config.Data.FUTURE_WINDOW_SIZE, 6)  # 6시간
    
    def test_feature_columns_count(self):
        """특성 컬럼 기본 구성 확인"""
        feature_columns = config.Data.FEATURE_COLUMNS
        self.assertGreaterEqual(len(feature_columns), 18)

        required_columns = {
            'close', 'volume', 'rsi', 'macd',
            'market_index_return', 'historical_similarity',
            'alpha', 'beta'
        }
        self.assertTrue(required_columns.issubset(set(feature_columns)))
    
    def test_accuracy_threshold_for_retrain(self):
        """자동 재학습 임계값 확인"""
        from utils.auto_retrain import ACCURACY_THRESHOLD
        self.assertEqual(ACCURACY_THRESHOLD, 25.0)


if __name__ == '__main__':
    unittest.main()
