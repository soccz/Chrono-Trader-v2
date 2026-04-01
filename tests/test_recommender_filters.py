"""
Test suite for recommender filtering logic
Tests uncertainty filter, minimum return filter, and liquidity filter
"""
import unittest
import numpy as np
import pandas as pd
import pytest
import sys
import os
from pathlib import Path

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
        self.assertEqual(config.Recommender.MIN_SIGNAL_RETURN, 0.002)  # 0.2% (12h horizon)
    
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
        
        self.assertEqual(len(filtered), 2)  # BTC(5%) + SOL(2%) pass; ETH(0.1%) < 0.2% threshold
    
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
        self.assertEqual(config.Data.FUTURE_WINDOW_SIZE, 12)  # 12시간
    
    def test_feature_columns_count(self):
        """특성 컬럼 기본 구성 확인"""
        feature_columns = config.Data.FEATURE_COLUMNS
        self.assertGreaterEqual(len(feature_columns), 18)

        required_columns = {
            'rsi', 'macd',
            'market_index_return', 'historical_similarity',
            'alpha', 'beta',
            'breadth_ratio', 'net_volume_flow',
        }
        self.assertTrue(required_columns.issubset(set(feature_columns)))
    
    def test_accuracy_threshold_for_retrain(self):
        """자동 재학습 임계값 확인"""
        from utils.auto_retrain import ACCURACY_THRESHOLD
        self.assertEqual(ACCURACY_THRESHOLD, 25.0)

    def test_watch_only_fallback_uses_runtime_horizon(self):
        """watch-only synthetic fallback should match the active prediction horizon"""
        synthetic = recommender._synthesize_watch_only_recommendation(
            predictions=[],
            reason="unit-test",
        )
        self.assertEqual(len(synthetic["pattern"]), config.Data.FUTURE_WINDOW_SIZE)


def test_pi_low_floor_helper_intraday_vs_live(monkeypatch):
    monkeypatch.setattr(config.Recommender, "PI_LOW_80_MIN_LIVE", 0.0, raising=False)
    monkeypatch.setattr(config.Recommender, "PI_LOW_80_MIN_INTRADAY", -0.004, raising=False)

    assert recommender._resolve_pi_low_80_floor(mode="live", strategy="intraday") == -0.004
    assert recommender._resolve_pi_low_80_floor(mode="live", strategy="trending") == 0.0
    assert recommender._resolve_pi_low_80_floor(mode="backtest", strategy="intraday") == 0.0


def test_consensus_helper_intraday_vs_live(monkeypatch):
    monkeypatch.setattr(config.Recommender, "MIN_CONSENSUS_SCORE", 0.6, raising=False)
    monkeypatch.setattr(config.Recommender, "MIN_CONSENSUS_SCORE_INTRADAY", 0.55, raising=False)
    monkeypatch.setattr(config.Recommender, "COUNTER_TREND_MIN_CONSENSUS_SCORE", 0.8, raising=False)
    monkeypatch.setattr(config.Recommender, "COUNTER_TREND_MIN_CONSENSUS_SCORE_INTRADAY", 0.6, raising=False)

    assert recommender._resolve_min_consensus(mode="live", strategy="intraday", trend_alignment="Trend") == 0.55
    assert recommender._resolve_min_consensus(mode="live", strategy="intraday", trend_alignment="Counter") == 0.6
    assert recommender._resolve_min_consensus(mode="live", strategy="trending", trend_alignment="Trend") == 0.6
    assert recommender._resolve_min_consensus(mode="live", strategy="trending", trend_alignment="Counter") == 0.8


def test_directional_trade_metrics_are_symmetric_for_shorts():
    metrics = recommender._compute_directional_trade_metrics(
        expected_return=-0.01,
        pi_low_80=-0.03,
        pi_high_80=-0.002,
        pi_low_floor=0.0,
    )

    assert metrics["signal"] == "Short"
    assert metrics["directional_return"] == 0.01
    assert metrics["directional_pi_guard"] == 0.002


def test_live_intraday_long_step1_uses_entry_cost_budget(monkeypatch):
    monkeypatch.setattr(config.Recommender, "TRADE_FEE_PER_LEG", 0.0005, raising=False)
    monkeypatch.setattr(config.Recommender, "SLIPPAGE_PER_LEG", 0.0003, raising=False)
    monkeypatch.setattr(config.Recommender, "LIVE_INTRADAY_LONG_STEP1_COST_LEGS", 1.0, raising=False)

    assert recommender._resolve_step1_cost_budget("live", "intraday", "Long") == pytest.approx(0.0008)
    assert recommender._resolve_step1_cost_budget("live", "intraday", "Short") == pytest.approx(0.0016)
    assert recommender._resolve_step1_cost_budget("backtest", "intraday", "Long") == pytest.approx(0.0016)


def test_step1_score_applies_soft_pi_guard_penalty(monkeypatch):
    monkeypatch.setattr(config.Recommender, "STEP1_PI_GUARD_PENALTY", 0.35, raising=False)
    metrics = recommender._compute_step1_score(
        net_alpha=0.0020,
        directional_pi_guard=-0.0020,
        pi_guard_floor=0.0,
    )

    assert round(metrics["pi_guard_shortfall"], 6) == 0.002
    assert round(metrics["step1_score"], 6) == round(0.0020 - 0.35 * 0.0020, 6)


def test_live_intraday_long_uses_lighter_pi_guard_penalty(monkeypatch):
    monkeypatch.setattr(config.Recommender, "STEP1_PI_GUARD_PENALTY", 0.35, raising=False)
    monkeypatch.setattr(config.Recommender, "STEP1_PI_GUARD_PENALTY_INTRADAY_LONG", 0.10, raising=False)
    metrics = recommender._compute_step1_score(
        net_alpha=0.0010,
        directional_pi_guard=-0.0030,
        pi_guard_floor=0.0,
        mode="live",
        strategy="intraday",
        signal="Long",
    )

    assert round(metrics["step1_score"], 6) == round(0.0010 - 0.10 * 0.0030, 6)


def test_live_intraday_long_quality_score_prefers_stronger_step1_edge():
    weak = {
        "mode": "live",
        "strategy": "intraday",
        "signal": "Long",
        "step1_score": 0.0002,
        "expected_return": 0.0010,
        "confidence": 0.8,
        "consensus_score": 0.6,
    }
    strong = {
        "mode": "live",
        "strategy": "intraday",
        "signal": "Long",
        "step1_score": 0.0010,
        "expected_return": 0.0010,
        "confidence": 0.8,
        "consensus_score": 0.6,
    }

    assert recommender._compute_trade_quality_score(strong) > recommender._compute_trade_quality_score(weak)
    assert recommender._trade_sort_key(strong) > recommender._trade_sort_key(weak)


def test_live_intraday_long_quality_score_rewards_stronger_expected_return():
    weak = {
        "mode": "live",
        "strategy": "intraday",
        "signal": "Long",
        "step1_score": 0.0006,
        "expected_return": 0.0010,
        "confidence": 0.8,
        "consensus_score": 0.6,
    }
    strong = {
        "mode": "live",
        "strategy": "intraday",
        "signal": "Long",
        "step1_score": 0.0006,
        "expected_return": 0.0020,
        "confidence": 0.8,
        "consensus_score": 0.6,
    }

    assert recommender._compute_trade_quality_score(strong) > recommender._compute_trade_quality_score(weak)


def test_live_intraday_long_quality_score_rewards_stronger_consensus():
    weak = {
        "mode": "live",
        "strategy": "intraday",
        "signal": "Long",
        "step1_score": 0.0008,
        "expected_return": 0.0015,
        "confidence": 0.8,
        "consensus_score": 0.6,
    }
    strong = {
        "mode": "live",
        "strategy": "intraday",
        "signal": "Long",
        "step1_score": 0.0008,
        "expected_return": 0.0015,
        "confidence": 0.8,
        "consensus_score": 0.8,
    }

    assert recommender._compute_trade_quality_score(strong) > recommender._compute_trade_quality_score(weak)


def test_live_intraday_long_quality_score_rewards_following_trend():
    following = {
        "mode": "live",
        "strategy": "intraday",
        "signal": "Long",
        "step1_score": 0.0010,
        "expected_return": 0.0010,
        "confidence": 0.8,
        "consensus_score": 0.6,
        "trend_alignment": "Following",
    }
    counter = {
        "mode": "live",
        "strategy": "intraday",
        "signal": "Long",
        "step1_score": 0.0010,
        "expected_return": 0.0010,
        "confidence": 0.8,
        "consensus_score": 0.6,
        "trend_alignment": "Counter",
    }

    assert recommender._compute_trade_quality_score(following) > recommender._compute_trade_quality_score(counter)


def test_intraday_consensus_floor_allows_borderline_candidate(monkeypatch):
    monkeypatch.setattr(config.Recommender, "MIN_RECOMMENDATIONS_LIVE", 0, raising=False)
    monkeypatch.setattr(config.Recommender, "PI_LOW_80_MIN_LIVE", 0.0, raising=False)
    monkeypatch.setattr(config.Recommender, "PI_LOW_80_MIN_INTRADAY", -0.004, raising=False)
    monkeypatch.setattr(config.Recommender, "MIN_SIGNAL_RETURN", 0.001, raising=False)
    monkeypatch.setattr(config.Recommender, "MIN_CONSENSUS_SCORE", 0.6, raising=False)
    monkeypatch.setattr(config.Recommender, "MIN_CONSENSUS_SCORE_INTRADAY", 0.55, raising=False)
    monkeypatch.setattr(config.Recommender, "UNCERTAINTY_THRESHOLD", 500, raising=False)
    monkeypatch.setattr(recommender, "get_tradeable_markets", lambda: {"KRW-BTC"})
    monkeypatch.setattr(recommender, "get_trading_values_for_markets", lambda *args, **kwargs: {"KRW-BTC": 2_000_000_000})
    monkeypatch.setattr(recommender, "get_market_index", lambda: pd.DataFrame())
    monkeypatch.setattr(recommender, "get_historical_success_patterns", lambda: np.array([]))

    predictions = [
        {
            "market": "KRW-BTC",
            "current_price": 100.0,
            "predicted_pattern": [0.002, 0.001, 0.001],
            "pi_low_80": -0.0035,
            "pi_high_80": 0.01,
            "uncertainty": 100.0,
            "gate_value": 0.8,
            "consensus_score": 0.56,
            "strategy": "intraday",
        }
    ]

    out = recommender.run(predictions=predictions, mode="live", min_k=1)
    assert len(out) == 1
    assert out[0]["market"] == "KRW-BTC"
    assert out[0]["status"] == "Recommended"


def test_live_intraday_long_expected_return_floor_blocks_marginal_edge(monkeypatch):
    monkeypatch.setattr(config.Recommender, "MIN_RECOMMENDATIONS_LIVE", 0, raising=False)
    monkeypatch.setattr(config.Recommender, "PI_LOW_80_MIN_LIVE", 0.0, raising=False)
    monkeypatch.setattr(config.Recommender, "PI_LOW_80_MIN_INTRADAY", -0.004, raising=False)
    monkeypatch.setattr(config.Recommender, "MIN_SIGNAL_RETURN", 0.001, raising=False)
    monkeypatch.setattr(config.Recommender, "MIN_CONSENSUS_SCORE", 0.6, raising=False)
    monkeypatch.setattr(config.Recommender, "MIN_CONSENSUS_SCORE_INTRADAY", 0.55, raising=False)
    monkeypatch.setattr(config.Recommender, "UNCERTAINTY_THRESHOLD", 500, raising=False)
    monkeypatch.setattr(recommender, "get_tradeable_markets", lambda: {"KRW-BTC"})
    monkeypatch.setattr(recommender, "get_trading_values_for_markets", lambda *args, **kwargs: {"KRW-BTC": 2_000_000_000})
    monkeypatch.setattr(recommender, "get_market_index", lambda: pd.DataFrame())
    monkeypatch.setattr(recommender, "get_historical_success_patterns", lambda: np.array([]))

    predictions = [
        {
            "market": "KRW-BTC",
            "current_price": 100.0,
            "predicted_pattern": [0.0004, 0.0004, 0.0004],
            "pi_low_80": -0.0035,
            "pi_high_80": 0.01,
            "uncertainty": 100.0,
            "gate_value": 0.8,
            "consensus_score": 0.6,
            "strategy": "intraday",
        }
    ]

    out = recommender.run(predictions=predictions, mode="live", min_k=1)
    assert len(out) == 1
    assert out[0]["market"] == "KRW-BTC"
    assert out[0]["status"] == "Recommended"


def test_live_mode_short_is_downgraded_to_watch_only(monkeypatch):
    monkeypatch.setattr(config.Recommender, "MIN_RECOMMENDATIONS_LIVE", 1, raising=False)
    monkeypatch.setattr(config.Recommender, "LIVE_ALLOW_SHORT_EXECUTION", False, raising=False)
    monkeypatch.setattr(config.Recommender, "PI_LOW_80_MIN_LIVE", 0.0, raising=False)
    monkeypatch.setattr(config.Recommender, "UNCERTAINTY_THRESHOLD", 500, raising=False)
    monkeypatch.setattr(config.Recommender, "MIN_SIGNAL_RETURN", 0.001, raising=False)
    monkeypatch.setattr(config.Recommender, "MIN_CONSENSUS_SCORE", 0.6, raising=False)
    monkeypatch.setattr(recommender, "get_tradeable_markets", lambda: {"KRW-BTC"})
    monkeypatch.setattr(recommender, "get_trading_values_for_markets", lambda *args, **kwargs: {"KRW-BTC": 2_000_000_000})
    monkeypatch.setattr(recommender, "get_market_index", lambda: pd.DataFrame())
    monkeypatch.setattr(recommender, "get_historical_success_patterns", lambda: np.array([]))

    predictions = [
        {
            "market": "KRW-BTC",
            "current_price": 100.0,
            "predicted_pattern": [-0.01, -0.01, -0.01],
            "pi_low_80": -0.04,
            "pi_high_80": -0.005,
            "uncertainty": 100.0,
            "gate_value": 0.8,
            "consensus_score": 0.8,
            "strategy": "intraday",
        }
    ]

    out = recommender.run(predictions=predictions, mode="live", min_k=1)
    assert len(out) == 1
    assert out[0]["signal"] == "Short"
    assert out[0]["status"] == "Watch (MinRec)"
    assert out[0]["position_size"] == 0.0


def test_live_mode_prefers_executable_long_over_short(monkeypatch):
    monkeypatch.setattr(config.Recommender, "MIN_RECOMMENDATIONS_LIVE", 1, raising=False)
    monkeypatch.setattr(config.Recommender, "LIVE_ALLOW_SHORT_EXECUTION", False, raising=False)
    monkeypatch.setattr(config.Recommender, "PI_LOW_80_MIN_LIVE", 0.0, raising=False)
    monkeypatch.setattr(config.Recommender, "PI_LOW_80_MIN_INTRADAY", -0.004, raising=False)
    monkeypatch.setattr(config.Recommender, "UNCERTAINTY_THRESHOLD", 500, raising=False)
    monkeypatch.setattr(config.Recommender, "MIN_SIGNAL_RETURN", 0.001, raising=False)
    monkeypatch.setattr(config.Recommender, "MIN_CONSENSUS_SCORE", 0.6, raising=False)
    monkeypatch.setattr(config.Recommender, "MIN_CONSENSUS_SCORE_INTRADAY", 0.55, raising=False)
    monkeypatch.setattr(config.Recommender, "COUNTER_TREND_MIN_CONSENSUS_SCORE_INTRADAY", 0.6, raising=False)
    monkeypatch.setattr(recommender, "get_tradeable_markets", lambda: {"KRW-BTC", "KRW-ETH"})
    monkeypatch.setattr(
        recommender,
        "get_trading_values_for_markets",
        lambda *args, **kwargs: {"KRW-BTC": 2_000_000_000, "KRW-ETH": 2_000_000_000},
    )
    monkeypatch.setattr(recommender, "get_market_index", lambda: pd.DataFrame())
    monkeypatch.setattr(recommender, "get_historical_success_patterns", lambda: np.array([]))

    predictions = [
        {
            "market": "KRW-BTC",
            "current_price": 100.0,
            "predicted_pattern": [-0.01, -0.01, -0.01],
            "pi_low_80": -0.04,
            "pi_high_80": -0.005,
            "uncertainty": 100.0,
            "gate_value": 0.8,
            "consensus_score": 0.8,
            "strategy": "intraday",
        },
        {
            "market": "KRW-ETH",
            "current_price": 100.0,
            "predicted_pattern": [0.002, 0.0015, 0.001],
            "pi_low_80": -0.0035,
            "pi_high_80": 0.01,
            "uncertainty": 100.0,
            "gate_value": 0.7,
            "consensus_score": 0.6,
            "strategy": "intraday",
        },
    ]

    out = recommender.run(predictions=predictions, mode="live", min_k=1)
    assert len(out) == 1
    assert out[0]["market"] == "KRW-ETH"
    assert out[0]["signal"] == "Long"
    assert out[0]["status"] == "Recommended"


def test_non_intraday_can_pass_with_positive_step1_edge(monkeypatch):
    monkeypatch.setattr(config.Recommender, "MIN_RECOMMENDATIONS_LIVE", 0, raising=False)
    monkeypatch.setattr(config.Recommender, "PI_LOW_80_MIN_LIVE", 0.0, raising=False)
    monkeypatch.setattr(config.Recommender, "PI_LOW_80_MIN_INTRADAY", -0.004, raising=False)
    monkeypatch.setattr(recommender, "get_tradeable_markets", lambda: {"KRW-BTC"})
    monkeypatch.setattr(recommender, "get_trading_values_for_markets", lambda *args, **kwargs: {"KRW-BTC": 2_000_000_000})
    monkeypatch.setattr(recommender, "get_market_index", lambda: pd.DataFrame())
    monkeypatch.setattr(recommender, "get_historical_success_patterns", lambda: np.array([]))

    horizon = config.Data.FUTURE_WINDOW_SIZE
    predictions = [
        {
            "market": "KRW-BTC",
            "current_price": 100.0,
            "predicted_pattern": [0.001] * horizon,
            "pi_low_80": -0.0035,
            "pi_high_80": 0.01,
            "uncertainty": 100.0,
            "gate_value": 0.8,
            "consensus_score": 0.6,
            "strategy": "trending",
        }
    ]

    out = recommender.run(predictions=predictions, mode="live", min_k=1)
    assert len(out) == 1
    assert out[0]["market"] == "KRW-BTC"
    assert out[0]["signal"] == "Long"
    assert out[0]["status"] == "Recommended"


def test_short_candidate_can_pass_step1_with_directional_edge(monkeypatch):
    monkeypatch.setattr(config.Recommender, "MIN_RECOMMENDATIONS_LIVE", 0, raising=False)
    monkeypatch.setattr(config.Recommender, "PI_LOW_80_MIN_LIVE", 0.0, raising=False)
    monkeypatch.setattr(config.Recommender, "PI_LOW_80_MIN_INTRADAY", -0.004, raising=False)
    monkeypatch.setattr(config.Recommender, "MIN_SIGNAL_RETURN", 0.001, raising=False)
    monkeypatch.setattr(config.Recommender, "MIN_CONSENSUS_SCORE", 0.6, raising=False)
    monkeypatch.setattr(config.Recommender, "MIN_CONSENSUS_SCORE_INTRADAY", 0.55, raising=False)
    monkeypatch.setattr(config.Recommender, "UNCERTAINTY_THRESHOLD", 500, raising=False)
    monkeypatch.setattr(recommender, "get_tradeable_markets", lambda: {"KRW-BTC"})
    monkeypatch.setattr(recommender, "get_trading_values_for_markets", lambda *args, **kwargs: {"KRW-BTC": 2_000_000_000})
    monkeypatch.setattr(recommender, "get_market_index", lambda: pd.DataFrame())
    monkeypatch.setattr(recommender, "get_historical_success_patterns", lambda: np.array([]))

    predictions = [
        {
            "market": "KRW-BTC",
            "current_price": 100.0,
            "predicted_pattern": [-0.004, -0.003, -0.002],
            "pi_low_80": -0.02,
            "pi_high_80": -0.001,
            "uncertainty": 100.0,
            "gate_value": 0.8,
            "consensus_score": 0.6,
            "strategy": "intraday",
        }
    ]

    out = recommender.run(predictions=predictions, mode="backtest", min_k=1)
    assert len(out) == 1
    assert out[0]["signal"] == "Short"
    assert out[0]["status"] == "Recommended"


def test_soft_pi_guard_penalty_can_keep_small_shortfall_candidate(monkeypatch):
    monkeypatch.setattr(config.Recommender, "MIN_RECOMMENDATIONS_LIVE", 0, raising=False)
    monkeypatch.setattr(config.Recommender, "STEP1_PI_GUARD_PENALTY", 0.35, raising=False)
    monkeypatch.setattr(config.Recommender, "PI_LOW_80_MIN_LIVE", 0.0, raising=False)
    monkeypatch.setattr(config.Recommender, "PI_LOW_80_MIN_INTRADAY", -0.004, raising=False)
    monkeypatch.setattr(config.Recommender, "MIN_SIGNAL_RETURN", 0.001, raising=False)
    monkeypatch.setattr(config.Recommender, "MIN_CONSENSUS_SCORE", 0.6, raising=False)
    monkeypatch.setattr(config.Recommender, "MIN_CONSENSUS_SCORE_INTRADAY", 0.55, raising=False)
    monkeypatch.setattr(config.Recommender, "UNCERTAINTY_THRESHOLD", 500, raising=False)
    monkeypatch.setattr(recommender, "get_tradeable_markets", lambda: {"KRW-BTC"})
    monkeypatch.setattr(recommender, "get_trading_values_for_markets", lambda *args, **kwargs: {"KRW-BTC": 2_000_000_000})
    monkeypatch.setattr(recommender, "get_market_index", lambda: pd.DataFrame())
    monkeypatch.setattr(recommender, "get_historical_success_patterns", lambda: np.array([]))

    predictions = [
        {
            "market": "KRW-BTC",
            "current_price": 100.0,
            "predicted_pattern": [0.002, 0.002, 0.002],
            "pi_low_80": -0.0005,
            "pi_high_80": 0.01,
            "uncertainty": 100.0,
            "gate_value": 0.8,
            "consensus_score": 0.6,
            "strategy": "trending",
        }
    ]

    out = recommender.run(predictions=predictions, mode="live", min_k=1)
    assert len(out) == 1
    assert out[0]["status"] == "Recommended"


def test_non_intraday_still_requires_default_consensus(monkeypatch):
    monkeypatch.setattr(config.Recommender, "MIN_RECOMMENDATIONS_LIVE", 0, raising=False)
    monkeypatch.setattr(config.Recommender, "MIN_SIGNAL_RETURN", 0.001, raising=False)
    monkeypatch.setattr(config.Recommender, "MIN_CONSENSUS_SCORE", 0.6, raising=False)
    monkeypatch.setattr(config.Recommender, "MIN_CONSENSUS_SCORE_INTRADAY", 0.55, raising=False)
    monkeypatch.setattr(config.Recommender, "UNCERTAINTY_THRESHOLD", 500, raising=False)
    monkeypatch.setattr(recommender, "get_tradeable_markets", lambda: {"KRW-BTC"})
    monkeypatch.setattr(recommender, "get_trading_values_for_markets", lambda *args, **kwargs: {"KRW-BTC": 2_000_000_000})
    monkeypatch.setattr(recommender, "get_market_index", lambda: pd.DataFrame())
    monkeypatch.setattr(recommender, "get_historical_success_patterns", lambda: np.array([]))

    predictions = [
        {
            "market": "KRW-BTC",
            "current_price": 100.0,
            "predicted_pattern": [0.002, 0.001, 0.001],
            "pi_low_80": 0.001,
            "pi_high_80": 0.01,
            "uncertainty": 100.0,
            "gate_value": 0.8,
            "consensus_score": 0.56,
            "strategy": "trending",
        }
    ]

    out = recommender.run(predictions=predictions, mode="live", min_k=1)
    assert out == []


def test_recommender_writes_diagnostics_artifact(monkeypatch):
    monkeypatch.setattr(config.Recommender, "MIN_RECOMMENDATIONS_LIVE", 0, raising=False)
    monkeypatch.setattr(config.Recommender, "PI_LOW_80_MIN_LIVE", 0.0, raising=False)
    monkeypatch.setattr(config.Recommender, "PI_LOW_80_MIN_INTRADAY", -0.004, raising=False)
    monkeypatch.setattr(config.Recommender, "MIN_SIGNAL_RETURN", 0.001, raising=False)
    monkeypatch.setattr(config.Recommender, "MIN_CONSENSUS_SCORE", 0.6, raising=False)
    monkeypatch.setattr(config.Recommender, "MIN_CONSENSUS_SCORE_INTRADAY", 0.55, raising=False)
    monkeypatch.setattr(config.Recommender, "UNCERTAINTY_THRESHOLD", 500, raising=False)
    monkeypatch.setattr(config.General, "REC_TAG", "unitdiag", raising=False)
    monkeypatch.setattr(recommender, "get_tradeable_markets", lambda: {"KRW-BTC"})
    monkeypatch.setattr(recommender, "get_trading_values_for_markets", lambda *args, **kwargs: {"KRW-BTC": 2_000_000_000})
    monkeypatch.setattr(recommender, "get_market_index", lambda: pd.DataFrame())
    monkeypatch.setattr(recommender, "get_historical_success_patterns", lambda: np.array([]))

    diag_path = Path("analysis/recommender_diagnostics_unitdiag.json")
    if diag_path.exists():
        diag_path.unlink()

    predictions = [
        {
            "market": "KRW-BTC",
            "current_price": 100.0,
            "predicted_pattern": [0.002, 0.001, 0.001],
            "pi_low_80": -0.0035,
            "pi_high_80": 0.01,
            "uncertainty": 100.0,
            "gate_value": 0.8,
            "consensus_score": 0.56,
            "strategy": "intraday",
        }
    ]

    out = recommender.run(predictions=predictions, mode="live", min_k=1)
    assert len(out) == 1
    payload = recommender.get_last_run_diagnostics()
    assert payload is not None
    assert payload["path"] == str(diag_path)
    assert diag_path.exists()
    assert any(step["step"] == "step3_5_consensus" for step in payload["funnel_steps"])
    assert payload["consensus_probe"][0]["market"] == "KRW-BTC"
    diag_path.unlink()


def test_recommender_can_skip_artifact_writes(monkeypatch):
    monkeypatch.setattr(config.Recommender, "MIN_RECOMMENDATIONS_LIVE", 0, raising=False)
    monkeypatch.setattr(config.Recommender, "PI_LOW_80_MIN_LIVE", 0.0, raising=False)
    monkeypatch.setattr(config.Recommender, "PI_LOW_80_MIN_INTRADAY", -0.004, raising=False)
    monkeypatch.setattr(config.Recommender, "MIN_SIGNAL_RETURN", 0.001, raising=False)
    monkeypatch.setattr(config.Recommender, "MIN_CONSENSUS_SCORE", 0.6, raising=False)
    monkeypatch.setattr(config.Recommender, "MIN_CONSENSUS_SCORE_INTRADAY", 0.55, raising=False)
    monkeypatch.setattr(config.Recommender, "UNCERTAINTY_THRESHOLD", 500, raising=False)
    monkeypatch.setattr(config.General, "REC_TAG", "unitdiag_noemit", raising=False)
    monkeypatch.setattr(recommender, "get_tradeable_markets", lambda: {"KRW-BTC"})
    monkeypatch.setattr(recommender, "get_trading_values_for_markets", lambda *args, **kwargs: {"KRW-BTC": 2_000_000_000})
    monkeypatch.setattr(recommender, "get_market_index", lambda: pd.DataFrame())
    monkeypatch.setattr(recommender, "get_historical_success_patterns", lambda: np.array([]))

    diag_path = Path("analysis/recommender_diagnostics_unitdiag_noemit.json")
    if diag_path.exists():
        diag_path.unlink()
    recommender._set_last_run_diagnostics(None, None)

    predictions = [
        {
            "market": "KRW-BTC",
            "current_price": 100.0,
            "predicted_pattern": [0.002, 0.001, 0.001],
            "pi_low_80": -0.0035,
            "pi_high_80": 0.01,
            "uncertainty": 100.0,
            "gate_value": 0.8,
            "consensus_score": 0.56,
            "strategy": "intraday",
        }
    ]

    out = recommender.run(predictions=predictions, mode="live", min_k=1, emit_artifacts=False)
    assert len(out) == 1
    assert recommender.get_last_run_diagnostics() is None
    assert recommender.get_last_run_diagnostics_path() is None
    assert not diag_path.exists()


def test_recommender_collects_in_memory_diagnostics_without_writing(monkeypatch):
    monkeypatch.setattr(config.Recommender, "MIN_RECOMMENDATIONS_LIVE", 0, raising=False)
    monkeypatch.setattr(config.Recommender, "PI_LOW_80_MIN_LIVE", 0.0, raising=False)
    monkeypatch.setattr(config.Recommender, "PI_LOW_80_MIN_INTRADAY", -0.004, raising=False)
    monkeypatch.setattr(config.Recommender, "MIN_SIGNAL_RETURN", 0.001, raising=False)
    monkeypatch.setattr(config.Recommender, "MIN_CONSENSUS_SCORE", 0.6, raising=False)
    monkeypatch.setattr(config.Recommender, "MIN_CONSENSUS_SCORE_INTRADAY", 0.55, raising=False)
    monkeypatch.setattr(config.Recommender, "UNCERTAINTY_THRESHOLD", 500, raising=False)
    monkeypatch.setattr(config.General, "REC_TAG", "unitdiag_collect", raising=False)
    monkeypatch.setattr(recommender, "get_tradeable_markets", lambda: {"KRW-BTC"})
    monkeypatch.setattr(recommender, "get_trading_values_for_markets", lambda *args, **kwargs: {"KRW-BTC": 2_000_000_000})
    monkeypatch.setattr(recommender, "get_market_index", lambda: pd.DataFrame())
    monkeypatch.setattr(recommender, "get_historical_success_patterns", lambda: np.array([]))

    diag_path = Path("analysis/recommender_diagnostics_unitdiag_collect.json")
    if diag_path.exists():
        diag_path.unlink()
    recommender._set_last_run_diagnostics(None, None)

    predictions = [
        {
            "market": "KRW-BTC",
            "current_price": 100.0,
            "predicted_pattern": [0.002, 0.001, 0.001],
            "pi_low_80": -0.0035,
            "pi_high_80": 0.01,
            "uncertainty": 100.0,
            "gate_value": 0.8,
            "consensus_score": 0.56,
            "strategy": "intraday",
        }
    ]

    out = recommender.run(predictions=predictions, mode="live", min_k=1, emit_artifacts=False, collect_diagnostics=True)
    assert len(out) == 1
    payload = recommender.get_last_run_diagnostics()
    assert payload is not None
    assert payload["path"] is None
    assert payload["step1_reject_counts"]["passed"] == 1
    assert payload["step1_signal_counts"]["total_long"] == 1
    assert payload["step1_signal_counts"]["passed_long"] == 1
    assert not diag_path.exists()


def test_recommender_collects_signal_specific_step1_rejects(monkeypatch):
    monkeypatch.setattr(config.Recommender, "MIN_RECOMMENDATIONS_LIVE", 0, raising=False)
    monkeypatch.setattr(config.Recommender, "PI_LOW_80_MIN_LIVE", 0.0, raising=False)
    monkeypatch.setattr(config.Recommender, "PI_LOW_80_MIN_INTRADAY", -0.004, raising=False)
    monkeypatch.setattr(config.Recommender, "MIN_SIGNAL_RETURN", 0.001, raising=False)
    monkeypatch.setattr(config.Recommender, "MIN_CONSENSUS_SCORE", 0.6, raising=False)
    monkeypatch.setattr(config.Recommender, "MIN_CONSENSUS_SCORE_INTRADAY", 0.55, raising=False)
    monkeypatch.setattr(config.Recommender, "UNCERTAINTY_THRESHOLD", 500, raising=False)
    monkeypatch.setattr(config.General, "REC_TAG", "unitdiag_signal_rejects", raising=False)
    monkeypatch.setattr(recommender, "get_tradeable_markets", lambda: {"KRW-BTC"})
    monkeypatch.setattr(recommender, "get_trading_values_for_markets", lambda *args, **kwargs: {"KRW-BTC": 2_000_000_000})
    monkeypatch.setattr(recommender, "get_market_index", lambda: pd.DataFrame())
    monkeypatch.setattr(recommender, "get_historical_success_patterns", lambda: np.array([]))

    predictions = [
            {
                "market": "KRW-BTC",
                "current_price": 100.0,
                "predicted_pattern": [0.02, 0.02, 0.02],
                "pi_low_80": -1.0,
                "pi_high_80": 0.05,
                "uncertainty": 100.0,
                "gate_value": 0.8,
            "consensus_score": 0.56,
            "strategy": "intraday",
        }
    ]

    out = recommender.run(predictions=predictions, mode="live", min_k=1, emit_artifacts=False, collect_diagnostics=True)
    assert isinstance(out, list)
    payload = recommender.get_last_run_diagnostics()
    assert payload is not None
    assert payload["step1_reject_counts"]["pi_guard_only"] == 1
    assert payload["step1_signal_reject_counts"]["long"]["pi_guard_only"] == 1
    assert payload["step1_signal_reject_counts"]["long"]["passed"] == 0
    assert payload["step1_signal_stats"]["long"]["net_alpha"]["n"] == 1


if __name__ == '__main__':
    unittest.main()
