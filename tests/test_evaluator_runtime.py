import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training import evaluator


def test_parse_recommender_mode_env_defaults_backtest(monkeypatch):
    monkeypatch.delenv("AETHER_BACKTEST_RECOMMENDER_MODE", raising=False)
    assert evaluator._parse_recommender_mode_env() == "backtest"


def test_parse_recommender_mode_env_accepts_live(monkeypatch):
    monkeypatch.setenv("AETHER_BACKTEST_RECOMMENDER_MODE", "live")
    assert evaluator._parse_recommender_mode_env() == "live"


def test_parse_strategy_env_normalizes_blank(monkeypatch):
    monkeypatch.setenv("AETHER_BACKTEST_STRATEGY", "  intraday  ")
    assert evaluator._parse_strategy_env() == "intraday"

    monkeypatch.setenv("AETHER_BACKTEST_STRATEGY", " ")
    assert evaluator._parse_strategy_env() is None


def test_parse_screen_limit_env_clamps_and_defaults(monkeypatch):
    monkeypatch.delenv("AETHER_BACKTEST_SCREEN_LIMIT", raising=False)
    assert evaluator._parse_screen_limit_env(default=7) == 7

    monkeypatch.setenv("AETHER_BACKTEST_SCREEN_LIMIT", "0")
    assert evaluator._parse_screen_limit_env(default=7) == 1


def test_parse_min_k_env_clamps_and_defaults(monkeypatch):
    monkeypatch.delenv("AETHER_BACKTEST_MIN_K", raising=False)
    assert evaluator._parse_min_k_env(default=3) == 3

    monkeypatch.setenv("AETHER_BACKTEST_MIN_K", "-2")
    assert evaluator._parse_min_k_env(default=3) == 1


def test_configured_future_horizon_hours_uses_runtime_config(monkeypatch):
    monkeypatch.setattr(evaluator.config.Data, "FUTURE_WINDOW_SIZE", 3, raising=False)
    assert evaluator._configured_future_horizon_hours() == 3


def test_is_watch_only_output_detects_watch_status():
    assert evaluator._is_watch_only_output({"status": "Watch (MinRec)", "position_size": 0.1}) is True


def test_is_watch_only_output_detects_zero_position_without_status():
    assert evaluator._is_watch_only_output({"status": "Recommended", "position_size": 0.0}) is True


def test_is_watch_only_output_allows_positive_position_trade():
    assert evaluator._is_watch_only_output({"status": "Recommended", "position_size": 0.05}) is False


def test_calc_trade_pnl_respects_short_direction():
    row = {"signal": "Short", "predicted_return": -0.01, "actual_return": -0.02}
    assert evaluator._calc_trade_pnl(row) == 0.02 - 0.0016


def test_calc_trade_pnl_infers_long_when_signal_missing():
    row = {"signal": "", "predicted_return": 0.01, "actual_return": 0.02}
    assert evaluator._calc_trade_pnl(row) == 0.02 - 0.0016


def test_merge_count_dict_recurses_nested():
    target = {}
    evaluator._merge_count_dict(target, {"Long": {"passed": 1, "pi_guard_only": 2}, "Short": {"passed": 3}})
    evaluator._merge_count_dict(target, {"Long": {"passed": 4, "pi_guard_only": 1}, "Short": {"passed": 2}})
    assert target == {"Long": {"passed": 5, "pi_guard_only": 3}, "Short": {"passed": 5}}
