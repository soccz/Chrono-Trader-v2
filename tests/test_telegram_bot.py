import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main
from utils import telegram_bot
from utils.config import config


def test_format_short_term_report_focuses_on_entry_and_distribution():
    msg = telegram_bot.format_short_term_report(
        scalp_recs=[
            {
                "market": "KRW-XRP",
                "signal": "Long",
                "strategy": "intraday",
                "status": "Recommended",
                "current_price": 100.0,
                "expected_return": 0.05,
                "pi_low_80": -0.01,
                "pi_high_80": 0.08,
            },
            {
                "market": "KRW-BTC",
                "signal": "Long",
                "strategy": "intraday",
                "status": "Watch (MinRec)",
                "current_price": 100.0,
                "expected_return": 0.02,
                "pi_low_80": -0.02,
                "pi_high_80": 0.03,
            },
        ],
        pump_recs=[],
        meta={},
    )

    assert "KRW-XRP" in msg
    assert "target 105" in msg
    assert "range 99 ~ 108" in msg
    assert "KRW-BTC" not in msg
    assert "WATCH ONLY" not in msg


def test_send_intraday_report_skips_when_no_actionable(monkeypatch):
    sent = []

    monkeypatch.setattr("utils.telegram_bot.send_alert", lambda message, **kwargs: sent.append(message))

    main.send_intraday_report(
        [
            {"market": "KRW-BTC", "status": "Watch (MinRec)", "current_price": 100.0},
        ],
        run_meta={},
    )

    assert sent == []


def test_send_morning_report_sends_only_actionable_sections(monkeypatch, tmp_path):
    sent = []
    state_path = tmp_path / "telegram_signal_state.json"
    original_should_send = telegram_bot.should_send_signal_alert

    monkeypatch.setattr("utils.telegram_bot.send_alert", lambda message, **kwargs: sent.append(message))
    monkeypatch.setattr(
        "utils.telegram_bot.should_send_signal_alert",
        lambda channel, items, path=telegram_bot.SIGNAL_STATE_FILE: original_should_send(channel, items, path=str(state_path)),
    )

    main.send_morning_report(
        trending_recs=[
            {
                "market": "KRW-ETH",
                "signal": "Long",
                "strategy": "trending",
                "status": "Recommended",
                "current_price": 200.0,
                "expected_return": 0.03,
                "pi_low_80": -0.01,
                "pi_high_80": 0.05,
            }
        ],
        pattern_recs=[
            {"market": "KRW-XRP", "status": "Watch (MinRec)", "current_price": 100.0}
        ],
        pump_recs=[],
        run_meta={},
    )

    assert len(sent) == 1
    assert "KRW-ETH" in sent[0]
    assert "KRW-XRP" not in sent[0]


def test_should_send_signal_alert_dedups_same_signal_set(tmp_path):
    path = tmp_path / "telegram_signal_state.json"
    items = [
        {
            "market": "KRW-XRP",
            "signal": "Long",
            "strategy": "intraday",
            "current_price": 100.0,
            "expected_return": 0.05,
            "pi_low_80": -0.01,
            "pi_high_80": 0.08,
        }
    ]

    assert telegram_bot.should_send_signal_alert("intraday", items, path=str(path)) is True
    assert telegram_bot.should_send_signal_alert("intraday", items, path=str(path)) is False


def test_send_intraday_report_dedups_unchanged_actionable_set(monkeypatch, tmp_path):
    sent = []
    state_path = tmp_path / "telegram_signal_state.json"
    original_should_send = telegram_bot.should_send_signal_alert

    monkeypatch.setattr("utils.telegram_bot.send_alert", lambda message, **kwargs: sent.append(message))
    monkeypatch.setattr(
        "utils.telegram_bot.should_send_signal_alert",
        lambda channel, items, path=telegram_bot.SIGNAL_STATE_FILE: original_should_send(channel, items, path=str(state_path)),
    )

    payload = [
        {
            "market": "KRW-XRP",
            "signal": "Long",
            "strategy": "intraday",
            "status": "Recommended",
            "current_price": 100.0,
            "expected_return": 0.05,
            "pi_low_80": -0.01,
            "pi_high_80": 0.08,
        }
    ]

    main.send_intraday_report(payload, run_meta={})
    main.send_intraday_report(payload, run_meta={})

    assert len(sent) == 1


def test_should_send_signal_alert_respects_intraday_daily_cap(monkeypatch, tmp_path):
    path = tmp_path / "telegram_signal_state.json"
    monkeypatch.setattr(config.Recommender, "TELEGRAM_ALERT_CAP_INTRADAY_PER_DAY", 2)
    monkeypatch.setattr("utils.telegram_bot._current_report_date_str", lambda: "2026-03-21")

    first = [
        {
            "market": "KRW-XRP",
            "signal": "Long",
            "strategy": "intraday",
            "current_price": 100.0,
            "expected_return": 0.05,
            "pi_low_80": -0.01,
            "pi_high_80": 0.08,
        }
    ]
    second = [
        {
            "market": "KRW-ETH",
            "signal": "Long",
            "strategy": "intraday",
            "current_price": 200.0,
            "expected_return": 0.04,
            "pi_low_80": -0.01,
            "pi_high_80": 0.06,
        }
    ]
    third = [
        {
            "market": "KRW-BTC",
            "signal": "Long",
            "strategy": "intraday",
            "current_price": 300.0,
            "expected_return": 0.03,
            "pi_low_80": -0.01,
            "pi_high_80": 0.05,
        }
    ]

    assert telegram_bot.should_send_signal_alert("intraday", first, path=str(path)) is True
    assert telegram_bot.should_send_signal_alert("intraday", second, path=str(path)) is True
    assert telegram_bot.should_send_signal_alert("intraday", third, path=str(path)) is False


def test_should_send_signal_alert_resets_daily_cap_on_next_day(monkeypatch, tmp_path):
    path = tmp_path / "telegram_signal_state.json"
    monkeypatch.setattr(config.Recommender, "TELEGRAM_ALERT_CAP_INTRADAY_PER_DAY", 1)

    item_day1 = [
        {
            "market": "KRW-XRP",
            "signal": "Long",
            "strategy": "intraday",
            "current_price": 100.0,
            "expected_return": 0.05,
            "pi_low_80": -0.01,
            "pi_high_80": 0.08,
        }
    ]
    item_day2 = [
        {
            "market": "KRW-ETH",
            "signal": "Long",
            "strategy": "intraday",
            "current_price": 200.0,
            "expected_return": 0.04,
            "pi_low_80": -0.01,
            "pi_high_80": 0.06,
        }
    ]

    monkeypatch.setattr("utils.telegram_bot._current_report_date_str", lambda: "2026-03-21")
    assert telegram_bot.should_send_signal_alert("intraday", item_day1, path=str(path)) is True

    monkeypatch.setattr("utils.telegram_bot._current_report_date_str", lambda: "2026-03-22")
    assert telegram_bot.should_send_signal_alert("intraday", item_day2, path=str(path)) is True


def test_format_short_term_report_keeps_decimal_precision_for_low_price_coin():
    msg = telegram_bot.format_short_term_report(
        scalp_recs=[
            {
                "market": "KRW-WAXP",
                "signal": "Long",
                "strategy": "intraday",
                "status": "Recommended",
                "current_price": 11.2,
                "expected_return": 0.004252366293079124,
                "pi_low_80": -0.008271433436311782,
                "pi_high_80": 0.018974180705845358,
            },
        ],
        pump_recs=[],
        meta={},
    )

    assert "entry 11.2" in msg
    assert "target 11.25" in msg
    assert "range 11.11 ~ 11.41" in msg
