from datetime import datetime, timezone
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_mode_health_reports_ok_for_fresh_metrics(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    metrics_path = analysis_dir / "run_markets_metrics_intraday.json"
    metrics_path.write_text(
        """
        {
          "ts": "2026-03-19T10:00:00+00:00",
          "recs": {"n": 3, "has_watch": false},
          "meta": {"run_markets_kept": 8, "freshness_dropped": 2, "freshness_used_fallback": true}
        }
        """.strip(),
        encoding="utf-8",
    )

    from utils.ops_health import mode_health

    result = mode_health(
        mode="intraday",
        max_age_h=5.0,
        now_utc=datetime(2026, 3, 19, 12, 0, tzinfo=timezone.utc),
    )

    assert result["ok"] is True
    assert result["status"] == "ok"
    assert result["recs_n"] == 3
    assert result["kept"] == 8
    assert result["dropped"] == 2
    assert result["used_fallback"] is True


def test_mode_health_surfaces_offline_auto_refresh_context(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    metrics_path = analysis_dir / "run_markets_metrics_intraday.json"
    metrics_path.write_text(
        """
        {
          "ts": "2026-03-19T10:00:00+00:00",
          "recs": {"n": 1, "has_watch": false},
          "meta": {
            "run_markets_kept": 8,
            "freshness_dropped": 0,
            "freshness_used_fallback": false,
            "auto_refresh_skipped_offline": true,
            "auto_refresh_dns_error": "name or service not known"
          }
        }
        """.strip(),
        encoding="utf-8",
    )

    from utils.ops_health import mode_health

    result = mode_health(
        mode="intraday",
        max_age_h=5.0,
        now_utc=datetime(2026, 3, 19, 12, 0, tzinfo=timezone.utc),
    )

    assert result["ok"] is True
    assert result["auto_refresh_skipped_offline"] is True
    assert result["auto_refresh_dns_error"] == "name or service not known"
    assert "offline_auto_refresh" in result["message"]


def test_mode_health_reports_stale_for_old_metrics(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    metrics_path = analysis_dir / "run_markets_metrics_morning.json"
    metrics_path.write_text(
        """
        {
          "ts": "2026-03-18T00:00:00+00:00",
          "recs": {"n": 5, "has_watch": true},
          "meta": {"run_markets_kept": 5, "freshness_dropped": 0, "freshness_used_fallback": false}
        }
        """.strip(),
        encoding="utf-8",
    )

    from utils.ops_health import mode_health

    result = mode_health(
        mode="morning",
        max_age_h=6.0,
        now_utc=datetime(2026, 3, 19, 12, 0, tzinfo=timezone.utc),
    )

    assert result["ok"] is False
    assert result["status"] == "stale"
    assert result["recs_n"] == 5


def test_mode_health_reports_empty_recs_when_recent_but_empty(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    metrics_path = analysis_dir / "run_markets_metrics_intraday.json"
    metrics_path.write_text(
        """
        {
          "ts": "2026-03-19T11:00:00+00:00",
          "recs": {"n": 0, "has_watch": false},
          "meta": {"run_markets_kept": 0, "freshness_dropped": 10, "freshness_used_fallback": false}
        }
        """.strip(),
        encoding="utf-8",
    )

    from utils.ops_health import mode_health

    result = mode_health(
        mode="intraday",
        max_age_h=5.0,
        now_utc=datetime(2026, 3, 19, 12, 0, tzinfo=timezone.utc),
    )

    assert result["ok"] is False
    assert result["status"] == "empty_recs"
    assert result["dropped"] == 10


def test_mode_health_reports_refresh_db_offline(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    metrics_path = analysis_dir / "run_markets_metrics_refresh-db.json"
    metrics_path.write_text(
        """
        {
          "ts": "2026-03-19T11:00:00+00:00",
          "recs": {"n": 0, "has_watch": false},
          "meta": {
            "network_status": "offline",
            "auto_refresh_skipped_offline": true,
            "auto_refresh_dns_error": "name or service not known"
          }
        }
        """.strip(),
        encoding="utf-8",
    )

    from utils.ops_health import mode_health

    result = mode_health(
        mode="refresh-db",
        max_age_h=30.0,
        now_utc=datetime(2026, 3, 19, 12, 0, tzinfo=timezone.utc),
    )

    assert result["ok"] is False
    assert result["status"] == "offline"
    assert result["auto_refresh_skipped_offline"] is True
    assert "offline" in result["message"]
