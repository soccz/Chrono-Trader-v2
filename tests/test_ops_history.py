import json
import importlib.util
from pathlib import Path


def _load_ops_history_module():
    module_path = Path(__file__).resolve().parents[1] / "utils" / "ops_history.py"
    spec = importlib.util.spec_from_file_location("test_ops_history_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_read_recent_ops_runs_merges_and_sorts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()

    (analysis_dir / "run_markets_metrics_intraday.jsonl").write_text(
        "\n".join(
            [
                json.dumps({
                    "mode": "intraday",
                    "ts": "2026-03-19T09:00:00+00:00",
                    "markets": ["KRW-BTC", "KRW-ETH"],
                    "meta": {"elapsed_sec": 11.2, "run_markets_kept": 2, "freshness_dropped": 0},
                    "recs": {"n": 1, "has_watch": False, "has_forced": False},
                }),
                json.dumps({
                    "mode": "intraday",
                    "ts": "2026-03-19T11:00:00+00:00",
                    "markets": ["KRW-BTC"],
                    "meta": {"elapsed_sec": 7.5, "run_markets_kept": 1, "freshness_dropped": 1},
                    "recs": {"n": 1, "has_watch": True, "has_forced": False},
                }),
            ]
        ),
        encoding="utf-8",
    )
    (analysis_dir / "run_markets_metrics_morning.jsonl").write_text(
        json.dumps({
            "mode": "morning",
            "ts": "2026-03-19T10:00:00+00:00",
            "markets": ["KRW-BTC", "KRW-XRP", "KRW-SOL"],
            "meta": {"elapsed_sec": 20.0, "run_markets_kept": 3, "freshness_dropped": 2},
            "recs": {"n": 2, "has_watch": False, "has_forced": True},
        }) + "\n",
        encoding="utf-8",
    )

    ops_history = _load_ops_history_module()
    rows = ops_history.read_recent_ops_runs(limit_per_mode=2)
    assert [row["mode"] for row in rows] == ["intraday", "morning", "intraday"]
    assert rows[0]["has_watch"] is True
    assert rows[1]["has_forced"] is True
    assert rows[2]["markets_n"] == 2
