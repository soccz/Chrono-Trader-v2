# systemd (User) Deployment

These unit files schedule AETHER runs on this host (timezone: KST).

Units:
- `aether-intraday.timer`: every 4 hours
- `aether-morning.timer`: daily 08:00
- `aether-autotune.timer`: daily 08:15 (optional)

The services run `scripts/run_scheduled.py` which:
- runs `refresh-db`
- runs inference with freshness gate
- if freshness aborts (exit=2) reruns once with `--allow_stale_data` + watch-only
- guarantees >=1 output item per run (MinRec + synthetic watch-only final fallback)

