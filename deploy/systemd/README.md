# systemd (User) Deployment

These unit files schedule AETHER runs on this host (timezone: KST).

Units:
- `aether-intraday.timer`: every 4 hours
- `aether-morning.timer`: daily 08:00
- `aether-autotune.timer`: daily 08:15 (optional)
- `aether-healthcheck.timer`: every 15 minutes (alerts if outputs are missing/stale)

The services run `scripts/run_ops_job.py`, which resolves the repo runtime in this order:
- `AETHER_PYTHON_BIN`
- `.venv_local/bin/python`
- `python3`

It then launches `scripts/run_scheduled.py`, which:
- runs `refresh-db`
- runs inference with freshness gate
- if freshness aborts (exit=2) reruns once with `--allow_stale_data` + watch-only
- guarantees >=1 output item per run (MinRec + synthetic watch-only final fallback)

Notes:
- The `.service` files in this folder embed an absolute repo path (host-specific). If you deploy elsewhere, update `WorkingDirectory=...`, `PYTHONPATH=...`, and `ExecStart=...` paths accordingly.
- If you want to pin a different interpreter, add `Environment=AETHER_PYTHON_BIN=/abs/path/to/python` to the service override.
