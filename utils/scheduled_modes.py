import time

from data import database
from utils import screener


def run_refresh_db_mode(
    args,
    *,
    config,
    logger,
    collect_refresh_batch,
    summarize_refresh_results,
    alert_index_refresh_failures,
    persist_run_outputs,
    send_refresh_done_alert,
):
    from utils.run_lock import run_lock
    from utils.netcheck import resolution_status
    from utils.market_selection import select_markets_for_scheduled_run
    from utils.run_markets_cache import rotate_markets
    from utils.watchdog import time_limit

    logger.info("Starting DB refresh job...")
    send_tg = (not args.no_telegram)
    if args.send_telegram:
        send_tg = True

    with run_lock("refresh-db"):
        timeout = int(args.timeout_sec)
        if timeout == 0:
            timeout = 20 * 60
        if timeout < 0:
            timeout = 0

        with time_limit(timeout, name="refresh-db"):
            t0 = time.time()
            dns_attempts = int(getattr(config.Recommender, "DNS_RESOLVE_RETRIES", 3) or 3)
            dns_delay_sec = float(getattr(config.Recommender, "DNS_RESOLVE_RETRY_DELAY_SEC", 0.25) or 0.25)
            dns_status = resolution_status("api.upbit.com", attempts=dns_attempts, delay_sec=dns_delay_sec)
            offline = not bool(dns_status.get("ok"))
            if offline:
                run_meta = {
                    "mode": "refresh-db",
                    "budget": int(args.market_budget or 0),
                    "seed_n": 0,
                    "network_status": "offline",
                    "auto_refresh_enabled": False,
                    "auto_refresh_skipped_offline": True,
                    "auto_refresh_dns_attempts": int(dns_status.get("attempts") or dns_attempts),
                    "auto_refresh_dns_error": dns_status.get("error"),
                    "auto_refresh_dns_ips": list(dns_status.get("ips") or []),
                    "refresh_attempted": 0,
                    "refresh_ok": 0,
                    "refresh_up_to_date": 0,
                    "refresh_no_data": 0,
                    "refresh_failed": 0,
                    "refresh_saved_records": 0,
                    "refresh_index_failed": [],
                }
                logger.warning(
                    "Upbit host resolution failed; refresh-db cannot proceed "
                    f"(attempts={int(dns_status.get('attempts') or dns_attempts)}, error={dns_status.get('error') or 'unknown'})."
                )
                try:
                    run_meta["elapsed_sec"] = float(time.time() - t0)
                    _, manifest_path = persist_run_outputs("refresh-db", [], run_meta, recs=[])
                    run_meta["output_manifest_path"] = manifest_path
                except Exception as e:
                    logger.debug(f"[Metrics] refresh-db offline record_run failed: {e}")
                if send_tg:
                    try:
                        from utils.telegram_bot import send_alert
                        send_alert(
                            "<b>⚠️ REFRESH-DB OFFLINE</b>\n\n"
                            f"DNS resolution failed for api.upbit.com\n"
                            f"<code>attempts={int(dns_status.get('attempts') or dns_attempts)}</code>\n"
                            f"<code>error={dns_status.get('error') or 'unknown'}</code>",
                            bypass_dedup=True,
                        )
                    except Exception as e:
                        logger.error(f"Failed to send refresh-db offline alert: {e}")
                if args.offline_ok:
                    logger.info("offline_ok enabled; skipping refresh-db and exiting 0.")
                    raise SystemExit(0)
                raise SystemExit(2)

            seed_markets = screener.get_trending_markets(mode='live', limit=args.limit, lookback_days=max(args.lookback_days, 1))
            budget = int(args.market_budget or 0)
            if budget <= 0:
                budget = 32 if str(config.Device.DEVICE).startswith("cuda") else 24

            run_meta = {
                "mode": "refresh-db",
                "budget": budget,
                "seed_n": len(seed_markets or []),
            }

            refresh_markets, sel_meta = select_markets_for_scheduled_run(
                mode="intraday",
                seed_markets=seed_markets,
                budget=budget,
                tv_hours=int(args.refresh_tv_hours),
                candidate_top=max(200, budget * 5),
                lookback_hours=168,
                exploit_target=int(args.refresh_top_n),
                max_holdings=5,
                max_core=10,
                max_lag_h=None,
                return_meta=True,
            )
            run_meta.update(sel_meta or {})

            try:
                refresh_markets, rot_meta = rotate_markets(
                    mode="refresh-db",
                    new_markets=refresh_markets,
                    budget=budget,
                    keep_ratio=float(args.rotation_keep),
                    max_lag_h=None,
                )
                logger.info(f"[RefreshDB] rotation kept={rot_meta.get('kept')} added={rot_meta.get('added')} budget={budget}")
                run_meta["rotation_kept"] = rot_meta.get("kept")
                run_meta["rotation_added"] = rot_meta.get("added")
            except Exception as e:
                logger.warning(f"[RefreshDB] rotation failed (proceeding): {e}")

            logger.info(f"Refreshing latest data for {len(refresh_markets)} markets (days={args.refresh_days})...")
            refresh_results = collect_refresh_batch(refresh_markets, days=args.refresh_days, sleep_sec=0.15)
            refresh_summary = summarize_refresh_results(refresh_results)
            run_meta["refresh_attempted"] = refresh_summary["attempted"]
            run_meta["refresh_ok"] = refresh_summary["ok"]
            run_meta["refresh_up_to_date"] = refresh_summary["up_to_date"]
            run_meta["refresh_no_data"] = refresh_summary["no_data"]
            run_meta["refresh_failed"] = refresh_summary["failed"]
            run_meta["refresh_saved_records"] = refresh_summary["saved_records"]
            run_meta["refresh_index_failed"] = list(refresh_summary["index_failed"])
            alert_index_refresh_failures(send_tg, "refresh-db", refresh_summary)

            try:
                run_meta["elapsed_sec"] = float(time.time() - t0)
                _, manifest_path = persist_run_outputs("refresh-db", refresh_markets, run_meta, recs=[])
                run_meta["output_manifest_path"] = manifest_path
            except Exception as e:
                logger.debug(f"[Metrics] refresh-db record_run failed: {e}")

    logger.info("DB refresh job finished.")


def run_intraday_mode(
    args,
    *,
    config,
    logger,
    maybe_auto_refresh_markets,
    enforce_freshness_gate,
    run_recommender_for_markets,
    persist_run_outputs,
    send_intraday_report,
    send_timeout_alert,
):
    from utils.run_lock import run_lock
    from utils.watchdog import time_limit
    from utils.market_selection import select_markets_for_scheduled_run

    config.General.REC_TAG = "intraday"
    logger.info("Starting intraday inference run (no training)...")

    send_tg = (not args.no_telegram)
    if args.send_telegram:
        send_tg = True

    with run_lock("intraday"):
        timeout = int(args.timeout_sec)
        if timeout == 0:
            timeout = 20 * 60
        if timeout < 0:
            timeout = 0

        def _intraday_body():
            t0 = time.time()
            seed_markets = screener.get_trending_markets(mode='live', limit=args.limit, lookback_days=args.lookback_days)
            seed_fallback_index = False
            if not seed_markets:
                seed_markets = list(getattr(config.Data, "MARKET_INDEX_COINS", [])) or ["KRW-BTC", "KRW-ETH"]
                seed_fallback_index = True
                logger.warning(
                    f"No markets found for intraday screening; falling back to index coins: {seed_markets}"
                )

            budget = int(args.market_budget or 0)
            if budget <= 0:
                budget = 32 if str(config.Device.DEVICE).startswith("cuda") else 24

            max_lag_sel = float(getattr(config.Recommender, "DATA_FRESHNESS_MAX_LAG_HOURS_INTRADAY", 6))
            run_meta = {
                "mode": "intraday",
                "budget": budget,
                "seed_n": len(seed_markets or []),
                "seed_fallback_index": bool(seed_fallback_index),
            }

            run_markets, sel_meta = select_markets_for_scheduled_run(
                mode="intraday",
                seed_markets=seed_markets,
                budget=budget,
                tv_hours=int(args.refresh_tv_hours),
                candidate_top=max(200, budget * 5),
                lookback_hours=168,
                exploit_target=int(args.refresh_top_n),
                max_holdings=5,
                max_core=10,
                max_lag_h=max_lag_sel,
                return_meta=True,
            )
            run_meta.update(sel_meta or {})

            try:
                from utils.run_markets_cache import rotate_markets
                rot_meta = {}
                run_markets, rot_meta = rotate_markets(
                    mode="intraday",
                    new_markets=run_markets,
                    budget=budget,
                    keep_ratio=float(args.rotation_keep),
                    max_lag_h=max_lag_sel,
                )
                logger.info(f"[RunMarkets] rotation kept={rot_meta.get('kept')} added={rot_meta.get('added')} budget={budget}")
                run_meta["rotation_kept"] = rot_meta.get("kept")
                run_meta["rotation_added"] = rot_meta.get("added")
            except Exception as e:
                logger.warning(f"[RunMarkets] rotation failed (proceeding): {e}")

            auto_refresh_enabled = (not args.no_auto_refresh) or args.refresh_data
            if auto_refresh_enabled:
                try:
                    maybe_auto_refresh_markets(
                        run_markets,
                        refresh_days=args.refresh_days,
                        run_meta=run_meta,
                        send_tg=send_tg,
                        context_label="intraday-refresh",
                    )
                except Exception as e:
                    logger.warning(f"Auto-refresh failed (proceeding): {e}")
            else:
                run_meta["auto_refresh_enabled"] = False

            try:
                max_lag_h = float(getattr(config.Recommender, "DATA_FRESHNESS_MAX_LAG_HOURS_INTRADAY", 6))
                run_markets = enforce_freshness_gate(
                    mode_label="intraday",
                    run_markets=run_markets,
                    run_meta=run_meta,
                    send_tg=send_tg,
                    max_lag_h=max_lag_h,
                    allow_stale_data=bool(args.allow_stale_data),
                )
            except SystemExit:
                raise
            except Exception as e:
                logger.error(f"[Freshness] Freshness gate check failed: {e}")
                raise SystemExit(2)  # Treat as stale abort -- safe fallback will handle

            logger.info(
                f"[OpsTrace] intraday recommender dispatch run_markets_n={len(run_markets or [])} "
                f"markets={run_markets}"
            )
            intraday_recs = run_recommender_for_markets(
                run_markets,
                strategy_name="intraday",
                min_k=args.min_k,
            )
            logger.info(
                f"[OpsTrace] intraday recommender returned recs_n={len(intraday_recs or [])}"
            )

            try:
                run_meta["elapsed_sec"] = float(time.time() - t0)
                logger.info(
                    f"[OpsTrace] intraday persist_run_outputs start "
                    f"elapsed_sec={run_meta['elapsed_sec']:.2f}"
                )
                _, manifest_path = persist_run_outputs("intraday", run_markets, run_meta, recs=intraday_recs)
                run_meta["output_manifest_path"] = manifest_path
                logger.info(
                    f"[OpsTrace] intraday persist_run_outputs finished manifest={manifest_path}"
                )
            except Exception as e:
                logger.debug(f"[Metrics] intraday record_run failed: {e}")

            if send_tg:
                try:
                    send_intraday_report(intraday_recs, run_meta=run_meta)
                except Exception as e:
                    logger.error(f"Failed to send intraday telegram report: {e}")

        try:
            with time_limit(timeout, name="intraday"):
                _intraday_body()
        except TimeoutError as e:
            logger.error(f"[Watchdog] Intraday timeout: {e}")
            if send_tg:
                try:
                    send_timeout_alert("intraday", str(e))
                except Exception as ex:
                    logger.error(f"Failed to send intraday timeout telegram: {ex}")
            raise SystemExit(3)

    logger.info("Intraday inference run finished.")


def run_morning_report_mode(
    args,
    *,
    config,
    logger,
    maybe_auto_refresh_markets,
    enforce_freshness_gate,
    run_recommender_for_markets,
    run_pattern_followers_section,
    run_pump_radar_section,
    persist_run_outputs,
    send_morning_report,
    send_timeout_alert,
):
    from utils.run_lock import run_lock
    from utils.watchdog import time_limit
    from utils.market_selection import select_markets_for_scheduled_run

    config.General.REC_TAG = "morning"
    logger.info("Starting morning report run (no training)...")

    send_tg = (not args.no_telegram)
    if args.send_telegram:
        send_tg = True

    timeout = int(args.timeout_sec)
    if timeout == 0:
        timeout = 45 * 60
    if timeout < 0:
        timeout = 0

    def _morning_body():
        t0 = time.time()
        seed_markets = screener.get_trending_markets(mode='live', limit=args.limit, lookback_days=max(args.lookback_days, 1))
        seed_fallback_index = False
        if not seed_markets:
            seed_markets = list(getattr(config.Data, "MARKET_INDEX_COINS", [])) or ["KRW-BTC", "KRW-ETH"]
            seed_fallback_index = True
            logger.warning(
                f"No trending markets found for morning report; falling back to index coins: {seed_markets}"
            )

        budget = int(args.market_budget or 0)
        if budget <= 0:
            budget = 64 if str(config.Device.DEVICE).startswith("cuda") else 40

        max_lag_sel = float(getattr(config.Recommender, "DATA_FRESHNESS_MAX_LAG_HOURS_MORNING", 12))
        run_meta = {
            "mode": "morning",
            "budget": budget,
            "seed_n": len(seed_markets or []),
            "seed_fallback_index": bool(seed_fallback_index),
        }
        run_markets, sel_meta = select_markets_for_scheduled_run(
            mode="morning",
            seed_markets=seed_markets,
            budget=budget,
            tv_hours=int(args.refresh_tv_hours),
            candidate_top=max(200, budget * 5),
            lookback_hours=168,
            exploit_target=int(args.refresh_top_n),
            max_holdings=5,
            max_core=10,
            max_lag_h=max_lag_sel,
            return_meta=True,
        )
        run_meta.update(sel_meta or {})

        try:
            from utils.run_markets_cache import rotate_markets
            rot_meta = {}
            run_markets, rot_meta = rotate_markets(
                mode="morning",
                new_markets=run_markets,
                budget=budget,
                keep_ratio=float(args.rotation_keep),
                max_lag_h=max_lag_sel,
            )
            logger.info(f"[RunMarkets] rotation kept={rot_meta.get('kept')} added={rot_meta.get('added')} budget={budget}")
            run_meta["rotation_kept"] = rot_meta.get("kept")
            run_meta["rotation_added"] = rot_meta.get("added")
        except Exception as e:
            logger.warning(f"[RunMarkets] rotation failed (proceeding): {e}")

        auto_refresh_enabled = (not args.no_auto_refresh) or args.refresh_data
        if auto_refresh_enabled:
            try:
                maybe_auto_refresh_markets(
                    run_markets,
                    refresh_days=args.refresh_days,
                    run_meta=run_meta,
                    send_tg=send_tg,
                    context_label="morning-refresh",
                )
            except Exception as e:
                logger.warning(f"Auto-refresh failed (proceeding): {e}")
        else:
            run_meta["auto_refresh_enabled"] = False

        try:
            max_lag_h = float(getattr(config.Recommender, "DATA_FRESHNESS_MAX_LAG_HOURS_MORNING", 12))
            run_markets = enforce_freshness_gate(
                mode_label="morning",
                run_markets=run_markets,
                run_meta=run_meta,
                send_tg=send_tg,
                max_lag_h=max_lag_h,
                allow_stale_data=bool(args.allow_stale_data),
            )
        except SystemExit:
            raise
        except Exception as e:
            logger.warning(f"[Freshness] Freshness gate check failed (proceeding): {e}")

        trending_recs = run_recommender_for_markets(
            run_markets,
            strategy_name="trending",
            min_k=args.min_k,
        )

        pattern_recs = []
        skip_pattern_followers = bool(args.skip_aux or args.skip_pattern_followers)
        if skip_pattern_followers:
            logger.info("[Aux] Skipping pattern followers section.")
        else:
            try:
                pattern_recs = run_pattern_followers_section(
                    seed_markets=seed_markets,
                    run_markets=run_markets,
                    min_k=args.min_k,
                )
            except Exception as e:
                logger.error(f"Pattern follower section failed: {e}")

        pump_recs = []
        skip_pump = bool(args.skip_aux or args.skip_pump_radar)
        if skip_pump:
            logger.info("[Aux] Skipping pump radar section.")
        else:
            try:
                pump_recs = run_pump_radar_section()
            except Exception as e:
                logger.error(f"Pump radar failed: {e}")

        try:
            run_meta["elapsed_sec"] = float(time.time() - t0)
            _, manifest_path = persist_run_outputs(
                "morning",
                run_markets,
                run_meta,
                recs=(trending_recs or []) + (pattern_recs or []),
                include_pump_preds=bool(pump_recs),
            )
            run_meta["output_manifest_path"] = manifest_path
        except Exception as e:
            logger.debug(f"[Metrics] morning record_run failed: {e}")

        if send_tg:
            try:
                send_morning_report(
                    trending_recs=trending_recs,
                    pattern_recs=pattern_recs,
                    pump_recs=pump_recs,
                    run_meta=run_meta,
                )
            except Exception as e:
                logger.error(f"Failed to send morning telegram report: {e}")

    with run_lock("morning-report"):
        try:
            with time_limit(timeout, name="morning-report"):
                _morning_body()
        except TimeoutError as e:
            logger.error(f"[Watchdog] Morning-report timeout: {e}")
            if send_tg:
                try:
                    send_timeout_alert("morning", str(e))
                except Exception as ex:
                    logger.error(f"Failed to send morning timeout telegram: {ex}")
            raise SystemExit(3)

    logger.info("Morning report run finished.")
