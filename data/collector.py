import requests
import pandas as pd
from datetime import datetime, timedelta
import time
from typing import Optional

from utils.config import config
from utils.logger import logger
from data.database import save_data, get_db_connection, init_db


def _request_upbit_json(url: str, params: dict, timeout: int = 10):
    retries = int(getattr(config.Data, "COLLECTOR_REQUEST_MAX_RETRIES", 3) or 3)
    backoff_sec = float(getattr(config.Data, "COLLECTOR_REQUEST_BACKOFF_SEC", 1.0) or 1.0)
    last_err = None

    for attempt in range(1, retries + 1):
        for fmt in ("T", " "):
            req_params = dict(params)
            if "to" in req_params and isinstance(req_params["to"], str):
                req_params["to"] = req_params["to"].replace("T", fmt)
            try:
                res = requests.get(url, params=req_params, timeout=timeout)
                res.raise_for_status()
                data = res.json()
                if isinstance(data, dict) and "error" in data:
                    last_err = RuntimeError(str(data.get("error")))
                    continue
                return data, attempt, None
            except requests.exceptions.RequestException as e:
                last_err = e
                continue

        if attempt < retries:
            sleep_for = backoff_sec * attempt
            logger.warning(
                f"Upbit request retry {attempt}/{retries} failed for {params.get('market')}; sleeping {sleep_for:.1f}s"
            )
            time.sleep(sleep_for)

    return None, retries, last_err

def get_oldest_timestamp(market: str):
    """Get the oldest timestamp for a given market from the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT MIN(timestamp) FROM crypto_data WHERE market = ?", (market,))
    result = cursor.fetchone()
    conn.close()
    if result and result[0]:
        ts_str = result[0]
        if 'T' in ts_str:
            return datetime.strptime(ts_str, '%Y-%m-%dT%H:%M:%S')
        else:
            return datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')
    return None

def backfill_market_data(market: str, days: int = 2600):
    """Backfills historical data going backwards from the oldest existing record."""
    logger.info(f"Starting backfill for {market} ({days} days back)")
    url = "https://api.upbit.com/v1/candles/minutes/60"
    oldest_ts = get_oldest_timestamp(market)
    target_date = datetime.utcnow() - timedelta(days=days)

    if oldest_ts is None:
        to_datetime = (datetime.utcnow() + timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%S')
    else:
        if oldest_ts <= target_date:
            logger.info(f"{market} already has data back to {oldest_ts}, target={target_date}. Skipping.")
            return
        to_datetime = oldest_ts.strftime('%Y-%m-%dT%H:%M:%S')

    all_data = []
    while True:
        params = {'market': market, 'count': 200, 'to': to_datetime}
        try:
            data = None
            for _fmt in ("T", " "):
                params['to'] = to_datetime.replace("T", _fmt)
                try:
                    res = requests.get(url, params=params, timeout=10)
                    res.raise_for_status()
                    data = res.json()
                    if isinstance(data, dict) and "error" in data:
                        data = None
                        continue
                    break
                except requests.exceptions.RequestException:
                    data = None
                    continue
            if not data:
                break

            all_data.extend(data)
            oldest_ts_str = data[-1]['candle_date_time_utc']
            oldest_ts = datetime.strptime(oldest_ts_str, '%Y-%m-%dT%H:%M:%S')
            logger.info(f"Backfill {market}: fetched to {oldest_ts}")

            if oldest_ts <= target_date:
                logger.info(f"Reached target date {target_date}. Stopping.")
                break

            to_datetime = oldest_ts.strftime('%Y-%m-%dT%H:%M:%S')
            time.sleep(0.5)

            if len(all_data) >= 5000:
                _save_candles(all_data, market)
                all_data = []

        except requests.exceptions.RequestException as e:
            logger.error(f"Backfill request failed for {market}: {e}")
            break

    if all_data:
        _save_candles(all_data, market)
    logger.info(f"Backfill complete for {market}")

def _save_candles(data: list, market: str):
    """Save raw candle list to DB."""
    df = pd.DataFrame(data)
    df.rename(columns={
        'candle_date_time_utc': 'timestamp',
        'opening_price': 'open',
        'high_price': 'high',
        'low_price': 'low',
        'trade_price': 'close',
        'candle_acc_trade_volume': 'volume'
    }, inplace=True)
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    df['market'] = market
    df = df.drop_duplicates(subset=['timestamp', 'market'])
    save_data(df, 'crypto_data')

def get_last_timestamp(market: str):
    """Get the last timestamp for a given market from the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(timestamp) FROM crypto_data WHERE market = ?", (market,))
    result = cursor.fetchone()
    conn.close()
    if result and result[0]:
        ts_str = result[0]
        if 'T' in ts_str:
            return datetime.strptime(ts_str, '%Y-%m-%dT%H:%M:%S')
        else:
            return datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')
    return None

def collect_market_data(market: str, days: int = 90):
    """Collects historical data for a single market from Upbit."""
    logger.info(f"Starting data collection for market: {market}")
    url = "https://api.upbit.com/v1/candles/minutes/60"
    last_ts = get_last_timestamp(market)
    started_at = time.time()
    summary = {
        "market": market,
        "days": int(days),
        "had_existing_data": bool(last_ts is not None),
        "status": "started",
        "pages": 0,
        "saved_records": 0,
        "fetched_records_raw": 0,
        "request_attempts": 0,
        "error": None,
    }

    if last_ts:
        # Upbit "to" supports ISO8601; keep consistent with DB timestamps.
        to_datetime = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')
        logger.info(f"Last timestamp for {market} is {last_ts}. Fetching new data up to {to_datetime}.")
    else:
        to_datetime = (datetime.utcnow() + timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%S')
        logger.info(f"No existing data for {market}. Fetching last {days} days of data.")

    all_data = []
    while True:
        params = {
            'market': market,
            'count': 200,
            'to': to_datetime
        }
        try:
            data, attempts_used, last_err = _request_upbit_json(url, params=params, timeout=10)
            summary["request_attempts"] += int(attempts_used or 0)
            if data is None:
                raise requests.exceptions.RequestException(last_err)

            if not data:
                logger.info(f"No more data to fetch for {market}.")
                break

            summary["pages"] += 1
            summary["fetched_records_raw"] += len(data)

            if last_ts:
                new_data = []
                stop_collecting = False
                for candle in data:
                    candle_ts = datetime.strptime(candle['candle_date_time_utc'], '%Y-%m-%dT%H:%M:%S')
                    if candle_ts > last_ts:
                        new_data.append(candle)
                    else:
                        stop_collecting = True
                        break
                all_data.extend(new_data)
                if stop_collecting:
                    logger.info("Reached the last saved timestamp. Stopping collection.")
                    break
            else:
                all_data.extend(data)

            oldest_ts_str = data[-1]['candle_date_time_utc']
            oldest_ts = datetime.strptime(oldest_ts_str, '%Y-%m-%dT%H:%M:%S')
            logger.info(f"Fetched {len(data)} records for {market}. Oldest timestamp: {oldest_ts}")

            if not last_ts and (datetime.utcnow() - oldest_ts).days >= days:
                logger.info(f"Collected approximately {days} days of data. Stopping collection.")
                break

            to_datetime = oldest_ts.strftime('%Y-%m-%dT%H:%M:%S')
            time.sleep(float(getattr(config.Data, "COLLECTOR_PAGE_SLEEP_SEC", 0.5) or 0.5))

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed for {market}: {e}")
            summary["status"] = "failed"
            summary["error"] = str(e)
            break

    if all_data:
        df = pd.DataFrame(all_data)
        df.rename(columns={
            'candle_date_time_utc': 'timestamp',
            'opening_price': 'open',
            'high_price': 'high',
            'low_price': 'low',
            'trade_price': 'close',
            'candle_acc_trade_volume': 'volume'
        }, inplace=True)
        
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df['market'] = market
        
        df = df.drop_duplicates(subset=['timestamp', 'market'])
        summary["saved_records"] = int(len(df))
        save_data(df, 'crypto_data')
        summary["status"] = "ok"
    else:
        logger.info(f"No new data collected for {market}.")
        if summary["status"] != "failed":
            summary["status"] = "up_to_date" if last_ts else "no_data"

    summary["elapsed_sec"] = round(float(time.time() - started_at), 3)
    if summary["status"] == "failed":
        logger.error(
            f"[CollectSummary] {market} failed attempts={summary['request_attempts']} "
            f"pages={summary['pages']} saved={summary['saved_records']} error={summary['error']}"
        )
    else:
        logger.info(
            f"[CollectSummary] {market} status={summary['status']} attempts={summary['request_attempts']} "
            f"pages={summary['pages']} saved={summary['saved_records']} elapsed={summary['elapsed_sec']}s"
        )
    return summary


def get_all_krw_markets():
    """Fetches all KRW market symbols from Upbit."""
    logger.info("Fetching all KRW market symbols from Upbit...")
    url = "https://api.upbit.com/v1/market/all"
    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        data = res.json()
        krw_markets = [item['market'] for item in data if item['market'].startswith('KRW-')]
        logger.info(f"Found {len(krw_markets)} KRW markets.")
        return krw_markets
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch market list from Upbit: {e}")
        return []

def run_all(days: int = 90):
    """Collects data for all KRW markets."""
    init_db() # Ensure database and tables are created
    logger.info(f"=== Starting data collection for ALL KRW markets ({days} days) ===")
    markets = get_all_krw_markets()
    if not markets:
        logger.error("Could not retrieve market list. Aborting data collection.")
        return

    for i, market in enumerate(markets):
        logger.info(f"--- Collecting market {i+1}/{len(markets)}: {market} ---")
        collect_market_data(market, days)
        time.sleep(1.1) # Be respectful to the API
    logger.info("=== Full data collection finished. ===")

def run(days: int = 90):
    logger.info("=== Starting Data Collection ===")
    # Backward-compatible default market set.
    for market in getattr(config.Data, "TRAIN_COINS", []):
        collect_market_data(market, days)
    logger.info("=== Data Collection Finished ===")

def get_current_price(market: str) -> Optional[float]:
    """Fetches the real-time trade price for a single market from Upbit."""
    # logger.info(f"Fetching real-time price for {market}...")
    url = "https://api.upbit.com/v1/ticker"
    params = {"markets": market}
    try:
        res = requests.get(url, params=params, timeout=10)
        res.raise_for_status()
        data = res.json()
        if data:
            price = data[0].get('trade_price')
            # logger.info(f"Real-time price for {market}: {price}")
            return price
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch current price for {market}: {e}")
    except (KeyError, IndexError) as e:
        logger.error(f"Failed to parse price data for {market}: {e}")
    return None
