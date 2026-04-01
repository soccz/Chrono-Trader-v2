#!/usr/bin/env python3
"""
전체 데이터 수집 스크립트: 바이낸스 USDT + 업비트 KRW
- 바이낸스: 180개 마켓 × 2년 (메인 학습 데이터)
- 업비트: 243개 마켓 × 전체 히스토리 (추론 + 업비트 전용 코인)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import time
import sqlite3
import pandas as pd
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "crypto_data.db")

# ── DB helpers ──────────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("CREATE TABLE IF NOT EXISTS crypto_data (timestamp TEXT, market TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL)")
    conn.execute("CREATE TABLE IF NOT EXISTS binance_data (timestamp TEXT, market TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_crypto_ts_mkt ON crypto_data(timestamp, market)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_binance_ts_mkt ON binance_data(timestamp, market)")
    return conn

def save_batch(conn, table, rows):
    if not rows:
        return
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=['timestamp', 'market'])
    df.to_sql(table, conn, if_exists='append', index=False)

def get_latest_ts(conn, table, market):
    cur = conn.execute(f"SELECT MAX(timestamp) FROM {table} WHERE market=?", (market,))
    r = cur.fetchone()
    return r[0] if r and r[0] else None

def get_oldest_ts(conn, table, market):
    cur = conn.execute(f"SELECT MIN(timestamp) FROM {table} WHERE market=?", (market,))
    r = cur.fetchone()
    return r[0] if r and r[0] else None

def count_rows(conn, table, market):
    cur = conn.execute(f"SELECT COUNT(*) FROM {table} WHERE market=?", (market,))
    return cur.fetchone()[0]

# ── 바이낸스 수집 ──────────────────────────────────────────────────────────
def get_binance_usdt_markets():
    """업비트 KRW와 겹치는 바이낸스 USDT 마켓 목록"""
    # 업비트 KRW 마켓
    resp = requests.get("https://api.upbit.com/v1/market/all", timeout=10)
    upbit_symbols = {m['market'].replace('KRW-', '') for m in resp.json() if m['market'].startswith('KRW-')}

    # 바이낸스 USDT 마켓
    resp2 = requests.get("https://api.binance.com/api/v3/exchangeInfo", timeout=10)
    binance_usdt = {s['symbol'].replace('USDT', ''): s['symbol']
                    for s in resp2.json()['symbols']
                    if s['quoteAsset'] == 'USDT' and s['status'] == 'TRADING'}

    overlap = []
    for sym in upbit_symbols:
        if sym in binance_usdt:
            overlap.append((f"KRW-{sym}", binance_usdt[sym]))

    # BTC, ETH 먼저
    priority = ['BTC', 'ETH', 'XRP', 'SOL', 'DOGE']
    overlap.sort(key=lambda x: (x[0].replace('KRW-', '') not in priority, x[0]))

    logger.info(f"바이낸스 USDT ∩ 업비트 KRW: {len(overlap)}개")
    return overlap

def collect_binance(symbol, days=730, conn=None):
    """바이낸스 1h 캔들 수집 (1000개/요청)"""
    table = "binance_data"
    market = f"KRW-{symbol.replace('USDT', '')}"  # 업비트 마켓명으로 저장

    # 이미 있는 데이터 확인
    existing = count_rows(conn, table, market)
    target_hours = days * 24
    if existing >= target_hours * 0.95:
        logger.info(f"  {symbol}: 이미 {existing:,}행 있음, skip")
        return existing

    end_ms = int(datetime.utcnow().timestamp() * 1000)
    start_ms = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)

    # 기존 데이터가 있으면 최신 이후부터만 수집
    latest = get_latest_ts(conn, table, market)
    if latest:
        latest_dt = datetime.strptime(latest[:19], '%Y-%m-%dT%H:%M:%S')
        start_ms = int(latest_dt.timestamp() * 1000)

    all_rows = []
    cursor_ms = start_ms
    retries = 0

    while cursor_ms < end_ms:
        try:
            resp = requests.get("https://api.binance.com/api/v3/klines", params={
                "symbol": symbol, "interval": "1h", "limit": 1000,
                "startTime": cursor_ms
            }, timeout=15)

            if resp.status_code == 429:
                logger.warning(f"  {symbol}: rate limited, waiting 60s")
                time.sleep(60)
                continue

            resp.raise_for_status()
            data = resp.json()
            if not data:
                break

            for k in data:
                ts = datetime.utcfromtimestamp(k[0] / 1000).strftime('%Y-%m-%dT%H:%M:%S')
                all_rows.append({
                    'timestamp': ts, 'market': market,
                    'open': float(k[1]), 'high': float(k[2]),
                    'low': float(k[3]), 'close': float(k[4]),
                    'volume': float(k[5])
                })

            cursor_ms = data[-1][0] + 3600000  # 다음 캔들
            retries = 0

            if len(all_rows) >= 5000:
                save_batch(conn, table, all_rows)
                conn.commit()
                all_rows = []

            time.sleep(0.05)  # 바이낸스 rate limit 여유

        except requests.exceptions.RequestException as e:
            retries += 1
            if retries > 3:
                logger.error(f"  {symbol}: 3회 실패, 중단 - {e}")
                break
            time.sleep(2 ** retries)

    if all_rows:
        save_batch(conn, table, all_rows)
        conn.commit()

    final = count_rows(conn, table, market)
    return final

# ── 업비트 수집 ────────────────────────────────────────────────────────────
def get_upbit_krw_markets():
    resp = requests.get("https://api.upbit.com/v1/market/all", timeout=10)
    markets = [m['market'] for m in resp.json() if m['market'].startswith('KRW-')]
    # BTC, ETH 먼저
    priority = ['KRW-BTC', 'KRW-ETH', 'KRW-XRP', 'KRW-SOL', 'KRW-DOGE']
    markets.sort(key=lambda x: (x not in priority, x))
    logger.info(f"업비트 KRW 마켓: {len(markets)}개")
    return markets

def collect_upbit(market, days=900, conn=None):
    """업비트 1h 캔들 수집 (200개/요청)"""
    table = "crypto_data"

    existing = count_rows(conn, table, market)
    target_hours = days * 24
    if existing >= target_hours * 0.9:
        logger.info(f"  {market}: 이미 {existing:,}행 있음, skip")
        return existing

    url = "https://api.upbit.com/v1/candles/minutes/60"
    target_date = datetime.utcnow() - timedelta(days=days)

    # 가장 오래된 데이터 확인 → 그 이전부터 수집
    oldest = get_oldest_ts(conn, table, market)
    if oldest:
        to_dt = oldest[:19]
    else:
        to_dt = (datetime.utcnow() + timedelta(hours=1)).strftime('%Y-%m-%dT%H:%M:%S')

    # 최신 데이터도 수집 (앞쪽 갭 채우기)
    latest = get_latest_ts(conn, table, market)
    if latest:
        latest_dt = datetime.strptime(latest[:19], '%Y-%m-%dT%H:%M:%S')
        if datetime.utcnow() - latest_dt > timedelta(hours=2):
            _collect_upbit_forward(market, latest, conn)

    # 과거 방향 수집
    all_rows = []
    retries = 0

    while True:
        try:
            resp = requests.get(url, params={"market": market, "count": 200, "to": to_dt}, timeout=10)

            if resp.status_code == 429:
                time.sleep(1)
                continue

            if resp.status_code != 200:
                break

            data = resp.json()
            if not data or (isinstance(data, dict) and "error" in data):
                break

            for c in data:
                all_rows.append({
                    'timestamp': c['candle_date_time_utc'],
                    'market': market,
                    'open': c['opening_price'],
                    'high': c['high_price'],
                    'low': c['low_price'],
                    'close': c['trade_price'],
                    'volume': c['candle_acc_trade_volume']
                })

            oldest_in_batch = data[-1]['candle_date_time_utc']
            oldest_dt = datetime.strptime(oldest_in_batch[:19], '%Y-%m-%dT%H:%M:%S')

            if oldest_dt <= target_date:
                break

            to_dt = oldest_in_batch
            retries = 0

            if len(all_rows) >= 5000:
                save_batch(conn, table, all_rows)
                conn.commit()
                all_rows = []

            time.sleep(0.15)  # 업비트 rate limit

        except requests.exceptions.RequestException as e:
            retries += 1
            if retries > 3:
                logger.error(f"  {market}: 3회 실패, 중단 - {e}")
                break
            time.sleep(2 ** retries)

    if all_rows:
        save_batch(conn, table, all_rows)
        conn.commit()

    final = count_rows(conn, table, market)
    return final

def _collect_upbit_forward(market, latest_ts, conn):
    """최신 데이터 채우기 (latest → 현재)"""
    url = "https://api.upbit.com/v1/candles/minutes/60"
    all_rows = []

    now_str = (datetime.utcnow() + timedelta(hours=1)).strftime('%Y-%m-%dT%H:%M:%S')
    resp = requests.get(url, params={"market": market, "count": 200, "to": now_str}, timeout=10)
    if resp.status_code == 200 and resp.json():
        for c in resp.json():
            if c['candle_date_time_utc'] > latest_ts:
                all_rows.append({
                    'timestamp': c['candle_date_time_utc'],
                    'market': market,
                    'open': c['opening_price'], 'high': c['high_price'],
                    'low': c['low_price'], 'close': c['trade_price'],
                    'volume': c['candle_acc_trade_volume']
                })
    if all_rows:
        save_batch(conn, "crypto_data", all_rows)
        conn.commit()
        logger.info(f"  {market}: +{len(all_rows)} 최신 행 추가")

# ── 메인 ───────────────────────────────────────────────────────────────────
def main():
    conn = get_db()
    started = time.time()

    # Phase 1: 바이낸스 USDT (180개 × 2년)
    print("=" * 60)
    print("Phase 1: 바이낸스 USDT 수집 (2년치)")
    print("=" * 60)

    binance_pairs = get_binance_usdt_markets()
    binance_total = 0
    for i, (upbit_mkt, binance_sym) in enumerate(binance_pairs):
        rows = collect_binance(binance_sym, days=730, conn=conn)
        binance_total += rows
        if (i + 1) % 20 == 0:
            elapsed = time.time() - started
            logger.info(f"  바이낸스 진행: {i+1}/{len(binance_pairs)} ({binance_total:,} rows, {elapsed/60:.1f}분)")

    print(f"\n바이낸스 완료: {binance_total:,} rows")

    # Phase 2: 업비트 KRW (243개 × 전체)
    print("=" * 60)
    print("Phase 2: 업비트 KRW 수집 (전체 히스토리)")
    print("=" * 60)

    upbit_markets = get_upbit_krw_markets()
    upbit_total = 0
    phase2_start = time.time()
    for i, market in enumerate(upbit_markets):
        rows = collect_upbit(market, days=900, conn=conn)  # ~2.5년
        upbit_total += rows
        if (i + 1) % 30 == 0:
            elapsed = time.time() - phase2_start
            logger.info(f"  업비트 진행: {i+1}/{len(upbit_markets)} ({upbit_total:,} rows, {elapsed/60:.1f}분)")

    print(f"\n업비트 완료: {upbit_total:,} rows")

    # 최종 통계
    elapsed_total = (time.time() - started) / 60

    cur = conn.cursor()
    cur.execute("SELECT COUNT(*), COUNT(DISTINCT market) FROM crypto_data")
    up_rows, up_mkts = cur.fetchone()
    cur.execute("SELECT COUNT(*), COUNT(DISTINCT market) FROM binance_data")
    bn_rows, bn_mkts = cur.fetchone()

    print("\n" + "=" * 60)
    print(f"전체 수집 완료 ({elapsed_total:.1f}분)")
    print(f"  업비트:  {up_rows:>10,} rows | {up_mkts:>3} markets")
    print(f"  바이낸스: {bn_rows:>10,} rows | {bn_mkts:>3} markets")
    print(f"  합계:    {up_rows + bn_rows:>10,} rows")
    print(f"  DB 크기: {os.path.getsize(DB_PATH)/1024/1024:.0f} MB")
    print("=" * 60)

    conn.close()

if __name__ == "__main__":
    main()
