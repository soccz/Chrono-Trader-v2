import sqlite3
import pandas as pd
from utils.config import config
from utils.logger import logger
from datetime import datetime, timedelta, timezone


def _parse_db_ts(ts: str):
    if not ts:
        return None
    try:
        # Stored as ISO strings like "YYYY-MM-DDTHH:MM:SS"
        dt = datetime.fromisoformat(ts)
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
    except Exception:
        return None


def get_latest_db_timestamp(markets: list = None):
    """Returns latest timestamp in DB (UTC) for given markets (or globally if None)."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        if markets:
            placeholders = ', '.join('?' for _ in markets)
            cur.execute(f"SELECT MAX(timestamp) FROM crypto_data WHERE market IN ({placeholders})", markets)
        else:
            cur.execute("SELECT MAX(timestamp) FROM crypto_data")
        row = cur.fetchone()
        max_ts = row[0] if row else None
        return _parse_db_ts(max_ts)
    except Exception:
        return None
    finally:
        conn.close()


def get_latest_db_timestamps_by_market(markets: list):
    """Returns {market: latest_timestamp_utc} for the provided markets."""
    if not markets:
        return {}

    conn = get_db_connection()
    try:
        cur = conn.cursor()
        placeholders = ", ".join("?" for _ in markets)
        cur.execute(
            f"SELECT market, MAX(timestamp) as max_ts FROM crypto_data WHERE market IN ({placeholders}) GROUP BY market",
            markets,
        )
        rows = cur.fetchall() or []
        out = {}
        for row in rows:
            m = row[0] if not isinstance(row, sqlite3.Row) else row["market"]
            ts = row[1] if not isinstance(row, sqlite3.Row) else row["max_ts"]
            out[m] = _parse_db_ts(ts)
        return out
    except Exception:
        return {}
    finally:
        conn.close()

def get_db_connection():
    """Creates a database connection using the path from the config."""
    conn = sqlite3.connect(config.General.DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    # Reduce temp-file usage and flakiness under large GROUP BY / ORDER BY queries.
    # Also improve concurrent read/write behavior for long-running jobs.
    try:
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA busy_timeout=5000")
    except Exception:
        # Best-effort; keep connection usable even if PRAGMAs fail.
        pass
    return conn

def init_db():
    """Initializes the database and creates tables."""
    logger.info("Initializing database...")
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS crypto_data (
        timestamp DATETIME,
        market TEXT,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume REAL,
        PRIMARY KEY (timestamp, market)
    )
    """)
    logger.info("Table 'crypto_data' created or already exists.")

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS metadata (
        key TEXT PRIMARY KEY,
        value TEXT
    )
    """)
    logger.info("Table 'metadata' created or already exists.")

    conn.commit()
    conn.close()

def save_data(df: pd.DataFrame, table_name: str):
    """Saves a DataFrame to the specified table."""
    if df.empty:
        logger.warning(f"DataFrame for table '{table_name}' is empty. Nothing to save.")
        return

    logger.info(f"Saving {len(df)} records to table '{table_name}'...")
    conn = get_db_connection()
    try:
        df.to_sql(table_name, conn, if_exists='append', index=False)
        logger.info("Data saved successfully.")
    except Exception as e:
        logger.error(f"Error saving data to table '{table_name}': {e}")
    finally:
        conn.close()

def load_data(query: str, params=None) -> pd.DataFrame:
    """Loads data from the database using a SQL query."""
    # logger.info(f"Executing query: {query}")
    conn = get_db_connection()
    try:
        df = pd.read_sql_query(query, conn, params=params)
        # logger.info(f"Loaded {len(df)} records.")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

def get_data_period() -> int:
    """
    Calculates the total number of days of data available in the database.
    """
    logger.info("Calculating total data period available in the database...")
    query = "SELECT MIN(timestamp), MAX(timestamp) FROM crypto_data"
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        result = cursor.fetchone()
        if result and result[0] and result[1]:
            min_ts_str, max_ts_str = result
            min_dt = datetime.fromisoformat(min_ts_str)
            max_dt = datetime.fromisoformat(max_ts_str)
            days = (max_dt - min_dt).days
            logger.info(f"Data available from {min_dt} to {max_dt} ({days} days).")
            return days
    except Exception as e:
        logger.error(f"Failed to calculate data period: {e}")
    finally:
        conn.close()
    
    logger.warning("Could not determine data period. Defaulting to 0.")
    return 0

def get_trading_values_for_markets(markets: list, end_time: datetime, hours: int = config.Recommender.LIQUIDITY_LOOKBACK_HOURS) -> dict:
    """
    Calculates the total trading value (SUM of close * volume) for a list of markets
    over a recent period ending at `end_time`.
    """
    if not markets:
        return {}

    logger.info(f"Calculating recent trading value for {len(markets)} markets...")

    if isinstance(end_time, str):
        logger.info("[DB] Received string end_time for trading value query; coercing to datetime.")
        end_time = pd.to_datetime(end_time, utc=True, errors='coerce')
    if isinstance(end_time, pd.Timestamp):
        end_time = end_time.to_pydatetime()
    if isinstance(end_time, pd.Series) or isinstance(end_time, pd.DataFrame):
        logger.info(f"[DB] Received pandas object for end_time (type={type(end_time)}); coercing to scalar timestamp.")
        end_time = pd.to_datetime(end_time.squeeze(), utc=True, errors='coerce')
        if isinstance(end_time, pd.Timestamp):
            end_time = end_time.to_pydatetime()
    if not isinstance(end_time, datetime):
        logger.error(f"[DB] end_time could not be coerced to datetime (type={type(end_time)}); aborting trading value calculation.")
        return {}

    # Clamp to DB-latest timestamp so "recent window" is aligned to available data.
    db_latest = get_latest_db_timestamp(markets=markets)
    if db_latest is not None and end_time > db_latest:
        delta_h = (end_time - db_latest).total_seconds() / 3600.0
        logger.warning(
            f"[DB] end_time ({end_time.isoformat()}) is ahead of DB latest ({db_latest.isoformat()}) "
            f"by {delta_h:.2f}h; clamping end_time to DB latest for liquidity query."
        )
        end_time = db_latest

    logger.info(f"[DB] Using end_time={end_time} ({type(end_time)}) for trading value query")

    start_time = end_time - timedelta(hours=hours)
    market_placeholders = ', '.join('?' for _ in markets)
    
    query = f"""
        SELECT
            market,
            SUM(close * volume) as trading_value
        FROM
            crypto_data
        WHERE
            market IN ({market_placeholders})
            AND timestamp BETWEEN ? AND ?
        GROUP BY
            market
    """
    
    # DB timestamps are stored as ISO strings like "YYYY-MM-DDTHH:MM:SS" (no spaces).
    params = markets + [start_time.strftime('%Y-%m-%dT%H:%M:%S'), end_time.strftime('%Y-%m-%dT%H:%M:%S')]

    conn = get_db_connection()
    results = {}
    try:
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        for row in rows:
            results[row['market']] = row['trading_value']
        logger.info(f"Successfully fetched trading values for {len(results)} markets.")
    except Exception as e:
        logger.error(f"Failed to get trading values: {e}")
    finally:
        conn.close()
        
    return results


def get_top_markets_by_trading_value(limit: int = 20, hours: int = 24, market_prefix: str = "KRW-") -> list:
    """
    Returns top markets (by SUM(close*volume)) over the last `hours`, aligned to DB latest timestamp.
    Intended for selecting a "refresh set" / watchlist in scheduled inference runs.
    """
    try:
        limit = int(limit)
        hours = int(hours)
    except Exception:
        limit, hours = 20, 24

    if limit <= 0:
        return []

    end_time = get_latest_db_timestamp()
    if end_time is None:
        return []

    start_time = end_time - timedelta(hours=hours)
    query = (
        "SELECT market, SUM(close * volume) as trading_value "
        "FROM crypto_data "
        "WHERE market LIKE ? AND timestamp BETWEEN ? AND ? "
        "GROUP BY market "
        "ORDER BY trading_value DESC "
        "LIMIT ?"
    )
    params = [
        f"{market_prefix}%",
        start_time.strftime("%Y-%m-%dT%H:%M:%S"),
        end_time.strftime("%Y-%m-%dT%H:%M:%S"),
        limit,
    ]

    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(query, params)
        rows = cur.fetchall() or []
        out = []
        for row in rows:
            if isinstance(row, sqlite3.Row):
                out.append(row["market"])
            else:
                out.append(row[0])
        return out
    except Exception:
        return []
    finally:
        conn.close()
