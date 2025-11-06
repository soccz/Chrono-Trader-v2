import sqlite3
import pandas as pd
from utils.config import config
from utils.logger import logger
from datetime import datetime, timedelta

def get_db_connection():
    """Creates a database connection using the path from the config."""
    conn = sqlite3.connect(config.DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
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

def get_trading_values_for_markets(markets: list, end_time: datetime, hours: int = 24) -> dict:
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
    
    params = markets + [start_time.strftime('%Y-%m-%d %H:%M:%S'), end_time.strftime('%Y-%m-%d %H:%M:%S')]

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
