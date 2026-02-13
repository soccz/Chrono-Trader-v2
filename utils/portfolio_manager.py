import sqlite3
import pandas as pd
import os
from datetime import datetime
from utils.logger import logger
from utils.config import config

class PortfolioManager:
    """
    Manages a simulated portfolio ("Paper Trading") using a local SQLite database.
    Tracks trades, calculates PnL, and maintains trade history.
    """
    def __init__(self, db_path=None):
        if db_path is None:
            # Default to data/portfolio.db
            self.db_path = os.path.join("data", "portfolio.db")
        else:
            self.db_path = db_path
            
        self._init_db()

    def _get_conn(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """Initialize the trades table if it doesn't exist."""
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    market TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    entry_time TIMESTAMP NOT NULL,
                    exit_price REAL,
                    exit_time TIMESTAMP,
                    status TEXT DEFAULT 'OPEN',
                    pnl_percent REAL,
                    position_value REAL,  -- New: Value of position in KRW
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create index for faster querying by status and strategy
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON trades(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_strategy ON trades(strategy)")
            
            # Migration check: Add position_value if missing
            try:
                cursor.execute("SELECT position_value FROM trades LIMIT 1")
            except sqlite3.OperationalError:
                logger.info("Migrating DB: Adding position_value column")
                cursor.execute("ALTER TABLE trades ADD COLUMN position_value REAL")

            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to initialize portfolio DB: {e}")

    def add_trade(self, market, strategy, signal, entry_price, entry_time=None, position_value=0):
        """Records a new open trade."""
        if entry_time is None:
            entry_time = datetime.now()
            
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO trades (market, strategy, signal, entry_price, entry_time, status, position_value)
                VALUES (?, ?, ?, ?, ?, 'OPEN', ?)
            """, (market, strategy, signal, entry_price, entry_time, position_value))
            
            conn.commit()
            conn.close()
            logger.info(f"opened trade: {market} ({signal}) @ {entry_price} [Val: {position_value}]")
        except Exception as e:
            logger.error(f"Failed to add trade for {market}: {e}")

    def get_portfolio_summary(self):
        """
        Returns snapshot of current portfolio for Pie Chart.
        Returns: { 'total_krw': float, 'cash': float, 'positions': [{'market':..., 'value':..., 'weight':...}] }
        """
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute("SELECT market, position_value FROM trades WHERE status='OPEN'")
            rows = cursor.fetchall()
            conn.close()

            positions = []
            invested_sum = 0
            for market, val in rows:
                val = val or 0 # Handle None
                # Fallback: If position_value is 0 (migrated data), estimate using default size
                if val == 0:
                    val = 10000000 * 0.1 # 10M * 10% assumption
                invested_sum += val
                positions.append({'market': market, 'value': val})
            
            # Assume constant total capital for simulation OR derive from equity curve
            # For this MVP, let's assume we started with 10M KRW and add realized PnL? 
            # Or simpler: Total = Cash + Invested.
            # But we don't track Cash withdrawal.
            # Let's simple assumption: Total Portfolio Value = Equity Curve Last Value (normalized to 1M or 10M base)
            # OR fetch from `get_equity_curve` last value * scale?
            # Let's use a fixed simulation base for now or allow config.
            
            total_equity = 10000000 # 10 Million KRW Base Simulation
            
            # Adjust equity by realized profit?
            # Creating a complex ledger is out of scope for this task.
            # Let's just use Invested vs Cash (Remaining).
            
            cash = max(0, total_equity - invested_sum) # Don't go negative
            
            # Calculate weights
            for p in positions:
                p['weight'] = (p['value'] / total_equity) * 100 if total_equity > 0 else 0
            
            return {
                'total_equity': total_equity,
                'cash': cash,
                'cash_weight': (cash / total_equity) * 100 if total_equity > 0 else 0,
                'positions': positions
            }
        except Exception as e:
            logger.error(f"Summary error: {e}")
            return {'total_equity': 0, 'cash': 0, 'positions': []}

    def close_open_trades(self, strategy, current_prices, exit_time=None):
        """
        Closes all OPEN trades for a specific strategy using current market prices.
        This is typically called at the beginning of a run to realize PnL from the previous period.
        """
        if exit_time is None:
            exit_time = datetime.now()

        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            
            # Get open trades for this strategy
            cursor.execute("""
                SELECT id, market, signal, entry_price FROM trades 
                WHERE status = 'OPEN' AND strategy = ?
            """, (strategy,))
            
            open_trades = cursor.fetchall()
            
            for trade_id, market, signal, entry_price in open_trades:
                current_price = current_prices.get(market)
                
                if current_price:
                    # Calculate PnL
                    # Long: (Exit - Entry) / Entry
                    # Short: (Entry - Exit) / Entry  == - (Exit - Entry) / Entry
                    raw_return = (current_price - entry_price) / entry_price
                    
                    if signal == 'Short':
                        pnl_percent = -raw_return
                    else:
                        pnl_percent = raw_return
                        
                    cursor.execute("""
                        UPDATE trades 
                        SET exit_price = ?, exit_time = ?, status = 'CLOSED', pnl_percent = ?
                        WHERE id = ?
                    """, (current_price, exit_time, pnl_percent, trade_id))
                    
                    logger.info(f"Closed trade: {market} ({signal}) PnL: {pnl_percent:.2%}")
                else:
                    logger.warning(f"No current price found for {market}, keeping trade OPEN.")
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to close trades for {strategy}: {e}")

    def get_equity_curve(self, initial_balance=100.0):
        """
        Calculates the cumulative equity curve based on closed trades.
        Assumes equal weight 100% allocation (simple compounding or summing).
        Here we assume a simple 'Index' model starting at 100.
        Daily Return = Average PnL of trades closed on that day.
        """
        try:
            conn = self._get_conn()
            # Fetch all closed trades sorted by exit time
            df = pd.read_sql_query("""
                SELECT exit_time, pnl_percent 
                FROM trades 
                WHERE status = 'CLOSED' 
                ORDER BY exit_time ASC
            """, conn)
            conn.close()
            
            if df.empty:
                return []
            
            # Convert exit_time to datetime
            df['exit_time'] = pd.to_datetime(df['exit_time'])
            
            # Group by time (e.g., hourly or daily blocks to aggregate concurrent trades)
            # Since strategies run in batches, we can group by exact exit_time or round to hour
            df['period'] = df['exit_time'].dt.strftime('%Y-%m-%d %H:00')
            
            equity_curve = []
            current_equity = initial_balance
            
            # Calculate average return per batch (assuming portfolio split among all signals in a batch)
            # If 5 signals, we invested 1/5th in each. So Batch Return = Sum(PnL) / Count
            # OR simpler: Return = Mean(PnL)
            
            batch_returns = df.groupby('period')['pnl_percent'].mean()
            
            for date, ret in batch_returns.items():
                # Compound return
                current_equity = current_equity * (1 + ret)
                equity_curve.append({
                    'time': date,
                    'value': current_equity,
                    'pct_change': ret
                })
                
            return equity_curve
            
        except Exception as e:
            logger.error(f"Failed to calculate equity curve: {e}")
            return []

    def get_trade_history(self, limit=50):
        try:
            conn = self._get_conn()
            df = pd.read_sql_query(f"""
                SELECT * FROM trades ORDER BY entry_time DESC LIMIT {limit}
            """, conn)
            conn.close()
            # Replace NaN with None for valid JSON serialization
            df = df.astype(object).where(pd.notnull(df), None)
            return df.to_dict('records')
        except Exception as e:
            logger.error(f"Failed to get trade history: {e}")
            return []

    def reset_portfolio(self):
        """
        Deletes ALL trades from the database. 
        Used for resetting the simulation/paper trading environment.
        """
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM trades") # No WHERE clause = Delete All
            # Reset AutoIncrement?
            cursor.execute("DELETE FROM sqlite_sequence WHERE name='trades'")
            
            conn.commit()
            conn.close()
            logger.info("Portfolio successfully reset. All trade history deleted.")
            return True
        except Exception as e:
            logger.error(f"Failed to reset portfolio: {e}")
            return False

# Global instance
portfolio_manager = PortfolioManager()
