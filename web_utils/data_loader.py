import pandas as pd
import glob
import os
import math
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional

# Add project paths to import utils
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.price_cache import get_prices_batch, get_cached_price
from utils.portfolio_manager import portfolio_manager
from utils.output_contract import read_output_manifest

class DataLoader:
    """Utility for loading and parsing CSV data files"""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = base_dir
        self.recommendations_dir = os.path.join(base_dir, "recommendations")
        self.predictions_dir = os.path.join(base_dir, "predictions")
        self.archive_dir = os.path.join(self.recommendations_dir, "archive")

    def _read_csv(self, path: Optional[str]) -> Optional[pd.DataFrame]:
        if not path or not os.path.exists(path):
            return None
        try:
            return pd.read_csv(path)
        except Exception as e:
            print(f"Error reading CSV {path}: {e}")
            return None

    def _manifest_file_path(self, mode: str, key: str) -> Optional[str]:
        manifest = read_output_manifest(mode)
        if not manifest:
            return None
        entry = manifest.get(key) or {}
        path = entry.get("path")
        if not path:
            return None
        if os.path.isabs(path):
            return path
        return os.path.join(self.base_dir, path)

    def get_latest_output_contracts(self) -> Dict[str, Dict]:
        contracts = {}
        for mode in ("morning", "intraday", "refresh-db"):
            manifest = read_output_manifest(mode)
            if manifest:
                contracts[mode] = manifest
        return contracts
    
    def get_latest_recommendations(self) -> Optional[pd.DataFrame]:
        """Get the most recent recommendations, preferring scheduled-run manifests."""
        try:
            dfs = []
            seen_paths = set()

            for mode, strategy_type in (
                ("morning", "Morning Snapshot"),
                ("intraday", "Intraday"),
            ):
                path = self._manifest_file_path(mode, "recommendation")
                if path and path not in seen_paths:
                    df = self._read_csv(path)
                    if df is not None and not df.empty:
                        df['strategy_type'] = strategy_type
                        df['output_mode'] = mode
                        dfs.append(df)
                        seen_paths.add(path)

            if dfs:
                return pd.concat(dfs, ignore_index=True)
            
            # Prioritize root folder (Current Week)
            # 1. Get latest Daily (Long-Term) file
            daily_files = sorted(
                glob.glob(os.path.join(self.recommendations_dir, "recs_daily_*.csv")),
                reverse=True
            )
            if daily_files:
                daily_df = pd.read_csv(daily_files[0])
                daily_df['strategy_type'] = 'Daily (Long-Term)'
                dfs.append(daily_df)
                
            # 2. Get latest Short (Continuous) file
            short_files = sorted(
                glob.glob(os.path.join(self.recommendations_dir, "recs_short_*.csv")),
                reverse=True
            )
            if short_files:
                short_df = pd.read_csv(short_files[0])
                short_df['strategy_type'] = 'Short (4H Scalp)'
                dfs.append(short_df)
                
            # 3. Fallback
            if not dfs:
                # Logic same as before...
                generic_files = sorted(
                    glob.glob(os.path.join(self.recommendations_dir, "recs_*.csv")),
                    reverse=True
                )
                generic_files = [f for f in generic_files if "_daily_" not in f and "_short_" not in f]
                if generic_files:
                    return pd.read_csv(generic_files[0])

            if dfs:
                return pd.concat(dfs, ignore_index=True)
                
            return None
        except Exception as e:
            print(f"Error loading recommendations: {e}")
            return None
    
    def get_latest_pump_predictions(self) -> Optional[pd.DataFrame]:
        try:
            for mode in ("morning", "intraday"):
                path = self._manifest_file_path(mode, "pump_prediction")
                df = self._read_csv(path)
                if df is not None and not df.empty:
                    return df
            pump_files = sorted(
                glob.glob(os.path.join(self.predictions_dir, "pump_preds_*.csv")),
                reverse=True
            )
            if pump_files:
                return pd.read_csv(pump_files[0])
            return None
        except Exception as e:
            print(f"Error loading pump predictions: {e}")
            return None
    
    def get_all_recommendations(self, limit: int = 10) -> List[pd.DataFrame]:
        # Same as before
        try:
            rec_files = sorted(
                glob.glob(os.path.join(self.recommendations_dir, "recs_*.csv")),
                reverse=True
            )[:limit]
            return [pd.read_csv(f) for f in rec_files if os.path.exists(f)]
        except Exception as e:
            print(f"Error loading recommendation history: {e}")
            return []
    
    def calculate_performance_metrics(self, recommendations_df: pd.DataFrame, 
                                     current_prices: Dict[str, float]) -> Dict:
        if recommendations_df is None or recommendations_df.empty:
            return {
                "total": 0, "accuracy": 0, "avg_error": 0, "results": []
            }
        
        results = []
        for _, row in recommendations_df.iterrows():
            market = row['market']
            predicted_return = row['expected_return']
            entry_price = row['current_price']
            signal = row.get('signal', 'Unknown')
            
            # Using injected current_prices (from batch fetch)
            current_price = current_prices.get(market)
            if not current_price:
                 # Fallback to cached single fetch
                 current_price = get_cached_price(market)
            
            if current_price and current_price > 0:
                actual_return = (current_price - entry_price) / entry_price
                if signal == 'Short':
                    actual_return = -actual_return
                
                error = abs(predicted_return - actual_return)
                direction_correct = (predicted_return * actual_return) > 0
                
            # Safe float conversion helper
            def safe_float(val, default=0.0):
                if val is None:
                    return default
                try:
                    f = float(val)
                    return default if math.isnan(f) else f
                except (ValueError, TypeError):
                    return default

            # Dynamic position_size: if not in CSV, calculate using composite formula
            csv_position_size = safe_float(row.get('position_size'))
            csv_volatility = safe_float(row.get('volatility'), 0.01)
            csv_confidence = safe_float(row.get('confidence'), 0.5)
            
            if csv_position_size > 0:
                final_position_size = csv_position_size
            else:
                # Match recommender.py: Composite (Confidence × Volatility)
                base_position = 0.10  # 10% base
                max_position = 0.20   # 20% max
                min_position = 0.03   # 3% min
                
                confidence_factor = max(0.5, min(1.0, csv_confidence))
                volatility_factor = 1 / (1 + csv_volatility * 5)
                
                final_position_size = base_position * confidence_factor * volatility_factor
                final_position_size = max(min_position, min(max_position, final_position_size))

            results.append({
                'market': market,
                'signal': signal,
                'strategy_type': row.get('strategy_type', 'Unknown'),
                'predicted_return': safe_float(predicted_return),
                'actual_return': safe_float(actual_return),
                'error': safe_float(error),
                'direction_correct': bool(direction_correct),
                'entry_price': safe_float(entry_price),
                'current_price': safe_float(current_price),
                'position_size': final_position_size,
                'volatility': csv_volatility
            })
        
        if results:
            accuracy = sum(1 for r in results if r['direction_correct']) / len(results)
            avg_error = sum(r['error'] for r in results) / len(results)
        else:
            accuracy = 0
            avg_error = 0
        
        return {
            "total": len(results),
            "accuracy": round(accuracy * 100, 1),
            "avg_error": round(avg_error * 100, 1),
            "results": results
        }

    def get_equity_curve(self) -> List[Dict]:
        try:
            curve = portfolio_manager.get_equity_curve()
            # Sanitize NaN
            clean_curve = []
            for point in curve:
                clean_curve.append({
                    'time': point['time'],
                    'value': point['value'] if not math.isnan(point['value']) else 0,
                    'pct_change': point['pct_change'] if not math.isnan(point['pct_change']) else 0
                })
            return clean_curve
        except Exception as e:
            print(f"Error loading equity curve: {e}")
            return []

    def get_trade_history(self, limit: int = 100) -> List[Dict]:
        try:
            return portfolio_manager.get_trade_history(limit=limit)
        except Exception as e:
            print(f"Error loading trade history: {e}")
            return []

    def get_portfolio_summary(self) -> Dict:
        """New: Get composition for Pie Chart"""
        try:
            return portfolio_manager.get_portfolio_summary()
        except:
            return {'total_equity': 0, 'cash': 0, 'positions': []}

    def get_aggregated_history(self, limit: int = 30) -> Dict:
        """Aggregates historical performance metrics from past recommendation files."""
        try:
            # Get all recommendation files
            all_files = sorted(glob.glob(os.path.join(self.recommendations_dir, "**", "recs_*.csv"), recursive=True))
            
            # Optimization: Slice BEFORE processing to avoid API limits on thousands of files
            if limit > 0:
                all_files = all_files[-limit:]
            
            if limit > 0:
                all_files = all_files[-limit:]
            
            history = []
            
            # Batch Price Fetching
            # 1. Collect all unique markets from the selected files
            all_markets = set()
            loaded_dfs = []
            
            for file_path in all_files:
                try:
                    df = pd.read_csv(file_path)
                    if not df.empty:
                        loaded_dfs.append((file_path, df))
                        all_markets.update(df['market'].unique())
                except: pass
            
            # 2. Fetch prices once for all markets using batch API
            current_prices_map = get_prices_batch(list(all_markets)) 
            
            # 3. Process each file using cached prices
            for file_path, df in loaded_dfs:
                try:
                    filename = os.path.basename(file_path)
                    parts = filename.replace('.csv', '').split('_')
                    # Debug print
                    # print(f"Processing {filename}...") 
                    date_str = None
                    for part in parts:
                        if len(part) == 8 and part.isdigit(): 
                            date_str = part
                            idx = parts.index(part)
                            if idx + 1 < len(parts) and len(parts[idx+1]) == 6 and parts[idx+1].isdigit():
                                try:
                                    dt = datetime.strptime(date_str + parts[idx+1], "%Y%m%d%H%M%S")
                                    date_str = dt.strftime("%m-%d %H:%M")
                                except: pass
                            break
                    
                    if not date_str: 
                        print(f"Skipping {filename}: No date found")
                        continue

                    metrics = self.calculate_performance_metrics(df, current_prices_map)
                    
                    if metrics['total'] > 0:
                        history.append({
                            'date': date_str,
                            'accuracy': metrics['accuracy'],
                            'avg_error': metrics['avg_error'],
                            'total_recs': metrics['total']
                        })
                    else:
                        print(f"Skipping {filename}: Metrics total is 0")
                except Exception as e: 
                    print(f"Error processing {file_path}: {e}")
                    continue
            
            if limit > 0:
                history = history[-limit:]
                
            return {
                "labels": [h['date'] for h in history],
                "accuracy": [h['accuracy'] for h in history],
                "avg_error": [h['avg_error'] for h in history]
            }
        except Exception as e:
            print(f"Error aggregating history: {e}")
            return {"labels": [], "accuracy": [], "avg_error": []}
         
    def get_total_stats(self) -> Dict:
        """
        Scans all CSVs to calculate:
        - Total Prediction Count (All time)
        - Average Accuracy (All time) estimated
        """
        all_files = glob.glob(os.path.join(self.recommendations_dir, "**", "recs_*.csv"), recursive=True)
        total_count = 0
        # Accuracy is hard to calc without verifying every prediction.
        # Just return count for now.
        for f in all_files:
            try:
                # fast line count
                with open(f) as file:
                    total_count += sum(1 for line in file) - 1 # header
            except: pass
        
        # Calculate estimated accuracy from recent history (last 50 files)
        try:
            history = self.get_aggregated_history(limit=50)
            if history and history.get('accuracy') and len(history['accuracy']) > 0:
                avg_acc = sum(history['accuracy']) / len(history['accuracy'])
            else:
                avg_acc = 0
        except Exception as e:
            print(f"Error calculating estimated accuracy: {e}")
            avg_acc = 0
            
        return {"total": max(0, total_count), "accuracy": round(avg_acc, 1)}

    def get_weekly_stats(self) -> Dict:
        """
        Returns stats for the current week (Monday ~ Now).
        - Portfolio Return (Weekly)
        - Win Rate (Closed Trades this week)
        - Realized Profit (Closed Trades this week)
        """
        today = datetime.now()
        start_of_week = today - timedelta(days=today.weekday())
        start_of_week = start_of_week.replace(hour=0, minute=0, second=0, microsecond=0)
        
        try:
            # 1. Realized Profit & Win Rate from CLOSED trades
            trades = portfolio_manager.get_trade_history(limit=1000)
            weekly_trades = []
            for t in trades:
                if t['status'] == 'CLOSED' and t['exit_time']:
                    # Parse exit_time if string
                    et = t['exit_time']
                    if isinstance(et, str):
                        try:
                            if '.' in et:
                                et = datetime.strptime(et, '%Y-%m-%d %H:%M:%S.%f')
                            else:
                                et = datetime.strptime(et, '%Y-%m-%d %H:%M:%S')
                        except ValueError:
                            continue
                        except: 
                            try: et = datetime.strptime(et, '%Y-%m-%d %H:%M:%S')
                            except: pass
                    
                    if isinstance(et, datetime) and et >= start_of_week:
                        weekly_trades.append(t)

            realized_profit_krw = 0
            STARTING_CAPITAL = 1000000  # 시뮬레이션 시작 자본
            
            win_count = 0
            for t in weekly_trades:
                pnl = t.get('pnl_percent', 0) or 0
                
                # pnl_percent가 없으면 entry/exit price로 직접 계산
                if pnl == 0:
                    entry_p = t.get('entry_price', 0) or 0
                    exit_p = t.get('exit_price', 0) or 0
                    if entry_p > 0 and exit_p > 0:
                        raw_return = (exit_p - entry_p) / entry_p
                        signal = t.get('signal', 'Long')
                        pnl = -raw_return if signal == 'Short' else raw_return
                
                val = t.get('position_value', 0) or 0
                
                # Fallback: position_value가 없으면 position_size 기반으로 계산
                if val <= 0:
                    pos_size = t.get('position_size', 0.1) or 0.1
                    val = STARTING_CAPITAL * pos_size
                
                realized_profit_krw += val * pnl
                
                if pnl > 0: win_count += 1
            
            win_rate = (win_count / len(weekly_trades) * 100) if weekly_trades else 0
            
            # Weekly Return (Portfolio)
            # Using Equity Curve
            curve = self.get_equity_curve()
            if curve:
                # Find value at start of week
                start_val = curve[0]['value']
                for pt in curve:
                    # Parse time
                    pt_time = pt['time']
                    # ... compare date ...
                    pass
                current_val = curve[-1]['value']
                # Simplified
                weekly_return = 0 
                
            return {
                "profit": realized_profit_krw,
                "win_rate": win_rate,
                "return": 0, # TODO
                "count": len(weekly_trades)
            }
        except Exception as e:
            print(f"Weekly stats error: {e}")
            return {"profit": 0, "win_rate": 0, "return": 0, "count": 0}
