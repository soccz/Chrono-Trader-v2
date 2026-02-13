"""
Backtest Analyzer - 백테스트 성과 분석 모듈
7개 핵심 지표: Total Return, Sharpe Ratio, Max Drawdown, Win Rate, Alpha, Trade Count, Profit Factor
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import os

# 프로젝트 경로 설정
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.database import load_data


class BacktestAnalyzer:
    """백테스트 성과 분석기"""
    
    HOLDING_PERIOD_HOURS = 4  # 고정 보유 기간
    ANNUALIZATION_FACTOR = np.sqrt(365 * 24 / HOLDING_PERIOD_HOURS)  # 연율화 계수
    
    def __init__(self, recommendations_dir: str = "recommendations"):
        self.recommendations_dir = recommendations_dir
        self._predictions_df = None
        self._price_data = None
    
    def _load_predictions(self) -> pd.DataFrame:
        """recommendations 폴더에서 예측 데이터 로드"""
        if self._predictions_df is not None:
            return self._predictions_df
        
        if not os.path.exists(self.recommendations_dir):
            return pd.DataFrame()
        
        # recommendations 폴더의 모든 CSV 파일 읽기
        all_dfs = []
        import glob
        csv_files = glob.glob(os.path.join(self.recommendations_dir, "recs_*.csv"))
        
        for csv_file in csv_files:
            try:
                # 파일명에서 날짜 추출 (예: recs_20251228_223405.csv → 2025-12-28 22:34:05)
                basename = os.path.basename(csv_file)
                parts = basename.replace('recs_', '').replace('.csv', '').split('_')
                
                # 여러 형식 지원: recs_YYYYMMDD_HHMMSS.csv 또는 recs_standard_trending_YYYYMMDD_HHMMSS.csv
                date_str = None
                for p in parts:
                    if len(p) == 8 and p.isdigit():
                        date_str = p
                        idx = parts.index(p)
                        if idx + 1 < len(parts) and len(parts[idx + 1]) == 6:
                            time_str = parts[idx + 1]
                            date_str = f"{p}_{time_str}"
                        break
                
                if date_str is None:
                    continue
                
                # CSV 읽기
                df = pd.read_csv(csv_file)
                
                if 'market' not in df.columns or 'signal' not in df.columns:
                    continue
                
                # 타임스탬프 파싱
                if '_' in date_str:
                    d, t = date_str.split('_')
                    ts = datetime(
                        int(d[:4]), int(d[4:6]), int(d[6:8]),
                        int(t[:2]), int(t[2:4]), int(t[4:6])
                    )
                else:
                    ts = datetime(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))
                
                df['prediction_timestamp'] = ts
                all_dfs.append(df)
                
            except Exception as e:
                continue
        
        if not all_dfs:
            return pd.DataFrame()
        
        # 모든 데이터 병합
        combined = pd.concat(all_dfs, ignore_index=True)
        
        # 필요한 컬럼 정규화
        if 'current_price' not in combined.columns:
            combined['current_price'] = 0
        
        # Add entry_time alias for compatibility
        if 'prediction_timestamp' in combined.columns:
            combined['entry_time'] = combined['prediction_timestamp']
        
        self._predictions_df = combined
        return combined

    
    def _get_price_at_time(self, market: str, timestamp: datetime) -> Optional[float]:
        """특정 시점의 가격 조회"""
        try:
            query = f"""
                SELECT close FROM crypto_data 
                WHERE market = '{market}' 
                AND timestamp <= '{timestamp.strftime('%Y-%m-%dT%H:%M:%S')}'
                ORDER BY timestamp DESC LIMIT 1
            """
            result = load_data(query)
            if not result.empty:
                return float(result.iloc[0]['close'])
        except:
            pass
        return None
    
    def _get_btc_return(self, start_date: datetime, end_date: datetime) -> float:
        """BTC 벤치마크 수익률 계산"""
        try:
            start_price = self._get_price_at_time('KRW-BTC', start_date)
            end_price = self._get_price_at_time('KRW-BTC', end_date)
            if start_price and end_price:
                return (end_price - start_price) / start_price
        except:
            pass
        return 0.0
    
    def calculate_trade_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """각 거래의 실제 수익률 계산"""
        results = []
        
        for _, row in df.iterrows():
            market = row['market']
            signal = row['signal']
            entry_time = row['prediction_timestamp']
            entry_price = row['current_price']
            
            if pd.isna(entry_price) or entry_price <= 0:
                continue
            
            # 4시간 후 가격 조회
            exit_time = entry_time + timedelta(hours=self.HOLDING_PERIOD_HOURS)
            exit_price = self._get_price_at_time(market, exit_time)
            
            if exit_price is None:
                continue
            
            # 수익률 계산 (Short의 경우 반대)
            raw_return = (exit_price - entry_price) / entry_price
            if signal == 'Short':
                raw_return = -raw_return
            
            results.append({
                'entry_time': entry_time, # Frontend expects entry_time
                'prediction_timestamp': entry_time,
                'market': market,
                'signal': signal,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl_percent': raw_return * 100, # Frontend expects pnl_percent, and usually handled as % value (e.g. 5.0 for 5%)
                'status': 'CLOSED', # Explicitly mark as closed for frontend filter
                'is_win': raw_return > 0
            })
        
        return pd.DataFrame(results)
    
    def calculate_metrics(self, start_date: datetime = None, end_date: datetime = None) -> Dict:
        """7개 핵심 지표 계산 (포트폴리오 시뮬레이션 기반)"""
        df = self._load_predictions()
        
        if df.empty:
            return self._empty_metrics()
        
        # 날짜 필터링
        if start_date:
            df = df[df['prediction_timestamp'] >= start_date]
        if end_date:
            df = df[df['prediction_timestamp'] <= end_date]
        
        if df.empty:
            return self._empty_metrics()
        
        # 시뮬레이션 실행
        simulation_result = self._run_simulation(df)
        
        # 지표 계산
        equity_curve = simulation_result['equity_curve']
        trades = simulation_result['trades']
        
        if not equity_curve:
            return self._empty_metrics()
            
        initial_balance = equity_curve[0]['equity']
        final_balance = equity_curve[-1]['equity']
        
        # 1. Total Return
        total_return = (final_balance - initial_balance) / initial_balance
        
        # 2. Sharpe Ratio
        returns_series = pd.Series([x['equity'] for x in equity_curve]).pct_change().dropna()
        if len(returns_series) > 1 and returns_series.std() > 0:
            # 4시간 단위 데이터라고 가정 시 연율화
            sharpe_ratio = (returns_series.mean() / returns_series.std()) * self.ANNUALIZATION_FACTOR
        else:
            sharpe_ratio = 0.0
            
        # 3. Max Drawdown
        equity_series = pd.Series([x['equity'] for x in equity_curve])
        running_max = equity_series.cummax()
        drawdown = (equity_series - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # 4. Win Rate
        winning_trades = [t for t in trades if t['pnl_percent'] > 0]
        win_rate = len(winning_trades) / len(trades) if trades else 0
        
        # 5. Alpha (vs BTC)
        actual_start = df['prediction_timestamp'].min()
        actual_end = df['prediction_timestamp'].max()
        btc_return = self._get_btc_return(actual_start, actual_end)
        alpha = total_return - btc_return
        
        # 6. Trade Count
        trade_count = len(trades)
        
        # 7. Profit Factor
        gross_profit = sum(t['profit'] for t in trades if t['profit'] > 0)
        gross_loss = abs(sum(t['profit'] for t in trades if t['profit'] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return {
            'total_return': round(total_return * 100, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'max_drawdown': round(max_drawdown * 100, 2),
            'win_rate': round(win_rate * 100, 1),
            'alpha': round(alpha * 100, 2),
            'trade_count': trade_count,
            'profit_factor': round(profit_factor, 2),
            'period': {
                'start': actual_start.isoformat(),
                'end': actual_end.isoformat()
            },
            'holding_period': f'{self.HOLDING_PERIOD_HOURS}H',
            # Include Simulation Data for Charts
            'trades': trades, 
            'equity_curve': self._format_equity_for_chart(equity_curve, initial_balance)
        }

    def _format_equity_for_chart(self, equity_curve, initial_balance):
        if not equity_curve: return []
        # Sample down if too large, but for now return all
        return [{'date': x['timestamp'].strftime('%Y-%m-%d %H:%M'), 
                 'value': round(((x['equity'] - initial_balance)/initial_balance)*100, 2),
                 'equity': x['equity']} for x in equity_curve]

    def _run_simulation(self, df: pd.DataFrame) -> Dict:
        """시간 흐름에 따른 포트폴리오 시뮬레이션"""
        INITIAL_BALANCE = 10_000_000  # 1,000만원
        MAX_POSITIONS = 10            # 최대 동시 포지션 수
        pos_size_ratio = 1.0 / MAX_POSITIONS  # 포지션당 10%
        
        balance = INITIAL_BALANCE  # 현금 잔고
        positions = []  # 현재 보유 포지션 목록
        closed_trades = []  # 청산된 거래 기록
        equity_history = []  # 자산 가치 기록
        
        # 모든 이벤트를 시간순 정렬 (매수 시그널)
        df = df.sort_values('prediction_timestamp')
        
        # 시뮬레이션 시간 스텝 (매 시그널 발생 시점)
        for _, row in df.iterrows():
            current_time = row['prediction_timestamp']
            
            # 1. 청산 로직 (시간 경과한 포지션 정리)
            active_positions = []
            for pos in positions:
                if pos['exit_time'] <= current_time:
                    # 청산 실행
                    exit_price = self._get_price_at_time(pos['market'], pos['exit_time'])
                    if exit_price:
                        # 수익금 계산
                        if pos['signal'] == 'Long':
                            pnl_pct = (exit_price - pos['entry_price']) / pos['entry_price']
                        else:
                            pnl_pct = (pos['entry_price'] - exit_price) / pos['entry_price']
                        
                        revenue = pos['amount'] * (1 + pnl_pct)
                        profit = revenue - pos['amount']
                        balance += revenue
                        
                        closed_trades.append({
                            'market': pos['market'],
                            'entry_time': pos['entry_time'],
                            'exit_time': pos['exit_time'],
                            'entry_price': pos['entry_price'],
                            'exit_price': exit_price,
                            'profit': profit,
                            'pnl_percent': pnl_pct * 100, # Percentage
                            'status': 'CLOSED',
                            'signal': pos['signal'] # Useful for frontend
                        })
                    else:
                        # 가격 데이터 없으면 원금 회수 가정 (보수적 접근 or 에러 처리)
                        balance += pos['amount']
                else:
                    active_positions.append(pos)
            
            positions = active_positions
            
            # 2. 신규 진입 로직
            if len(positions) < MAX_POSITIONS:
                # 진입 가능한 자금 계산 (전체 자산의 1/N)
                # 현재 총 자산 가치 추정 = 현금 + 보유 포지션 가치
                current_equity = balance + sum(p['amount'] for p in positions)
                target_amount = current_equity * pos_size_ratio
                
                # 가용 현금이 충분한지 확인
                if balance >= target_amount and row['current_price'] > 0:
                    balance -= target_amount
                    entry_time = row['prediction_timestamp']
                    exit_time = entry_time + timedelta(hours=self.HOLDING_PERIOD_HOURS)
                    
                    positions.append({
                        'market': row['market'],
                        'signal': row['signal'],
                        'entry_price': row['current_price'],
                        'amount': target_amount,
                        'entry_time': entry_time,
                        'exit_time': exit_time
                    })
            
            # 3. 자산 가치 기록 (Equity Curve용)
            # 현재 시점의 평가 가치
            estimated_equity = balance + sum(p['amount'] for p in positions)
            equity_history.append({
                'timestamp': current_time,
                'equity': estimated_equity
            })
            
        return {
            'equity_curve': equity_history,
            'trades': closed_trades
        }

    def get_equity_curve(self, start_date: datetime = None, end_date: datetime = None) -> List[Dict]:
        """누적 수익 곡선 데이터 (시뮬레이션 결과 사용)"""
        metrics = self.calculate_metrics(start_date, end_date)
        if metrics['total_return'] == 0 and metrics['trade_count'] == 0:
            return []
            
        # calculate_metrics 내부에서 이미 시뮬레이션을 돌리지만,
        # 여기서는 그래프용 데이터를 위해 다시 돌리거나 캐싱해야 함.
        # 편의상 다시 실행 (비용 크지 않음)
        df = self._load_predictions()
        if start_date: df = df[df['prediction_timestamp'] >= start_date]
        if end_date: df = df[df['prediction_timestamp'] <= end_date]
        
        sim_result = self._run_simulation(df)
        equity_curve = sim_result['equity_curve']
        
        if not equity_curve: return []
        
        initial_equity = equity_curve[0]['equity']
        
        # 일별 데이터로 리샘플링 (그래프가 너무 조밀하지 않게)
        df_eq = pd.DataFrame(equity_curve)
        df_eq['date'] = df_eq['timestamp'].dt.date
        daily_equity = df_eq.groupby('date')['equity'].last().reset_index()
        
        return [
            {
                'date': row['date'].strftime('%Y-%m-%d'),
                'value': round(((row['equity'] - initial_equity) / initial_equity) * 100, 2)
            }
            for _, row in daily_equity.iterrows()
        ]
    
    def get_monthly_returns(self) -> List[Dict]:
        """월별 수익률 (시뮬레이션 기반)"""
        # 전체 데이터로 시뮬레이션 먼저 실행
        df = self._load_predictions()
        sim_result = self._run_simulation(df)
        equity_curve = sim_result['equity_curve']
        
        if not equity_curve: return []
        
        df_eq = pd.DataFrame(equity_curve)
        df_eq['month'] = df_eq['timestamp'].dt.to_period('M')
        
        # Trade Data Parsing
        trades = sim_result.get('trades', [])
        df_trades = pd.DataFrame(trades)
        if not df_trades.empty and 'exit_time' in df_trades.columns:
            df_trades['exit_time'] = pd.to_datetime(df_trades['exit_time'])
            df_trades['month'] = df_trades['exit_time'].dt.to_period('M')

        monthly_data = []
        months = sorted(df_eq['month'].unique())
        
        for m in months:
            month_data = df_eq[df_eq['month'] == m]
            if month_data.empty: continue
            
            start_equity = month_data.iloc[0]['equity']
            end_equity = month_data.iloc[-1]['equity']
            monthly_return = (end_equity - start_equity) / start_equity
            
            # Calculate Trade Stats
            count = 0
            win_rate = 0.0
            profit_factor = 0.0
            
            if not df_trades.empty:
                m_trades = df_trades[df_trades['month'] == m]
                count = len(m_trades)
                if count > 0:
                    wins = m_trades[m_trades['profit'] > 0]
                    losses = m_trades[m_trades['profit'] <= 0]
                    win_rate = (len(wins) / count) * 100
                    
                    gross_profit = wins['profit'].sum()
                    gross_loss = abs(losses['profit'].sum())
                    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            
            monthly_data.append({
                'month': str(m),
                'return': round(monthly_return * 100, 2),
                'win_rate': round(win_rate, 1), 
                'trade_count': count,
                'profit_factor': round(profit_factor, 2)
            })
            
        return monthly_data

    def _empty_metrics(self) -> Dict:
        """빈 지표 반환"""
        return {
            'total_return': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'win_rate': 0,
            'alpha': 0,
            'trade_count': 0,
            'profit_factor': 0,
            'period': {'start': None, 'end': None},
            'holding_period': f'{self.HOLDING_PERIOD_HOURS}H'
        }
    
    def get_metrics_for_period(self, period: str, month: str = None) -> Dict:
        """기간별 지표 조회"""
        now = datetime.now()
        
        if period == 'monthly' and month:
            # 특정 월 (예: 2026-01)
            year, mon = map(int, month.split('-'))
            start_date = datetime(year, mon, 1)
            if mon == 12:
                end_date = datetime(year + 1, 1, 1) - timedelta(seconds=1)
            else:
                end_date = datetime(year, mon + 1, 1) - timedelta(seconds=1)
        elif period == '90d':
            start_date = now - timedelta(days=90)
            end_date = now
        elif period == '7d':
            # This Week (Start from Monday 00:00)
            today = now.replace(hour=0, minute=0, second=0, microsecond=0)
            start_date = today - timedelta(days=today.weekday())
            end_date = now
        elif period == 'all':
            start_date = None
            end_date = None
        else:
            start_date = now - timedelta(days=30)
            end_date = now
        
        return self.calculate_metrics(start_date, end_date)


    def get_recent_signals(self, limit: int = 100) -> List[Dict]:
        """최근 신호 목록 조회 (홈 화면용) - 가장 최근 배치만 반환 (텔레그램 발송 기준)"""
        df = self._load_predictions()
        if df.empty: return []
        
        # 가장 최근 타임스탬프(배치)만 필터링
        # 같은 시간에 생성된 추천들 = 하나의 배치
        latest_timestamp = df['prediction_timestamp'].max()
        df = df[df['prediction_timestamp'] == latest_timestamp]
        
        # limit 적용
        df = df.head(limit)
        
        if df.empty: return []
        
        # 실시간 가격 조회 (배치로 한 번에)
        try:
            from utils.price_cache import get_prices_batch
            markets = df['market'].unique().tolist()
            current_prices = get_prices_batch(markets)
        except Exception as e:
            print(f"[BacktestAnalyzer] Price fetch error: {e}")
            current_prices = {}
        
        results = []
        for _, row in df.iterrows():
            entry_price = row.get('current_price', 0)
            market = row['market']
            
            # 실시간 가격 사용 (없으면 진입가로 대체)
            real_time_price = current_prices.get(market, entry_price if entry_price > 0 else 0)
            
            # PnL 계산
            actual_return = 0
            is_win = False
            if entry_price > 0 and real_time_price > 0:
                raw_return = (real_time_price - entry_price) / entry_price
                if row['signal'] == 'Short':
                    raw_return = -raw_return
                actual_return = raw_return
                is_win = raw_return > 0
            
            results.append({
                'market': market,
                'signal': row['signal'],
                'prediction_timestamp': row['prediction_timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'entry_price': entry_price,
                'current_price': real_time_price,
                'predicted_return': 0, 
                'actual_return': round(actual_return * 100, 2),
                'direction_correct': is_win,
                'position_size': 0.1,  
                'volatility': 0,
                'strategy_type': 'AI Model'
            })
            
        return results

    def get_daily_stats_history(self, limit: int = 30) -> Dict:
        """일별 통계 히스토리 (차트용)"""
        # 시뮬레이션 결과 기반
        df = self._load_predictions()
        sim_result = self._run_simulation(df)
        trades = sim_result['trades']
        
        if not trades:
            return {'dates': [], 'accuracy': [], 'counts': []}
            
        # DataFrame으로 변환하여 일별 집계
        df_trades = pd.DataFrame(trades)
        df_trades['date'] = df_trades['entry_time'].dt.strftime('%Y-%m-%d')
        
        # 날짜별 그룹화
        daily = df_trades.groupby('date').agg(
            count=('profit', 'count'),
            wins=('profit', lambda x: (x > 0).sum())
        ).reset_index()
        
        # 최근 N일
        daily = daily.sort_values('date').tail(limit)
        
        return {
            'dates': daily['date'].tolist(),
            'accuracy': (daily['wins'] / daily['count'] * 100).round(1).tolist(),
            'counts': daily['count'].tolist()
        }




# 전역 인스턴스
backtest_analyzer = BacktestAnalyzer()
