"""
Recalibrate Gate Values - REAL DATA VERSION
실제 시장 데이터(변동성, 추세 강도)를 기반으로 Gate 값을 계산합니다.
Gate 값 = Transformer(추세) vs CNN(패턴) 비중을 결정하는 핵심 지표.

- Gate > 0.5: Transformer(추세) 우세 → 강한 추세 시장
- Gate < 0.5: CNN(패턴) 우세 → 레인지/변동성 시장

사용법:
    python3 scripts/recalibrate_gate.py
"""
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import logger


def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Average True Range 계산 (변동성 지표)"""
    if len(df) < period + 1:
        return 0.0
    
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
    )
    
    atr = np.mean(tr[-period:])
    return float(atr)


def calculate_adx(df: pd.DataFrame, period: int = 14) -> float:
    """Average Directional Index 계산 (추세 강도 지표)"""
    if len(df) < period * 2:
        return 0.0
    
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    # +DM, -DM
    plus_dm = np.maximum(high[1:] - high[:-1], 0)
    minus_dm = np.maximum(low[:-1] - low[1:], 0)
    
    # Where +DM > -DM, keep +DM, else 0
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm < plus_dm] = 0
    
    # True Range
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
    )
    
    # Smooth with EMA
    def ema(data, period):
        alpha = 2.0 / (period + 1)
        result = np.zeros_like(data, dtype=float)
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_smooth = ema(tr, period)
    plus_di = 100 * ema(plus_dm, period) / np.where(atr_smooth > 0, atr_smooth, 1)
    minus_di = 100 * ema(minus_dm, period) / np.where(atr_smooth > 0, atr_smooth, 1)
    
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) > 0, (plus_di + minus_di), 1)
    adx = np.mean(dx[-period:])
    
    return float(np.clip(adx, 0, 100))


def load_market_data(market: str = "KRW-BTC", hours: int = 168) -> pd.DataFrame:
    """DB에서 시장 OHLCV 데이터 로드"""
    try:
        from data.database import load_data
        
        cutoff = (datetime.now() - timedelta(hours=hours)).strftime('%Y-%m-%d %H:%M:%S')
        query = f"""
            SELECT timestamp, open, high, low, close, volume 
            FROM crypto_data 
            WHERE market = '{market}' AND timestamp >= '{cutoff}'
            ORDER BY timestamp ASC
        """
        df = load_data(query)
        return df if df is not None and not df.empty else pd.DataFrame()
    except Exception as e:
        logger.warning(f"시장 데이터 로드 실패 ({market}): {e}")
        return pd.DataFrame()


def compute_gate_value(df: pd.DataFrame) -> float:
    """
    시장 데이터를 기반으로 Gate 값을 계산합니다.
    
    로직:
    - ADX > 25: 강한 추세 → Gate 높음 (Transformer 우세)
    - ADX < 15: 약한 추세 → Gate 낮음 (CNN/패턴 우세)
    - ATR 기반 변동성 보정
    
    Returns:
        Gate value (0.1 ~ 0.9)
    """
    if df.empty or len(df) < 30:
        return 0.5  # 데이터 부족 시 중립
    
    adx = calculate_adx(df)
    atr = calculate_atr(df)
    
    # 최근 가격 대비 ATR 비율 (상대 변동성)
    current_price = df['close'].iloc[-1]
    relative_vol = (atr / current_price) if current_price > 0 else 0
    
    # ADX → Gate 기본값 (선형 변환: ADX 10~40 → Gate 0.3~0.7)
    gate_base = np.clip((adx - 10) / 30, 0, 1) * 0.4 + 0.3
    
    # 변동성 보정: 높은 변동성 → CNN 쪽으로 약간 이동
    vol_adjustment = -0.1 if relative_vol > 0.03 else (0.05 if relative_vol < 0.01 else 0)
    
    gate_value = np.clip(gate_base + vol_adjustment, 0.1, 0.9)
    
    return float(gate_value)


def run():
    print("=" * 50)
    print("Gate 값 재보정 (실제 시장 데이터 기반)")
    print("=" * 50)
    
    # 분석 디렉토리 생성
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    # 대표 시장: config → DB에서 최근 데이터가 있는 시장으로 폴백
    try:
        from utils.config import config
        markets = getattr(config.Data, 'MARKET_INDEX_COINS', ['KRW-BTC'])
        if not markets:
            markets = ['KRW-BTC']
    except Exception:
        markets = ['KRW-BTC']
    
    # DB에서 최근 7일 내 데이터가 있는 시장 추가 (폴백)
    try:
        from data.database import load_data
        from datetime import datetime, timedelta
        cutoff = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d %H:%M:%S')
        recent_df = load_data(f"""
            SELECT market, COUNT(*) as cnt 
            FROM crypto_data 
            WHERE timestamp >= '{cutoff}' 
            GROUP BY market 
            HAVING cnt >= 24 
            ORDER BY cnt DESC 
            LIMIT 10
        """)
        if recent_df is not None and not recent_df.empty:
            db_markets = recent_df['market'].tolist()
            # config 시장 중 데이터 있는 것 우선, 나머지 DB에서 보충
            valid_config = [m for m in markets if m in db_markets]
            extra = [m for m in db_markets if m not in markets]
            markets = (valid_config + extra)[:5]
            if not valid_config:
                print(f"  ⚠️ Config 시장({', '.join(getattr(config.Data, 'MARKET_INDEX_COINS', []))})에 최근 데이터 없음. DB에서 대체 시장 사용.")
    except Exception as e:
        logger.warning(f"DB 시장 조회 실패: {e}")
    
    all_records = []
    
    for market in markets[:5]:  # 최대 5개 시장만 분석
        print(f"\n📈 {market} 분석 중...")
        df = load_market_data(market, hours=168)  # 7일
        
        if df.empty:
            print(f"  ⚠️ 데이터 없음, 건너뜀")
            continue
        
        adx = calculate_adx(df)
        atr = calculate_atr(df)
        gate = compute_gate_value(df)
        
        current_price = df['close'].iloc[-1]
        rel_vol = (atr / current_price * 100) if current_price > 0 else 0
        
        print(f"  ADX: {adx:.1f} | ATR: {atr:.2f} ({rel_vol:.2f}%)")
        print(f"  Gate Value: {gate:.3f}", end="")
        if gate > 0.6:
            print(" → 🔵 Transformer(추세) 우세")
        elif gate < 0.4:
            print(" → 🔴 CNN(패턴) 우세")
        else:
            print(" → 🟡 균형")
        
        all_records.append({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'market': market,
            'adx': round(adx, 2),
            'atr': round(atr, 4),
            'relative_volatility': round(rel_vol, 4),
            'gate_value': round(gate, 4)
        })
    
    if not all_records:
        print("\n⚠️  분석 가능한 시장 데이터가 없습니다.")
        return
    
    # 종합 Gate 값 (시장별 평균)
    avg_gate = np.mean([r['gate_value'] for r in all_records])
    
    # 기존 gate_values.csv에 append (히스토리 유지)
    file_path = os.path.join(output_dir, 'gate_values.csv')
    new_df = pd.DataFrame(all_records)
    
    if os.path.exists(file_path):
        try:
            existing = pd.read_csv(file_path)
            combined = pd.concat([existing, new_df], ignore_index=True)
            # 최근 500 레코드만 유지
            combined = combined.tail(500)
            combined.to_csv(file_path, index=False)
        except Exception:
            new_df.to_csv(file_path, index=False)
    else:
        new_df.to_csv(file_path, index=False)
    
    print(f"\n{'=' * 50}")
    print(f"📊 종합 Gate Value: {avg_gate:.3f}")
    print(f"   저장 위치: {file_path}")
    print(f"   기록 수: {len(all_records)}개 시장")
    print(f"{'=' * 50}")
    print("✅ Gate 재보정 완료.")


if __name__ == "__main__":
    run()
