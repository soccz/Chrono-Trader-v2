"""
Recalibrate Ensemble Weights - REAL DATA VERSION
실제 예측 기록(recommendations/*.csv)과 현재 시장 가격을 비교하여
각 모델의 방향 예측 정확도를 측정하고, ModelPerformanceTracker를 업데이트합니다.

사용법:
    python3 scripts/recalibrate_ensemble.py
"""
import sys
import os
import glob
import pandas as pd
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_tracker import get_tracker
from utils.logger import logger


def get_actual_price(market: str) -> float:
    """Upbit API에서 현재 시장 가격을 조회합니다."""
    try:
        from data.collector import get_current_price
        price = get_current_price(market)
        return float(price) if price else 0.0
    except Exception as e:
        logger.warning(f"가격 조회 실패 ({market}): {e}")
        return 0.0


def evaluate_recommendations(rec_dir: str = "recommendations", lookback_days: int = 7) -> list:
    """
    최근 N일간의 추천 기록을 읽어서 예측 방향이 맞았는지 평가합니다.
    
    Returns:
        List of (model_id, was_correct) tuples
    """
    rec_files = sorted(glob.glob(os.path.join(rec_dir, "recs_*.csv")), reverse=True)
    
    if not rec_files:
        logger.warning("추천 기록 파일이 없습니다. 보정 불가.")
        return []
    
    cutoff_date = datetime.now() - timedelta(days=lookback_days)
    results = []
    evaluated_count = 0
    
    # Strategy → Model ID 매핑
    strategy_to_model = {
        'trending': 0,       # Model 1: Trend Following
        'mean_reversion': 1, # Model 2: Mean Reversion
        'continuous': 2,     # Model 3: Volatility Breakout
        'pattern': 3,        # Model 4: Pattern Recognition
        'daily': 4,          # Model 5: Market Neutral / General
    }
    
    for fpath in rec_files:
        try:
            # 파일명에서 타임스탬프 추출
            basename = os.path.basename(fpath)
            # Format: recs_daily_pattern_20260129_034707.csv or recs_YYYYMMDD_HHMMSS.csv
            parts = basename.replace("recs_", "").replace(".csv", "")
            
            # 날짜 부분 추출 (마지막 두 세그먼트가 YYYYMMDD_HHMMSS)
            segments = parts.split("_")
            if len(segments) >= 2:
                date_str = segments[-2]  # YYYYMMDD
                time_str = segments[-1]  # HHMMSS
                try:
                    rec_time = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
                except ValueError:
                    continue
            else:
                continue
            
            # 최근 데이터만 평가
            if rec_time < cutoff_date:
                continue
            
            # CSV 읽기
            df = pd.read_csv(fpath)
            if df.empty:
                continue
            
            # strategy 컬럼에서 모델 ID 결정
            strategy_col = df.get('strategy', pd.Series(['daily'] * len(df)))
            
            for idx, row in df.iterrows():
                market = row.get('market', '')
                signal = row.get('signal', 'Long')
                entry_price = float(row.get('current_price', 0))
                strategy = str(strategy_col.iloc[idx] if idx < len(strategy_col) else 'daily')
                
                if entry_price <= 0 or not market:
                    continue
                
                # 현재 가격 조회
                current_price = get_actual_price(market)
                if current_price <= 0:
                    continue
                
                # 방향 정확도 판정
                actual_return = (current_price - entry_price) / entry_price
                if signal == 'Short':
                    actual_return = -actual_return
                
                was_correct = actual_return > 0  # 예측 방향과 실제 방향 일치?
                
                # 모델 ID 결정
                # 파일명에서 strategy 추출 시도
                for key in strategy_to_model:
                    if key in basename.lower() or key in strategy.lower():
                        model_id = strategy_to_model[key]
                        break
                else:
                    model_id = 4  # 기본: Model 5
                
                results.append((model_id, was_correct))
                evaluated_count += 1
                
        except Exception as e:
            logger.warning(f"파일 평가 실패 ({fpath}): {e}")
            continue
    
    logger.info(f"총 {evaluated_count}개 추천 평가 완료.")
    return results


def run():
    print("=" * 50)
    print("앙상블 가중치 재보정 (실제 데이터 기반)")
    print("=" * 50)
    
    # 1. 트래커 초기화 (기존 데이터 유지, reset 하지 않음!)
    from utils.config import config
    tracker = get_tracker(n_models=config.Gan.N_ENSEMBLE_MODELS)
    
    # 2. 실제 추천 기록 평가
    model_results = evaluate_recommendations(lookback_days=7)
    
    if not model_results:
        print("⚠️  평가할 추천 데이터가 없습니다.")
        print("   main.py --mode daily 를 먼저 실행하여 추천을 생성하세요.")
        return
    
    # 3. 트래커에 결과 반영
    tracker.update_batch(model_results)
    
    # 4. 새 가중치 계산 및 출력
    weights = tracker.get_weights()
    stats = tracker.get_stats()
    
    model_names = [
        "Scalper (단기 모멘텀)",
        "Swing Trader (패턴 전이)",
        "Trend Follower (추세 추종)",
        "Regime Sentinel (변동성/꼬리 리스크)"
    ]

    print("\n📊 모델별 실제 성과:")
    print("-" * 60)
    for i in range(config.Gan.N_ENSEMBLE_MODELS):
        model_stat = stats.get(f'model_{i}', {})
        n_samples = model_stat.get('n_samples', 0)
        accuracy = model_stat.get('accuracy', 0.0)
        weight = weights[i]
        
        bar = "█" * int(accuracy * 20) + "░" * (20 - int(accuracy * 20))
        print(f"  Model {i+1} ({model_names[i]})")
        print(f"    정확도: {accuracy:.1%} [{bar}]")
        print(f"    샘플수: {n_samples}개 | 가중치: {weight:.3f}")
    
    print("-" * 60)
    print(f"  합계 가중치: {weights.sum():.3f}")
    
    # 5. 저장
    tracker.save()
    print("\n✅ model_performance.json 업데이트 완료.")


if __name__ == "__main__":
    run()
