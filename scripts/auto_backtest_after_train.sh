#!/bin/bash
# 학습 완료 대기 → 자동 백테스트
cd /mnt/20t/main/gan_t
source .venv_local/bin/activate

# PID 2439918 (학습) 완료 대기
while kill -0 2439918 2>/dev/null; do
    sleep 60
done

echo "$(date) Training finished. Starting 90-day backtest..."

# 학습 에러 체크
if grep -q "Traceback\|FATAL" training_final.log; then
    echo "$(date) ERROR: Training had errors. Skipping backtest."
    exit 1
fi

# 90일 백테스트 실행
python main.py --mode backtest --days 90 > backtest_final_90d.log 2>&1
echo "$(date) Backtest complete."

# 결과 요약 출력
python3 -c "
import json, glob
files = sorted(glob.glob('analysis/backtest_summary_main_*.json'))
if files:
    with open(files[-1]) as f:
        d = json.load(f)
    print('=== BACKTEST RESULT ===')
    for k in ['n_trades','win_rate_pct','avg_return_per_trade','sharpe_annualized','max_drawdown_pct','ece']:
        print(f'{k}: {d.get(k)}')
    print(f'trades_by_signal: {d.get(\"trades_by_signal\")}')
    print(f'pi_coverage_80: {d.get(\"pi_coverage_80\",{}).get(\"empirical_coverage_80\")}')
"
