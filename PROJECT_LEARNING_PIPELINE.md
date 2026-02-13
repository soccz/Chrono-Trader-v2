# 학습 파이프라인 정의서 (Learning Pipeline Contract)

> 이 문서는 프로젝트의 **학습-추론-평가-재보정** 사이클을 정의합니다.
> 모든 AI 어시스턴트는 이 문서를 숙지하고, 아래 규칙을 위반하는 코드를 작성해서는 안 됩니다.

---

## 1. 시스템 아키텍처

> **상세 정의서**: [PROJECT_ARCHITECTURE.md](./PROJECT_ARCHITECTURE.md) 참조

```
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│  데이터 수집   │──➤│   모델 학습    │──➤│    추론/예측    │
│  (collector)  │      │  (trainer)    │      │  (predictor)  │
└──────────────┘      └──────────────┘      └──────────────┘
       │                      │                      │
       │               ┌──────▼──────┐               ▼
       │               │  XGBoost    │      ┌──────────────┐
       └──────────────➤│ Pump Train  │──➤│  펌프 탐지     │
                       └─────────────┘      └──────┬───────┘
                                                   │
┌──────────────┐      ┌──────────────┐      ┌──────▼───────┐
│  가중치 갱신   │◀──│   성과 평가    │◀──│  추천/실행     │
│  (recalib)   │      │ (validator)   │      │ (recommender) │
└──────────────┘      └──────────────┘      └──────────────┘
```

**핵심**: 모든 단계가 **실제 데이터**로 동작. 난수(random) 사용 절대 금지.

---

## 2. 데이터 계약 (Interface Contracts)

### 2.1 공유 저장소 (Shared Data Stores)

| 저장소 | 경로 | 포맷 | 용도 |
|:---|:---|:---|:---|
| **portfolio.db** | `data/portfolio.db` | SQLite | 거래 기록 (OPEN/CLOSED) |
| **model_performance.json** | `models/model_performance.json` | JSON | 모델별 rolling accuracy |
| **all_predictions.csv** | `all_predictions.csv` | CSV | 전체 예측 기록 |
| **recommendations/** | `recommendations/recs_*.csv` | CSV | 일별 추천 기록 |
| **gate_values.csv** | `analysis/gate_values.csv` | CSV | Gate 비중 히스토리 |
| **model_*.pth** | `models/model_*.pth` | PyTorch | 학습된 모델 가중치 |

### 2.2 DB 스키마 (trades 테이블)

```sql
CREATE TABLE trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    market TEXT NOT NULL,        -- ex: KRW-BTC
    strategy TEXT NOT NULL,      -- ex: daily, continuous, pattern
    signal TEXT NOT NULL,        -- Long / Short
    entry_price REAL NOT NULL,
    entry_time TIMESTAMP NOT NULL,
    exit_price REAL,
    exit_time TIMESTAMP,
    status TEXT DEFAULT 'OPEN',  -- OPEN / CLOSED
    pnl_percent REAL,
    position_value REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 2.3 recommendations CSV 컬럼

```
market, signal, strategy, expected_return, confidence, position_size,
volatility, dtw_distance, current_price, pattern, reason
```

---

## 3. 절대 규칙 (Iron Rules)

### 🔴 금지 사항
1. `random.shuffle()`, `np.random.normal()` 등으로 **가짜 성과 데이터 생성 금지**
2. `utils/` 디렉토리의 **함수 시그니처(입출력 형식) 임의 변경 금지**
3. 웹 UI 수정 시 `training/`, `inference/`, `utils/model_tracker.py` **건드리기 금지**
4. 학습 코드 수정 시 `templates/`, `static/` **건드리기 금지**

### 🟢 필수 사항
1. `recalibrate_ensemble.py`는 반드시 **실제 예측 vs 실제 가격** 비교 후 가중치 산출
2. `recalibrate_gate.py`는 반드시 **실제 시장 지표(ATR, 변동성)** 기반으로 Gate 값 결정
3. 모든 학습/평가 결과는 `models/model_performance.json`에 **누적 저장**
4. `main.py --mode daily`의 Step 0 (성과 검증)이 `ModelPerformanceTracker.update()`를 **반드시 호출**

---

## 4. 운영 흐름 (Ops Cycle)

> 운영(스케줄) 모드는 `refresh-db`(수집)와 `intraday`/`morning-report`(추론)를 분리한다.  
> **권장 엔트리포인트**: `scripts/run_scheduled.py`  
> **운영 계약**: `OPS_ACCEPTANCE.md`

```
매일 08:00 (KST)  morning-report
  refresh-db  →  morning-report  →  recommendations/*.csv + analysis/run_markets_metrics_morning.*

4시간마다         intraday
  refresh-db  →  intraday        →  recommendations/*.csv + analysis/run_markets_metrics_intraday.*
```

헬스체크/리포트:
- `scripts/ops_healthcheck.py` (15분마다, 누락/스테일/추천수 경보)
- `scripts/ops_soak_report.py` (최근 N시간 운영 요약)

## 4.1 (선택) Daily Cycle (`daily`)

`daily`는 경량 파인튜닝/리포트용 모드이며, 운영 스케줄의 기본은 `intraday`/`morning-report`다.

```
main.py --mode daily
  ├─ Step 0: 어제 추천 성과 검증 → ModelPerformanceTracker.update()
  ├─ Step 1: Trending 전략 (스크리닝 → 학습 → 예측 → 추천)
  ├─ Step 2: Pattern 전략 (리더 코인 → 팔로워 탐색 → 예측)
  ├─ Step 3: Pump 전략 (XGBoost 급등 감지 모델)
  └─ Step 4: 포트폴리오 기록 + 텔레그램 리포트
```

---

## 4.5 모델 내부 구조 (요약)

### GAN Hybrid Ensemble (5 모델)
- **입력**: 168시간 × 19피처 (MinMaxScaler)
- **경로**: Transformer(168h 어텐션) + CNN(1D/2D) → GatedFusion → GAN Decoder
- **학습 Loss 5종**: WGAN-GP, Reconstruction(MSE), ECE, Direction Balance, Gate Reg
- **동적 조정**: λ_recon, λ_gp 에폭마다 자동 조정
- **Auto-Stop**: grad_norm/ratio 이상 200 steps 지속 시 중단

### XGBoost Pump Classifier
- **입력**: 17 피처 (alpha/beta 미포함, volume_spike_score/squeeze_on/roc 포함)
- **출력**: 4-class 급등 확률
- **임계값**: total_pump_prob > 0.2

### 추론 파이프라인
- **MC-Dropout**: 5모델 × 20회 = 100 predictions
- **Shrunk Beta**: 코인별 β를 cross-sectional mean 방향으로 수축
- **가중 투표**: ModelPerformanceTracker의 rolling accuracy 비례
- **추천 퍼널 6단계**: Tradeable → 방향 일관 → 체제/Lead-Lag → 유동성 → 수익률 → 불확실성 → DTW

> 전체 수치/상수는 [PROJECT_ARCHITECTURE.md](./PROJECT_ARCHITECTURE.md) 참조

---

## 5. 웹/학습 분리 원칙

```
app.py (웹)                          main.py (학습)
    │                                     │
    │  ← "읽기만 한다" ←                   │
    │                                     │
    ├─ portfolio.db (읽기)               ├─ portfolio.db (쓰기)
    ├─ model_performance.json (읽기)     ├─ model_performance.json (쓰기)
    ├─ recommendations/*.csv (읽기)      ├─ recommendations/*.csv (쓰기)
    └─ analysis/gate_values.csv (읽기)   └─ analysis/gate_values.csv (쓰기)
```

**웹은 "뷰어", 학습은 "엔진"**. 서로의 코드를 수정하지 않는다.
