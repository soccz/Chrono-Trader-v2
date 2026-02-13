# 🏗 Chrono-Trader v2 — 전체 시스템 아키텍처 정의서

> 이 문서는 Chrono-Trader v2의 **모든 구성 요소**를 정의합니다.
> 코드 수정 시 반드시 이 문서를 참조하고, 변경 후 동기화해야 합니다.
> **최종 검증일**: 2026-02-12 (소스코드 전수 재검증 — 40+ 파일)

---

## 목차

1. [시스템 전체 구조](#1-시스템-전체-구조)
2. [설계 의도 (Why?)](#2-설계-의도-why)
3. [GAN Hybrid Ensemble (AI 모델)](#3-gan-hybrid-ensemble-ai-모델)
4. [학습 프로세스](#4-학습-프로세스)
5. [추론 파이프라인](#5-추론-파이프라인)
6. [추천 퍼널](#6-추천-퍼널-6단계)
7. [XGBoost 펌프 분류기](#7-xgboost-펌프-분류기-system-2)
8. [main.py 모드 카탈로그](#8-mainpy-모드-카탈로그)
9. [웹 대시보드](#9-웹-대시보드)
10. [텔레그램 알림 시스템](#10-텔레그램-알림-시스템)
11. [포트폴리오 관리](#11-포트폴리오-관리)
12. [Research Lab](#12-research-lab)
13. [데이터 계층](#13-데이터-계층)
14. [자동화 인프라](#14-자동화-인프라)
15. [백테스트 & 분석 도구](#15-백테스트--분석-도구)

---

## 1. 시스템 전체 구조

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Upbit API (데이터 소스)                       │
│                       전체 KRW 코인 (100+개)                        │
└──────────┬─────────────────────────────────┬───────────────────────┘
           │                                 │
   ┌───────▼───────┐                ┌────────▼────────┐
   │  Data Collector│                │   Price Cache    │
   │  (collector.py)│                │ (price_cache.py) │
   │  → SQLite DB   │                │ → TTL 캐시       │
   └───────┬───────┘                └────────┬────────┘
           │                                 │
   ┌───────▼───────┐                         │
   │   Preprocessor │                         │
   │ 19피처 + 시퀀스 │                         │
   └──┬─────────┬──┘                         │
      │         │                            │
┌─────▼───┐ ┌──▼──────┐                     │
│GAN Hybrid│ │XGBoost  │                     │
│Ensemble  │ │Pump Det │                     │
│ (5모델)  │ │ (1모델) │                     │
└─────┬───┘ └──┬──────┘                     │
      │        │                            │
┌─────▼───┐ ┌──▼──────┐                     │
│Predictor │ │PumpPred │                     │
│MC-Dropout│ │전체 KRW │                     │
└─────┬───┘ └──┬──────┘                     │
      │        │                            │
┌─────▼────────▼──┐     ┌──────────────────▼───────────────────┐
│ Recommender      │     │         Flask Web Dashboard           │
│ 6단계 필터링      │     │  7 pages, 80+ API, SocketIO           │
└─────┬───────────┘     └──────────────────┬───────────────────┘
      │                                    │
┌─────▼───────────┐     ┌──────────────────▼───────────────────┐
│ Telegram Bot     │     │       Cloudflare Tunnel               │
│ 알림 + 리포트     │     │       외부 접속                        │
└─────────────────┘     └──────────────────────────────────────┘
```

### 1.1 데이터 흐름

> [!IMPORTANT]
> 이 시스템은 **전체 시장 스캐너**입니다. 업비트 KRW 전 종목(100+개)을 수집·스캔합니다.

| 단계 | 대상 범위 | 코드 |
|------|----------|------|
| **데이터 수집** | **전체 KRW 코인** (Upbit API) | `collector.run_all()` |
| **스크리닝** | **전체 KRW** → Top 5 트렌딩 | `screener.get_trending_markets()` |
| **Full Train** | BTC, ETH (`MARKET_INDEX_COINS`) | `trainer.run()` |
| **Daily Fine-tune** | 트렌딩 코인 5개 | `trainer.run(markets=trending)` |
| **Trending 예측** | 트렌딩 코인 | `predictor.run(trending)` |
| **Pattern 예측** | DB 전체 KRW → 팔로워 탐색 | `find_pattern_followers()` |
| **Pump 탐지** | DB 전체 KRW 스캔 | `pump_predictor.run()` |

---

## 2. 설계 의도 (Why?)

> 이 섹션은 아키텍처의 "왜?"를 설명합니다. 누구든 이 시스템의 설계 근거를 파악할 수 있도록.

### 2.1 왜 1시간 봉인가?

- **분봉(1분, 5분)**: 노이즈 과다, 데이터 저장/처리 비용 폭증
- **일봉**: 6시간 예측이 불가능할 정도로 해상도가 낮음
- **1시간 봉**: 추세도 보이고 단기 패턴도 잡히는 **최적 균형점**
- Upbit API가 1시간 캔들을 안정적으로 제공

### 2.2 왜 168시간(7일) 입력인가?

- `SEQUENCE_LENGTH = 168` = 정확히 **1주일**
- 암호화폐는 주말 구분 없음 → 7일 = 168시간이 **하나의 완전한 주기**
- 1~2일의 단기 모멘텀 + 5~7일의 중기 추세가 모두 포함됨
- 30일은 오래된 정보가 noise, 3일은 맥락 부족

### 2.3 왜 6시간 예측인가?

- `FUTURE_WINDOW_SIZE = 6` = 향후 6개 시점의 가격 변화율
- **단일 값이 아니라 "패턴"을 예측**하는 것이 핵심
  - 예: `[+0.2%, +0.5%, +0.8%, +1.0%, +0.7%, +0.3%]` → "상승 후 꺾이는 패턴"
- 단순 방향뿐 아니라 **타이밍**까지 추론 가능
- 6시간 = 스캘핑하기엔 길고 스윙하기엔 짧은, **데이 트레이딩 최적 구간**

### 2.4 왜 GAN인가?

| 방식 | 문제 |
|------|------|
| MLP/LSTM 회귀 | **평균값으로 수렴** → 뚜렷한 방향 예측 불가 |
| Classification | 방향만 나옴, 크기와 타이밍 정보 없음 |
| **GAN** | Generator가 **"실현 가능한 미래 패턴"** 생성 |

- Critic이 "이 패턴이 진짜인지 가짜인지" 판별
- Generator는 Critic을 속이면서도 실제 가격에 가까운 패턴을 만들어야 함
- → **평균이 아닌, 구체적으로 실현 가능한 시나리오** 출력
- WGAN-GP 사용: 일반 GAN 대비 학습 안정성 + Gradient Penalty로 붕괴 방지

### 2.5 왜 Transformer + CNN 하이브리드인가?

| 컴포넌트 | 역할 | 없으면? |
|----------|------|---------|
| **Transformer** | 168시간 전체 장기 의존성 포착 ("3일 전 급등의 영향") | 먼 과거 정보 무시 |
| **CNN (TCN)** | dilated conv로 국소 패턴 포착 ("직전 수 시간 형태") | 미세 형태 놓침 |
| **Gate Fusion** | 지금 상황에 뭐가 중요한지 자동 판단 | 항상 고정 비율 → 장세변화 대응 불가 |

- 추세장 → Transformer 가중 ↑ (gate > 0.6)
- 횡보/패턴장 → CNN 가중 ↑ (gate < 0.4)
- Gate는 learned prototypes와의 유사도로 결정 → **해석 가능(Explainable)**

### 2.6 왜 앙상블 5개인가?

- 단일 모델은 **특정 장세에서만 잘 맞고 다른 장세에서 실패**
- 5개 모델이 각각 다른 관점:
  - Model 1: 얕은 구조 → 단기 패턴 전문
  - Model 2: 깊은 구조 → 장기 추세 전문
  - Model 4: **2D CNN (GAF)** → 시계열을 이미지로 변환하는 독특한 시각
- MC-Dropout 20회 × 5모델 = 100개 예측 → **"모델이 얼마나 확신하는지"** 측정 가능
- 실적 기반 가중치로 잘하는 모델 의견이 더 반영됨

### 2.7 학습 데이터 vs 추론 데이터

> [!IMPORTANT]
> 추론 입력은 최근 7일이지만, 학습은 수개월 데이터로 수천 개 샘플을 만듭니다.

```
학습: 90~365일 데이터 → 슬라이딩 윈도우 → 수천 개 (168h→6h) 샘플
추론: 최근 168시간(7일) 1개 → 모델 → 6시간 예측 1개
```

| 구분 | 데이터 기간 | 샘플 수 | 대상 코인 |
|------|-----------|---------|----------|
| **Full Train** | 90~365일 | 수천 개 | BTC, ETH |
| **Daily Fine-tune** | 30일 | 수백 개 | 트렌딩 코인 5개 |
| **추론 (1회)** | 최근 7일 | **1개** | 대상 코인 |

### 2.8 인코더-디코더 구조 요약

```
[Transformer Encoder] ─┐
                       ├→ [Gate Fusion] → context vector (d_model차원)
[CNN Encoder] ─────────┘
                                         ↓
                              context + noise(32차원)
                                         ↓
                              [GAN Decoder (Generator)] → 6시간 예측 패턴
                                         ↓
                              [Critic (Discriminator)] → 진짜/가짜 판별 (학습시만)
```

- **인코더** = Transformer + CNN + Gate → 168h 입력을 하나의 context vector로 압축
- **디코더** = GAN Generator → context에 noise를 섞어 다양한 시나리오 생성
- **Critic** = 학습 때만 사용, 추론 시 버림
- noise를 섞는 이유: 같은 상황에서도 다양한 시나리오 생성 → 불확실성 측정

---

## 3. GAN Hybrid Ensemble (AI 모델)

### 3.1 입력 피처 (27개)

> **참조**: `config.Data.FEATURE_COLUMNS`

| # | 피처 | 그룹 | 존재 이유 |
|---|------|------|----------|
| 0 | `close` | 가격 | 가격 수준 |
| 1 | `volume` | 거래량 | 거래 활동 강도 |
| 2 | `rsi` | 모멘텀 | 과매수/과매도 |
| 3-5 | `macd`, `macdsignal`, `macdhist` | 추세 | 추세 강도와 방향 전환 |
| 6 | `adx` | 추세 | 추세 유무 판단 (강할수록 방향 확실) |
| 7 | `obv` | 거래량 | 거래량-가격 방향 일치 확인 |
| **8** | **`market_index_return`** | 매크로 | **시장 전체 vs 개별 코인 구분** |
| **9** | **`historical_similarity`** | RAG | **과거 유사 패턴 존재 여부** |
| 10-12 | `bb_upper/mid/lower` | 변동성 | Bollinger Band 위치 |
| 13 | `volume_ma` | 기준선 | 거래량 정상 수준 |
| 14-15 | `volatility_24h/7d` | 변동성 | 단기·중기 리스크 |
| 16 | `volume_volatility` | 변동성 | 거래량 스파이크 감지 |
| 17 | `alpha` | CAPM | 시장 대비 초과수익률 |
| 18 | `beta` | CAPM | 시장 민감도 (β>1 = 더 크게 움직임) |
| 19 | `price_position` | 위치 | 20일 범위 내 가격 위치 (0=저점, 1=고점) |
| 20 | `volume_ratio` | 거래량 | 현재 거래량 / 20일 평균 (폭발 감지) |
| 21 | `return_skew_24h` | 통계 | 24h 수익률 비대칭 → 급등/급락 전조 |
| 22 | `cross_corr_btc` | 상관 | BTC와 실시간 상관관계 변화 (rolling 24h) |
| **23** | **`factor_size`** | **FF팩터** | **SIZE: 소형 vs 대형 코인 수익률 차이** |
| **24** | **`factor_mom`** | **FF팩터** | **MOM: 7일 모멘텀 승자 vs 패자** |
| **25** | **`factor_vol`** | **FF팩터** | **VOL: 저변동 vs 고변동 코인 수익률** |
| **26** | **`factor_liq`** | **FF팩터** | **LIQ: 고유동 vs 저유동 코인 수익률** |

**핵심 차별점** — 피처 8, 9, 17, 18, 23-26:
- `market_index_return` = BTC+ETH 가중 지수 → 시장 전체 상승인지, 이 코인만 오르는지 구분
- `historical_similarity` = 과거 패턴 검색(RAG) → Transformer의 Positional Encoding에 주입
- `alpha/beta` = CAPM 기반 → 시장 대비 독립적 수익/민감도
- `factor_*` = FF 크립토 팩터 → 시장 미시 구조 (소형주 효과, 모멘텀, 변동성, 유동성 프리미엄)

**스케일링**: `MinMaxScaler` (0~1), **시퀀스**: 168h, **출력**: 6h 변화율

> [!NOTE]
> 추론 시 `beta`는 Shrinkage 적용: `β_shrunk = 0.5×β_coin + 0.5×β_cross_mean`
> → 유동성 낮은 코인의 추정 노이즈 감소

### 3.2 앙상블 모델 구성

> **참조**: `models/ensemble_configs.json`

| ID | 이름 | d_model | layers | heads | dropout | CNN | bagging |
|----|------|---------|--------|-------|---------|-----|---------|
| 1 | Micro-Pattern | 128 | 2 | 4 | 0.1 | 1D | 0.8 |
| 2 | Macro-Trend | 256 | 4 | 8 | 0.2 | 1D | 0.8 |
| 3 | Balanced A | 128 | 3 | 4 | 0.15 | 1D | 0.9 |
| 4 | Balanced B | 128 | 3 | 4 | 0.15 | **2D (GAF)** | 0.9 |
| 5 | Deep Complex | 256 | 4 | 8 | 0.3 | 1D | 1.0 |

### 3.3 모델 내부 구조

```
입력: (B, 168, 19)
  │
  ├──→ [Transformer Encoder]
  │     ├─ Linear(19→d_model) × √d_model
  │     ├─ ContextualPositionalEncoding:
  │     │   ├─ Static PE (sinusoidal)
  │     │   └─ context_proj(CONTEXT_DIM=2→d_model)
  │     │       → src[:,:,8:10] (market_index + hist_similarity)
  │     ├─ AttentionExtractorEncoderLayer × N:
  │     │   ├─ MultiheadAttention(d_model, n_heads)
  │     │   ├─ FFN: d_model→d_model×4→d_model
  │     │   └─ LayerNorm × 2
  │     └─ 마지막 스텝 → (B, d_model)
  │
  ├──→ [CNN Encoder] → (B, 128)
  │     ├─ 1D (TCN-style):
  │     │   ├─ Conv1d(19→64, k=3, dilation=1) → ReLU → Dropout
  │     │   ├─ Conv1d(64→64, k=3, dilation=2) → ReLU → Dropout
  │     │   ├─ Conv1d(64→128, k=3, dilation=4) → ReLU → Dropout
  │     │   └─ AdaptiveAvgPool1d(1)
  │     └─ 2D (GAF):
  │         ├─ 시계열 → Gramian Angular Field 이미지 변환
  │         ├─ Conv2d(19→32→64→128) + BN + MaxPool
  │         └─ AdaptiveAvgPool2d(1)
  │
  └──→ [ExplainableGatedFusion]
        ├─ prototype_bank: 16개 학습된 프로토타입
        ├─ 각 인코더 출력과 프로토타입의 cosine 유사도 계산
        ├─ gate = sigmoid(transformer + cnn + tf_sim + cnn_sim)
        ├─ output = gate × transformer + (1-gate) × cnn
        └─ 정규화: gate_range + gate_diversity + proto_diversity

  → [GAN Decoder] = Linear(d_model+32→256→512→256→6)
  → 출력: (B, 6) 6시간 가격변화율 예측

  → [Critic] = Linear(6→128→256→1) (학습 시만)
```

### 3.4 Gate 해석

| gate 값 | 모드 | 의미 |
|---------|------|------|
| < 0.4 | **Pattern** (CNN 우세) | 국소 패턴이 중요한 시장 |
| 0.4~0.6 | **Hybrid** (균형) | 혼합 상태 |
| > 0.6 | **Trend** (Transformer 우세) | 거시 추세가 중요한 시장 |

---

## 4. 학습 프로세스

> **참조**: `training/trainer.py`

### 4.1 손실 함수 (5종)

| 손실 | 가중치 | 역할 |
|------|--------|------|
| **WGAN-GP** | 동적 λ_gp | Wasserstein + Gradient Penalty |
| **Reconstruction** | 동적 λ_recon | MSE(생성, 실제) — 정확도 |
| **ECE** | 고정 0.1 | 확률 보정 — 과신 방지 |
| **Direction Balance** | 고정 0.5 | 방향 편향 방지 (일방적 매수/매도) |
| **Gate Reg** | 고정 2.0 | Gate 극단값 방지 + 다양성 |

### 4.2 동적 조정 (학습 안정화)

- **Dynamic λ**: 매 에폭 adv/recon 비율에 따라 λ_recon, λ_gp 자동 조정
- **Dynamic Critic Iters**: adv_loss에 따라 Critic 반복 횟수 5~10 자동 조정
- **Auto-Stop**: adv ratio < -10 또는 grad_norm 불안정 200 steps 지속 시 학습 중단
- **LR Scheduler**: CosineAnnealingLR (T_max=epochs, eta_min=lr×0.01)

### 4.3 Optuna 튜닝

- **목표**: Spearman ρ(|error|, uncertainty) 최대화
- **CV**: 5-Fold TimeSeriesSplit, 15 에폭 빠른 평가
- **튜닝 대상**: lr_g, lr_c, d_model, n_layers, n_heads, batch_size, 각종 λ, dropout_p

### 4.4 앙상블 다양성

- **Bagging**: 모델별 다른 bagging_ratio로 데이터 부분집합 (with replacement)
- **아키텍처**: 모델별 d_model, n_layers, n_heads, dropout, cnn_mode 차별화
- **Random seed**: `42 + model_id`

---

## 5. 추론 파이프라인

> **참조**: `inference/predictor.py`

### 5.1 MC-Dropout 앙상블 예측

```
5개 모델 × MC-Dropout 20회 = 100개 예측

for 각 모델:
    model.train()          # Dropout 활성화
    BatchNorm.eval()       # BN 통계 고정
    for 20회:
        예측 + gate값 수집

100개 예측 → 가중 평균 (모델별 실적 기반 가중치)
```

### 5.2 핵심 지표 계산

| 지표 | 계산 |
|------|------|
| **Uncertainty** | CV(100개 예측) × 100 — 낮을수록 확신 |
| **Consensus** | max(상승모델, 하락모델) / 전체모델 — 1.0=만장일치 |
| **Gate Value** | 전체 gate 평균 — Trend/Pattern 판단 |
| **Confidence** | 1 / (1 + uncertainty) |

---

## 6. 추천 퍼널 (6단계)

> **참조**: `inference/recommender.py`

| 단계 | 필터 | 임계값 |
|------|------|--------|
| 0 | Tradeable 마켓 | Upbit 상장 확인 |
| 1 | 방향 일관성 | ≥ 66% (6개 중 4개+) |
| 1.5 | 매크로 체제 + Lead-Lag | SMA20/60 교차, BTC 상관>0.3 → **15% boost** |
| 2 | 유동성 | live: 10억원, backtest: 5천만원 |
| 3 | 최소 수익률 | |expected_return| ≥ 0.1% |
| 4 | 불확실성 | ≤ 500 (역추세: 350) |
| 5 | DTW 유사도 | soft-DTW < 1.5 (성공패턴 유사) |
| Final | Top-K | min_k=3, 부족 시 Forced Fallback |

**포지션 사이징**: `base(10%) × confidence × volatility_factor` → 최소 3%, 최대 20%

---

## 7. XGBoost 펌프 분류기 (System 2)

> **참조**: `training/pump_trainer.py`, `inference/pump_predictor.py`

- **목적**: 10%+ 급등 확률 예측 (GAN과 별개 독립 모델)
- **피처**: 17개 (GAN의 19개에서 alpha/beta 빠지고, volume_spike_score/squeeze_on/roc 추가)
- **4-class**: 급등 없음 / 10~15% / 15~20% / 20%+
- **범위**: **DB 전체 KRW 코인**, 마켓당 최근 100캔들
- **임계값**: `total_pump_prob = P(class1+2+3) > 0.2`

---

## 8. main.py 모드 카탈로그

| 모드 | 데이터 범위 | 핵심 동작 |
|------|-----------|----------|
| `init_db` | - | SQLite 테이블 생성 |
| `collect-all` | **전체 KRW** | Upbit API → DB (기본 90일) |
| `train` | BTC, ETH | Optuna 튜닝 → 5모델 앙상블 학습 |
| `train-pump` | **전체 KRW** | XGBoost 펌프 분류기 학습 |
| `daily` | **전체 시장 스캔** | 5-Step 파이프라인 (아래 상세) |
| `continuous` | 트렌딩+보유 종목 | 30분 주기 연속 매매 엔진 |
| `screen` | **전체 KRW** | Top 5 트렌딩 선별 |
| `quick-recommend` | 트렌딩 코인 | 학습 없이 기존 모델로 예측 |
| `backtest` | 과거 데이터 | Walk-forward 시뮬레이션 |
| `find-pumps` | **전체 KRW** | 급등 후보 탐색 |
| `explain` | 단일 코인 | 어텐션 맵 + 게이트 분석 |

### 8.1 Daily 모드 상세

```
Step 0: 성과 추적
  └─ 어제 추천 → 현재가 비교 → ModelTracker 업데이트 → 이전 포지션 닫기

Step 1: Trending 전략
  └─ screener Top 5 → 30일 수집 → fine-tune → 예측 → Top 3 추천

Step 2: Pattern 전략
  └─ 리더 코인 → DTW로 팔로워 탐색 → 예측 → Top 3 추천

Step 3: Pump 전략
  └─ XGBoost 학습 → 전체 KRW 스캔

Step 4: 리포트
  └─ 포트폴리오 업데이트 → Telegram daily report → Research report
```

### 8.2 Continuous 모드 상세

```
while True (30분 주기):
  1. screener Top 5 + 현재 보유 종목
  2. 각 마켓 14일 데이터 수집
  3. predictor.run() → 예측
  4. SmartPortfolioManager.sync_target_weight() → 리밸런싱
  5. 매매 시 Telegram instant alert
  6. 4시간마다 Status Report
  7. 08:00 AM Daily Report
```

---

## 9. 웹 대시보드

> **참조**: `app.py` (1,506줄), `templates/` (11개 HTML), `static/` (CSS+JS), `web_utils/`

### 9.1 기술 스택

| 컴포넌트 | 기술 |
|---------|------|
| 백엔드 | Flask + Eventlet |
| 실시간 | Flask-SocketIO (WebSocket) + SSE (로그 스트리밍) |
| 프론트 | HTML/CSS/JS + Chart.js |
| 터널 | Cloudflare Tunnel (외부 접속) |
| 캐싱 | Flask-Caching (선택, TTL 기반) |
| 인증 | API Key (헤더: `X-API-Key`) |

### 9.2 페이지 구성

| 페이지 | 경로 | 기능 |
|--------|------|------|
| **대시보드** | `/` | 실시간 시세, 최근 추천, 주간 성과, 포지션 현황 |
| **컨트롤** | `/control` | 학습/예측/백테스트 원클릭 실행 + 실시간 로그 |
| **성과분석** | `/performance` | 에쿼티 커브, 월별 수익, 7대 지표, 거래 내역 |
| **모델** | `/model` | 앙상블 가중치, Gate 모드, 하이퍼파라미터 |
| **백테스트** | `/backtest` | 기간별 성과 지표 + 시뮬레이션 |
| **리서치** | `/research` | 연구 리포트 열람 + AI 챗봇 + PDF 업로드 |
| **문서** | `/docs` | 프로젝트 문서 |
| **작업관리** | `/tasks` | TODO/로드맵 관리 |

### 9.3 주요 API 그룹

| 그룹 | 엔드포인트 예시 | 기능 |
|------|---------------|------|
| Task Control | `/api/train`, `/api/daily`, `/api/backtest` | 백그라운드 작업 실행 |
| Model Status | `/api/ensemble-status`, `/api/gate-status` | 모델 상태 조회 |
| Performance | `/api/performance`, `/api/performance/history` | 성과 데이터 |
| Market | `/api/market-overview`, `/api/ticker` | Upbit 프록시 (CORS 우회) |
| Research | `/api/research/chat`, `/api/research/upload` | RAG 질의응답 |
| Tasks | `/api/tasks` | 작업 CRUD |

### 9.4 보조 모듈

| 파일 | 역할 |
|------|------|
| `web_utils/data_loader.py` | 추천 CSV 파싱, 성과 계산, 주간 통계 |
| `web_utils/task_runner.py` | subprocess로 main.py 백그라운드 실행 + 로그 스트리밍 |
| `web_utils/config_reader.py` | config.py 설정값을 프론트엔드용 JSON으로 변환 |

---

## 10. 텔레그램 알림 시스템

> **참조**: `utils/telegram_bot.py`

| 리포트 종류 | 발생 시점 | 내용 |
|------------|----------|------|
| **Daily Report** | 매일 daily 모드 완료 시 | Trending Top 3 + Pattern Top 3 + Pump 후보 |
| **Instant Alert** | continuous 모드 매매 시 | 매수/매도 + 가격 + 사유 + 실현 손익 |
| **Status Report** | 4시간마다 | 보유 포지션 + 총 자산 현황 |
| **Short-term Report** | continuous 리포트 시 | 4H 스캘핑 추천 + 펌프 |

- **중복 방지**: 메시지 해시(SHA256) 비교, 같은 내용 재발송 차단
- **대시보드 링크**: Cloudflare tunnel URL 자동 포함
- **HTML 모드**: 프리미엄 구조화 리포트 (이모지 + 테이블)

---

## 11. 포트폴리오 관리

### 11.1 PortfolioManager (`utils/portfolio_manager.py`)

> Daily 모드용 **Paper Trading** 시스템

- **저장소**: `data/portfolio.db` (SQLite)
- **기능**: 거래 기록, PnL 추적, 에쿼티 커브, 전략별 포지션 관리
- **전략 구분**: trending, pattern, continuous 등 전략별 별도 추적
- `close_open_trades(strategy, prices)`: 이전 주기 포지션 정리 + 실현 손익 계산

### 11.2 SmartPortfolioManager (`utils/smart_portfolio.py`)

> Continuous 모드용 **실시간 리밸런싱** 시스템

- **저장소**: `portfolio_state.json`
- **리밸런싱**: target_weight 기반 매매 판단
- **Gate/Consensus 반영**:
  - Consensus ≥ 0.8 + Trend 모드 → 1.2× 포지션 (공격적)
  - Consensus < 0.5 → 0.5× 포지션 (보수적)
  - 자산당 최대 50% cap
- **Position 관리**: Scale-in/out, Trailing stop 추적

### 11.3 ModelPerformanceTracker (`utils/model_tracker.py`)

- 모델별 rolling accuracy 추적 (최대 50개 기록)
- 정확도 비례 앙상블 가중치 반환 (데이터 부족 시 균등배분)
- Daily 모드에서 전략→모델 매핑으로 정확도 업데이트

---

## 12. Research Lab

> **참조**: `utils/research_assistant.py`, `utils/research_reporter.py`

### 12.1 Research Assistant (RAG 챗봇)

| 컴포넌트 | 기술 |
|---------|------|
| 문서 로더 | PyPDFLoader (PDF), 직접 파싱 (Markdown) |
| 텍스트 분할 | RecursiveCharacterTextSplitter (chunk=1000, overlap=200) |
| 임베딩 | OpenAI Embeddings |
| 벡터 스토어 | FAISS (`data/faiss_index`) |
| LLM | ChatOpenAI (gpt-4) |
| Chain | ConversationalRetrievalChain (대화 맥락 유지) |

- **기본 인덱싱**: `Chrono-Trader_v2_Paper.md`, `README.md` 자동 로드
- **PDF 업로드**: `data/papers/` 디렉토리에 저장 → 벡터 인덱스에 추가
- **AI 질의응답**: 업로드된 논문 + 프로젝트 문서 기반 RAG 대화
- **연구 노트 저장**: Obsidian 형식 Markdown 출력
- **대화 요약**: 채팅 이력 → 구조화된 연구 노트로 변환

### 12.2 Research Reporter (자동 리포트)

- 매일 Daily 모드 완료 후 `research_reports/` 에 Markdown 리포트 자동 생성
- 오늘 추천 분석 + 어제 추천 성과 검증 포함
- 전략별 통계(평균 수익률, 불확실성, 신뢰도) 요약

---

## 13. 데이터 계층

### 13.1 데이터베이스

| DB 파일 | 용도 | 핵심 테이블 |
|---------|------|-----------|
| `data/crypto_data.db` | 시세 데이터 (84MB+) | `crypto_data` (timestamp, market, OHLCV) |
| `data/portfolio.db` | 거래 기록 | `trades` |
| `data/tasks.db` | 작업 관리 | `tasks` |

### 13.2 Price Cache (`utils/price_cache.py`)

- **문제 해결**: N+1 쿼리 방지 — 단일 API 호출로 전체 시세 배치 조회
- **실시간용**: 10초 TTL → 포지션 PnL 계산
- **대시보드용**: 60초 TTL → 표시용 (API 호출 최소화)
- **Tradeable 마켓 캐시**: 1시간 TTL → 상장 여부 확인

### 13.3 파일 기반 데이터

| 파일/디렉토리 | 용도 |
|-------------|------|
| `predictions/` | 예측 결과 CSV |
| `recommendations/` | 추천 결과 CSV (일별) |
| `research_reports/` | 자동 연구 리포트 (Markdown) |
| `models/*.pth` | PyTorch 앙상블 모델 (5개) |
| `models/pump_classifier.joblib` | XGBoost 펌프 모델 |
| `models/model_performance.json` | 모델별 실적 데이터 |
| `models/ensemble_configs.json` | 앙상블 구성 정의 |
| `data/success_patterns.npy` | DTW 비교용 성공 패턴 |
| `data/pattern_library.joblib` | 패턴 라이브러리 |
| `portfolio_state.json` | SmartPortfolio 상태 |

---

## 14. 자동화 인프라

### 14.1 Cron 스케줄링

> **참조**: `cron_schedules.txt`

| 스케줄 | 모드 | 로그 |
|--------|------|------|
| 매일 08:00 | `daily` | `cron_daily.log` |
| 30분마다 | `continuous` | `cron_continuous.log` |

#### 14.1.1 Scheduled Ops Contract

운영(스케줄) 모드는 `refresh-db`(수집)와 `intraday`/`morning-report`(추론)를 분리한다.

- 권장 엔트리포인트: `scripts/run_scheduled.py`
- 계약(완료 기준/exit code/세이프 모드): `OPS_ACCEPTANCE.md`

### 14.2 서버 관리

| 파일 | 역할 |
|------|------|
| `start_server.sh` | Flask 서버 + Cloudflare 터널 시작 |
| `start_app.sh` | Flask 앱만 시작 |
| `restart_all.sh` | 전체 서비스 재시작 |
| `auto_recovery.sh` | 프로세스 crash 시 자동 재시작 |
| `keep_alive_server.sh` | Flask 서버 alive 체크 |
| `keep_alive_tunnel.sh` | Cloudflare 터널 alive 체크 |
| `run_cloudflared.sh` | 터널 설정 + 실행 |

### 14.3 Auto-Retrain (`utils/auto_retrain.py`)

- 정확도 ≤ 25% 시 재학습 트리거
- 최소 48시간 간격 제한
- `subprocess.Popen("python main.py --mode train --epochs 50")`

### 14.4 Logrotate

- `logrotate_chrono.conf`로 로그 자동 회전
- `.logrotate.state` 에 상태 저장

### 14.5 n8n 워크플로 연동

| 워크플로 | 용도 |
|---------|------|
| `workflow_auto_prediction.json` | 자동 예측 트리거 |
| `workflow_data_health.json` | 데이터 파이프라인 헬스체크 |
| `workflow_research_digest.json` | 연구 다이제스트 생성 |

---

## 15. 백테스트 & 분석 도구

### 15.1 BacktestAnalyzer (`utils/backtest_analyzer.py`)

**7대 핵심 지표**:
1. Total Return (총 수익률)
2. Sharpe Ratio (위험 대비 수익)
3. Max Drawdown (최대 낙폭)
4. Win Rate (승률)
5. Alpha (BTC 대비 초과수익)
6. Trade Count (거래 횟수)
7. Profit Factor (이익/손실 비율)

포트폴리오 시뮬레이션 기반 (초기 1,000만원)

### 15.2 Evaluator (`training/evaluator.py`)

- Walk-forward 백테스트 엔진
- screener → predictor → recommender 실제 파이프라인 재현
- ECE + Spearman ρ 검증

### 15.3 분석 스크립트 (`analysis/`)

| 파일 | 기능 |
|------|------|
| `explainability_analyzer.py` | 어텐션 히트맵, 프로토타입 t-SNE, 다중노이즈 예측분포 |
| `validate_uncertainty.py` | ECE + Spearman ρ + 불확실성 커트오프 곡선 |
| `tune_mc_dropout.py` | dropout_p × n_inferences 그리드 서치 |
| `visualize_gate.py` | Gate 분포/시계열/체제별 성과 시각화 |
| `check_mc_dropout.py` | MC-Dropout 동작 검증 |
| `explainability.py` | 단순 Explainability 유틸 |

### 15.4 유틸리티 스크립트 (`scripts/`)

| 파일 | 기능 |
|------|------|
| `build_pattern_library.py` | 성공 패턴 라이브러리 구축 |
| `recalibrate_ensemble.py` | 앙상블 가중치 재보정 |
| `recalibrate_gate.py` | Gate 분포 재보정 |
| `generate_deep_analysis_report.py` | 심층 분석 리포트 생성 |
| `sync_tasks_to_db.py` | 작업 동기화 |
| `verify_transformer_mask.py` | Transformer 마스크 검증 |

---

**이 문서의 모든 수치와 구조는 소스코드 기준입니다. 코드 변경 시 반드시 이 문서를 동기화하세요.**
