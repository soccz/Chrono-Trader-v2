# AETHER 시스템 아키텍처 정의서

> 최종 갱신: 2026-03-24 | 이 문서는 코드 기준 사실만 기술한다

---

## 0. 핵심 사상 (Why This Model Exists)

```
이 모델의 목적:
1. "지금 오르는 놈들"을 보고 "다음에 오를 놈"을 찾는다 (패턴 전이)
2. "지금 오르는 놈"이 계속 갈지 꺾일지 판단한다 (모멘텀 지속성)
3. 확신이 없으면 추천하지 않는다 (불확실성 우선)
```

**Non-Markovian 시장 가정** -- 현재 가격만으로 미래를 결정할 수 없다. 168시간 전체 경로가 필요하다.
Transformer의 Self-Attention이 이 경로 의존성을 학습한다.

**거시 -> 개별 전이** -- BTC/ETH 같은 리더가 먼저 움직이고, 알트가 따라온다.
market_index_return과 cross_corr_btc 피처가 이 리드-래그를 포착한다.
Contextual PE가 거시 상태를 시간 인코딩에 직접 주입한다.

**점예측 거부** -- 미래는 하나의 숫자가 아니라 가능한 경로들의 분포다.
GAN과 CVAE 두 디코더가 각각 다른 방식으로 경로 분포를 생성한다.
5 models x 20 MC-Dropout = 100개 시나리오에서 불확실성을 정량화한다.

**확신 없으면 침묵** -- 불확실성이 높으면 추천하지 않고 Watch로 빠진다.
추천 퍼널의 7단계 필터가 이를 강제한다.

---

## 1. 모델 구조

### 1.1 인코더: 두 개의 눈

| 구분 | Transformer (Global Eye) | TCN/CNN (Local Eye) |
|------|--------------------------|---------------------|
| 입력 | 168h x 32 features | 동일 |
| 역할 | 거시 흐름, 레짐 전환, 리드-래그 | 캔들 패턴, 국소 가격 구조 |
| 특징 | Full Self-Attention | 1D Conv (Model 1-3,5) / 2D GAF (Model 4) |
| PE | Contextual PE: idx 8 (market_index_return) + idx 9 (historical_similarity) 주입 | N/A |

### 1.2 Explainable Gated Fusion

- `gate` 값 [0,1]: Transformer vs CNN 비중을 동적 결정
- Prototype Bank: 학습된 과거 성공 패턴 저장소
- 현재 입력과 Prototype을 비교하여 gate 결정 -- "지금은 거시가 80% 지배적" 같은 설명 가능
- Gate regularization: range penalty + diversity penalty + prototype diversity

### 1.3 디코더: 미래를 분포로

| 구분 | GAN Decoder | CVAE Decoder |
|------|-------------|--------------|
| 입력 | noise(100d) + encoder context | encoder context + latent z(32d) |
| 출력 | 3h residual return 경로 | 3h residual return 경로 |
| 학습 | WGAN-GP Critic | KL divergence + reconstruction |
| 다양성 | noise 샘플링 | latent space 샘플링 |

**설계 판단**: CVAE가 조건부 분포 p(y|c) 학습에 더 적합. GAN Critic은 경로를 무조건적으로 평가(context 미사용)하여 주변 분포만 학습. CVAE를 주 디코더로 확정하고, GAN은 보조/폐기 검토.

### 1.4 불확실성 정량화

**현재 구현**:
- 5 models x 20 MC-Dropout = **100개 시나리오**
- `uncertainty = CV(std/mean)`, 분모 floor = 0.002
- `PI_80 = [10th, 90th percentile]` of cumulative returns
- `consensus = max(up_models, down_models) / total_models`

**문제점 및 개선 방향**:

| 현재 | 문제 | 권장 |
|------|------|------|
| CV(std/mean) | 평균 ~0일 때 분모 floor에 퇴화, 해석 불가 | `mean(std)` 단순 표준편차로 교체 |
| MC-Dropout on CVAE | CVAE는 eval모드로 돌려 인식론적 불확실성 0 | CVAE: `sample(n=200)` + 앙상블 분산 분리 |
| PI_80 from 100 samples | 10th percentile 추정 불안정 (최소 250+ 필요) | CVAE samples 200/model → 총 800+ |
| 보정 검증 없음 | 실제 80% 커버리지인지 미확인 | 평가기에 PICP(커버리지율) 추적 추가 |
| epistemic/aleatoric 혼재 | MC-Dropout이 두 불확실성을 혼합 | 앙상블 분산(인식론적) vs 모델내 분산(우연적) 분리 보고 |

---

## 2. 피처 설계 (32개)

### 2.1 피처 목록

| Idx | Feature | 카테고리 | 역할 |
|-----|---------|----------|------|
| 0 | close | 가격 | 종가 (정규화된) |
| 1 | volume | 거래량 | 거래량 |
| 2 | rsi | 기술지표 | 상대강도지수 |
| 3 | macd | 기술지표 | MACD 라인 |
| 4 | macdsignal | 기술지표 | MACD 시그널 |
| 5 | macdhist | 기술지표 | MACD 히스토그램 |
| 6 | adx | 기술지표 | 추세 강도 |
| 7 | obv | 기술지표 | On-Balance Volume |
| **8** | **market_index_return** | **맥락/PE** | **BTC+ETH 가중 시장 수익률 -- Contextual PE 입력** |
| **9** | **historical_similarity** | **맥락/PE** | **과거 유사 패턴 매칭 점수 -- Contextual PE 입력** |
| 10 | bb_upper | 기술지표 | 볼린저 밴드 상단 |
| 11 | bb_middle | 기술지표 | 볼린저 밴드 중앙 |
| 12 | bb_lower | 기술지표 | 볼린저 밴드 하단 |
| 13 | volume_ma | 거래량 | 거래량 이동평균 |
| 14 | volatility_24h | 변동성 | 24시간 변동성 |
| 15 | volatility_7d | 변동성 | 7일 변동성 |
| 16 | volume_volatility | 변동성 | 거래량 변동성 |
| 17 | alpha | 팩터 | CAPM 알파 (BTC 대비 초과수익) |
| 18 | beta | 팩터 | CAPM 베타 (BTC 민감도) |
| 19 | price_position | 기술지표 | 가격 위치 (최근 범위 내) |
| 20 | volume_ratio | 거래량 | 현재/평균 거래량 비율 |
| 21 | return_skew_24h | 통계 | 24시간 수익률 비대칭도 |
| 22 | cross_corr_btc | 맥락 | BTC와의 교차상관 |
| 23 | factor_size | 팩터 | 시가총액 팩터 |
| 24 | factor_mom | 팩터 | 모멘텀 팩터 |
| 25 | factor_vol | 팩터 | 변동성 팩터 |
| 26 | factor_liq | 팩터 | 유동성 팩터 |
| 27 | btc_regime | 레짐 | BTC 4-state 레짐 (0-3), 비정규화 |
| 28 | btc_regime_rv | 레짐 | BTC 실현변동성 포지션 (정규화) |
| 29 | btc_ma_distance | 레짐 | price/MA200h - 1 |
| 30 | factor_mom_x_bull | 레짐x팩터 | 모멘텀 x 불장 교호작용 |
| 31 | factor_liq_x_bear | 레짐x팩터 | 유동성 x 약장 교호작용 |

Contextual PE 입력: idx 8 (`MARKET_INDEX_FEATURE_IDX`), idx 9 (`HISTORICAL_SIMILARITY_FEATURE_IDX`), `CONTEXT_DIM = 2`

### 2.2 피처 개선 방향

**제거 대상 (중복/저가치)**

| 피처 | 이유 | 대체 |
|------|------|------|
| `close` (raw) | `price_position`이 이미 정규화된 위치 제공 | `log_return` 시리즈 |
| `volume` (raw) | `volume_ratio`, `volume_ma`로 대체됨 | `log_volume_change` |
| `obv` | MinMaxScaler 후 단조증가 프록시. `volume_ratio`와 중복 | 제거 |
| `macdsignal` | `macd - macdhist`로 선형 종속 | 제거 |
| `volatility_7d` | `volatility_24h`와 고상관. 168h 시퀀스가 이미 7일 커버 | 제거 |
| `bb_upper/middle/lower` | 3개가 하나의 정보. 스케일링 후 관계 소실 | `bb_position = (close-lower)/(upper-lower)` 1개로 |

**추가 대상 (패턴 전이 핵심)**

| 피처 | 카테고리 | 역할 | 우선순위 |
|------|----------|------|----------|
| `breadth_ratio` | 시장전이 | 4h 양수 수익률 코인 비율. 0.5=로테이션, 1.0=광범위 랠리 | P1 |
| `top_n_return_1h` | 시장전이 | 상위 5개 코인 평균 1h 수익률. "지금 뭐가 펌핑?" | P1 |
| `top_n_return_4h` | 시장전이 | 상위 5개 코인 평균 4h 수익률 | P1 |
| `rank_return_4h` | 시장전이 | 해당 코인의 4h 수익률 순위 백분위. 0=후행, 1=선행 | P1 |
| `net_volume_flow` | 자금흐름 | `sum(volume * sign(return))` 6h rolling. 매수/매도 방향 | P1 |
| `volume_breadth` | 자금흐름 | 거래량 > 2x MA인 코인 비율. 시장 전체 활성화 | P1 |
| `cross_corr_btc_lag1` | 리드-래그 | `corr(coin(t), btc(t-1))` 48h rolling. 1시간 후행 측정 | P2 |
| `cross_corr_btc_lag3` | 리드-래그 | `corr(coin(t), btc(t-3))` 48h rolling. 3시간 후행 측정 | P2 |
| `rs_vs_market_24h` | 상대강도 | `coin_return_24h - market_index_return_24h` | P2 |
| `rs_momentum` | 상대강도 | `rs_vs_market`의 4h 변화율. 상대강도 가속도 | P2 |

**적용 경로**: Phase 1에서 7개 제거 + `bb_position` 추가 (32→26), Phase 2에서 P1 6개 추가 (26→32), Phase 3에서 P2 4개 추가 (32→36)

### 2.3 시퀀스 길이 및 예측 호라이즌

**현재**: `SEQUENCE_LENGTH = 168` (7일), `FUTURE_WINDOW_SIZE = 3` (3시간)

**권장 변경**:
- 시퀀스: 168 → **96** (4일). 패턴 전이는 48-96h에 집중. Attention O(n^2) 비용 ~3x 절감. 7일 정보는 `return_7d` 요약 피처로 보상.
- 호라이즌: 3h 유지. 알트 후행 반응이 1-6h에 피크. 단, 멀티호라이즌 출력(1h, 3h, 6h) 멀티태스크 학습 고려.

---

## 3. 앙상블 설계

### 3.1 모델별 구성

| Model | 이름 | d_model | n_layers | n_heads | dropout | CNN | bagging |
|-------|------|---------|----------|---------|---------|-----|---------|
| 1 | Micro-Pattern Specialist | 128 | 2 | 4 | 0.10 | 1D | 0.8 |
| 2 | Macro-Trend Specialist | 256 | 4 | 8 | 0.20 | 1D | 0.8 |
| 3 | Balanced Hybrid A | 128 | 3 | 4 | 0.15 | 1D | 0.9 |
| 4 | Balanced Hybrid B | 128 | 3 | 4 | 0.15 | **2D** | 0.9 |
| 5 | Deep Complex | 256 | 4 | 8 | 0.30 | 1D | 1.0 |

### 3.2 다양성 분석

현재 추정 모델간 상관: ~0.91. 실효 앙상블 크기: ~2.2 (5개 중).

- **Model 3 vs 4**: Transformer 동일 (128/3/4/0.15). CNN만 1D vs 2D(GAF). 추정 rho=0.92-0.96.
- **Model 2 vs 5**: 구조 동일 (256/4/8). dropout만 0.2 vs 0.3. 추정 rho=0.88-0.93.
- 다양성 축이 부족: 동일 lookback, 동일 피처셋, 동일 loss, 동일 디코더.

### 3.3 앙상블 재설계 방향 (4모델, 역할 특화)

| Model | 이름 | 전문 영역 | lookback | 피처 그룹 | 구조 | 비고 |
|-------|------|-----------|----------|-----------|------|------|
| 1 | Scalper | 단기 마이크로 모멘텀 | **8h** | 가격 미시구조 | d=64, L=2, H=2, CNN=1D(k=2) | 짧은 수용장, 빠른 반전 감지 |
| 2 | Swing Trader | **패턴 전이** 핵심 | **24h** | 전체 | d=128, L=3, H=4, CNN=**2D**(GAF) | GAF 시각 패턴 매칭, Prototype Bank 활용 |
| 3 | Trend Follower | 장기 모멘텀 지속성 | **72h** | 추세/교차시장 | d=256, L=4, H=8, CNN=1D(k=5) | 대형 커널 TCN, 주간 구조 포착 |
| 4 | Regime Sentinel | 변동성/꼬리 리스크 | **48h** | 변동성/레짐 | d=128, L=3, H=4, **비대칭 loss** | 하락 패널티 2x. 위험 시 컨센서스 오버라이드 |

**기대 효과**: 평균 rho 0.91 → 0.41, 실효 앙상블 크기 2.2 → 3.1 (+41%), 모델 수는 5→4 (20% 연산 절감)

**구현 요구**: 모델별 lookback 슬라이싱, 피처 그룹 마스킹, 비대칭 loss 추가

---

## 4. 학습 계약

### 4.1 예측 대상

- **Residual return**: `coin_return - beta x BTC_return`, 3시간 horizon
- Beta: 168시간 rolling window (Optuna 탐색 범위 [72, 168, 336])

### 4.2 손실 함수

Optuna 탐색 시 (trainer.py:233):
```
L = L_adv + lambda_recon * L_recon_weighted + lambda_ece * L_ece
    + lambda_direction * L_direction + beta_KL * L_KL
```

Full training 시 (trainer.py:733):
```
L = L_adv + lambda_recon * L_recon_weighted + lambda_ece * L_ece
    + L_balance + 0.5 * L_gate_reg + lambda_direction * L_direction
    + beta_KL * L_KL
```

| 항목 | 현재 값 (model_config.json) | 설명 |
|------|---------------------------|------|
| lambda_recon | 2.917 | 가중 MSE (큰 움직임에 높은 가중치) |
| lambda_ece | 0.404 | Quantile Calibration Error |
| lambda_direction | 0.995 | BCE(방향 예측 정확도) |
| beta_KL | 0.01 (default) | CVAE KL divergence |
| gate_reg weight | 0.5 (하드코딩) | Gate range + diversity + prototype diversity |

### 4.3 학습 설정

- CV: Purged Walk-Forward, **6시간 embargo**
- Gradient clipping: **1.0**
- Optuna loss에는 `L_balance`, `L_gate_reg` 누락 (Known Gap #6)

---

## 5. 추론 계약

- MC-Dropout: `model.train()` + `BatchNorm.eval()`
- 20 samples per model, 5 models = **100 scenarios**
- Weighted ensemble average (model_tracker weights 기반)
- Prototype matching: soft-DTW 거리로 과거 성공 패턴과 비교

---

## 6. 추천 퍼널 (코드 기준 실제 순서)

| Step | 이름 | 기준 | 비고 |
|------|------|------|------|
| 0 | Tradeable | 상폐/스테이블/레버리지 제외 | EXCLUDE 리스트 |
| 1 | Net Alpha + Step1 Score | `net_alpha = directional_return - cost_budget` | fee 0.05% + slippage 0.03% per leg |
| | | `step1_score = net_alpha - penalty * max(0, floor - pi_guard)` | penalty: 0.35 (기본), 0.10 (intraday long) |
| 1.5 | 레짐/리드-래그 조정 | SMA(20/60) crossover, BTC lead-lag correlation | trend_confidence_boost = 1.05 |
| 2 | 유동성 | 24h 거래대금 >= 10억 KRW | backtest: 5천만 |
| 3 | 최소 기대수익률 | abs(expected_return) >= 0.1% | MIN_SIGNAL_RETURN |
| 3.5 | 컨센서스 | >= 0.6 (intraday >= 0.55, counter-trend >= 0.8) | |
| 4 | 불확실성 | 적응적 임계값 (batch 65th quantile 기반) | multiplier clamp [0.8x, 4.0x] |
| | | counter-trend: 0.7x 배수 (30% 더 엄격) | |
| 5 | DTW 패턴 유사도 | DTW distance <= 1.5 | success_pattern_min_return = 0.15 |
| 6 | Short 차단 | 현물 실행 환경에서 short 신호 watch-only 전환 | LIVE_ALLOW_SHORT_EXECUTION=false |
| 7 | 최종 선택 + MinRec | 생존자 없으면 MinRec 발동 | mode: "trade" or "watch" |
| | | MinRec 실패 시 watch-only fallback 보장 | MIN_REC_ALLOW_WATCH_ONLY_FALLBACK=true |

---

## 7. 알려진 갭 (Known Gaps)

### HIGH

| # | 문제 | 위치 | 영향 |
|---|------|------|------|
| ~~1~~ | ~~CVAE 경로 앙상블 집계~~ | predictor.py:333-334 | **해결됨** -- model_avg_patterns, model_directions 모두 CVAE 분기에서도 생성 확인 |
| 2 | fillna(0) 거짓 신호 | preprocessor.py:426,580 | RSI=0, volatility=0 등 초기 168행 오염 |
| 3 | Freshness gate 예외 시 무시 | scheduled_modes.py:261 | generic except가 gate를 bypass |
| 4 | 캔들 갭 미감지 | preprocessor.py (pct_change) | 업비트 점검 등으로 캔들 누락 시 pct_change가 다중 시간 수익률을 단일 봉으로 계산. 갭 감지/보간 로직 없음 |

### MEDIUM

| # | 문제 | 위치 | 영향 |
|---|------|------|------|
| 5 | roll_std bfill() 미래 정보 누출 | preprocessor.py:483 | return clipping용 rolling std의 초기 NaN을 미래 값으로 채움. beta 자체는 ffill (정상) |
| 6 | Optuna loss != 실제 학습 loss | trainer.py:233 vs 733 | gate_reg, balance 누락 -> HP 탐색 편향 |
| 7 | CVAE logvar 무제한 | cvae_decoder.py:67 | `clamp(-6, 2)` 필요. exp() 폭발 위험 |
| 8 | 앙상블 다양성 부족 | ensemble_configs.json | Model 3/4, 2/5 구조 거의 동일. 실효 크기 ~2.2 |
| ~~9~~ | ~~FAIL_ON_STALE_DATA_LIVE config class~~ | main.py:276 | **해결됨** -- config.Recommender에 정상 정의 확인 (config.py:183) |
| 10 | watch_only 메트릭 미기록 | run_markets_metrics.py | 성과 추적 불완전 |
| 11 | refresh-db 타임아웃 시 exit code 1 | refresh-db | 계약은 exit 3이어야 함 |
| 12 | ECE loss sigmoid 포화 | trainer.py:215-219 | unbounded sum에 sigmoid → gradient dead zone |
| 13 | CVAE val이 posterior 사용 | trainer.py:827 | val loss 낙관적 → 과적합 체크포인트 선택 |
| 14 | CV 불확실성 지표 퇴화 | predictor.py:363-368 | 평균 ~0일 때 floor에 의존, 해석 불가 |
| 15 | PI_80 보정 미검증 | evaluator 부재 | 실제 커버리지율 추적 없음 |

### LOW

| # | 문제 | 위치 | 영향 |
|---|------|------|------|
| 16 | 추천 퍼널 순서 문서 불일치 | (이 문서로 수정됨) | -- |
| 17 | Dead config | config.py | KELLY_FRACTION, MAX_POSITIONS, PUMP_PROBABILITY_THRESHOLD 미사용 |
| 18 | Gate reg weight 코드 0.5 vs 이전 문서 2.0 | trainer.py:738 | 문서 오류 (코드가 정본) |
| 19 | GAN Critic이 context를 안 봄 | critic.py | 무조건적 경로 평가 → 조건부 분포 미학습 |
