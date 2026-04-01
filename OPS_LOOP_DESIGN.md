# Ops Loop Design

## Goal

논문 루프가 아니라 운영 루프를 고정한다.

이 저장소의 다음 단계 목표는 아래 한 줄이다.

- `수집 -> 검증 -> 예측 -> 추천 -> 기록 -> 감시 -> 복구`가 매일 끊기지 않고 돈다.

모델 코드는 완성됐다고 가정한다.
이 문서는 그 모델을 실제로 매일 살려두는 운영 기준이다.

## Operational Status (as of 2026-03-24)

시스템은 현재 매일 가동 중이다.

- **systemd timers**: unit 파일은 `deploy/systemd/`에 존재하나, 시스템에 설치/로드되지 않음 (`systemctl list-timers` 결과 0개). 현재 스케줄 실행은 수동 또는 별도 메커니즘에 의존
- **Intraday**: 총 163회 실행, 최근 5일 연속 성공, 최신 실행 2026-03-24 12:05 KST
- **Morning**: 매일 08:05 KST 실행 중
- **refresh-db**: 매 inference 전 실행, 24개 마켓 대상
- **추천 산출물**: CSV 13,000건 이상 누적
- **Metrics**: jsonl append-only 로그 3개 모드(intraday/morning/healthcheck) 모두 정상 기록
- **Web dashboard**: gunicorn:5001, 2025-02-15 이후 가동 중 (신규 route 반영을 위해 재시작 필요)

## Current Entry Points

현재 코드 기준의 진입점은 이미 있다.

- `main.py --mode refresh-db`
- `main.py --mode intraday`
- `main.py --mode morning-report`
- `scripts/run_scheduled.py`
- `scripts/ops_healthcheck.py`
- `deploy/systemd/*.service`
- `deploy/systemd/*.timer`

즉 새 파이프라인을 발명할 필요는 없고, 현재 흩어진 로직을 표준 루프로 잠근다.

## Standard Loop

### 1. Refresh Loop

목적:
- 예측 전에 DB를 최신 상태로 만든다.
- 전 종목이 아니라 `선별된 run_markets`만 갱신한다.

입력:
- screener seed
- rotation cache
- market budget
- refresh days

동작:
- 네트워크 확인
- seed 시장 선정
- scheduled selection
- run_markets rotation
- `collector.collect_market_data()` 실행
- refresh 결과 기록

실패 규칙:
- DNS 실패: `exit=2`
- 일부 종목 refresh 실패: 전체 중단 대신 경고 후 계속
- refresh 전체 실패여도 DB가 충분히 fresh하면 inference는 진행 가능

현재 코드 대응:
- `main.py:231` — `maybe_auto_refresh_markets()` (예측 전 자동 refresh)
- `main.py:805` — `handle_refresh_db()` (refresh-db 모드 핸들러)
- `scripts/run_scheduled.py:163` — refresh-db 커맨드 조립 및 실행

### 2. Freshness Gate

목적:
- stale snapshot으로 실거래성 추천을 내지 않게 막는다.

판정 기준:
- intraday: `DATA_FRESHNESS_MAX_LAG_HOURS_INTRADAY`
- morning: `DATA_FRESHNESS_MAX_LAG_HOURS_MORNING`

동작:
- run_markets별 최신 timestamp 확인
- stale/empty market drop
- 전부 stale이면 `KRW-BTC`, `KRW-ETH` fallback 확인
- fallback도 stale면 `exit=2`

실패 규칙:
- fresh market 1개도 없으면 일반 inference 금지
- stale fallback rerun은 `watch-only`로만 허용

현재 코드 대응:
- `main.py:265` — `enforce_freshness_gate()` (stale market drop + fallback 판정)
- `scripts/run_scheduled.py:232` — freshness gate 위임 및 stale abort 시 watch-only rerun

### 3. Inference Loop

목적:
- 선정된 run_markets에 대해 예측과 추천을 생성한다.

동작:
- `predictor.run(markets=run_markets)`
- strategy tag 부여
- `recommender.run(...)`
- CSV/JSON/metrics 저장
- Telegram 발송

현재 코드 대응:
- `main.py:340` — `run_recommender_for_markets()` (predictor.run + recommender.run)
- `main.py:870` — `handle_intraday()` (intraday 모드 진입점)
- `main.py:885` — `handle_morning_report()` (morning 모드 진입점)

### 4. Persistence Loop

목적:
- 운영 결과를 나중에 분석할 수 있게 남긴다.

반드시 남길 것:
- run timestamp
- selected markets
- dropped stale markets
- refresh 수행 여부
- recommendation count
- watch-only 여부
- elapsed time

현재 코드 대응:
- `main.py:192` — `persist_run_outputs()` (CSV/metrics/output contract 저장)
- `utils/run_markets_metrics.py` — `record_run()` (jsonl append-only 기록)
- `scripts/ops_healthcheck.py:16` — `main()` (mode_health 판정 + Telegram 알림)

### 5. Health Loop

목적:
- 조용히 망가지는 것을 막는다.

체크 대상:
- intraday 최근 성공 여부
- morning 최근 성공 여부
- recommendation count
- output freshness

알림 기준:
- intraday age 초과
- morning age 초과
- recs=0

현재 코드 대응:
- `scripts/ops_healthcheck.py:16` — `main()` (intraday/morning mode_health 판정)
- `app.py:471` — `/api/ops/overview` (웹 대시보드 운영 상태 API)
- `app.py:613` — `/api/health/data-pipeline` (데이터 파이프라인 건강 체크 API)

## Operating Modes

### Intraday

목적:
- 4시간 단위로 짧은 추천 루프를 돈다.

기준:
- 빠른 refresh
- 낮은 budget
- 짧은 freshness 기준
- telegram short report

현재 기본 성격:
- 실시간성 우선
- 실패 시 watch-only fallback 허용

### Morning

목적:
- 하루 한 번 더 넓은 범위의 보고서를 만든다.

기준:
- 더 큰 budget
- 더 긴 refresh
- trending + optional pattern followers + optional pump radar

현재 기본 성격:
- 보고서성 우선
- intraday보다 느리지만 더 많은 시장을 본다

## Failure Ladder

운영 실패는 아래 순서로만 처리한다.

1. `retry`
- 네트워크 일시 실패
- 특정 종목 refresh 실패

2. `degrade`
- 일부 stale market drop
- aux section skip
- market budget 축소

3. `safe-mode`
- `--allow_stale_data`
- `watch-only`
- Telegram 경고

4. `hard-stop`
- fresh market 0개
- DB 자체 비정상
- run lock 충돌 지속

핵심 원칙:
- 실패했다고 바로 침묵하지 않는다.
- 추천 품질이 보장 안 되면 `trade` 대신 `watch-only`로 떨어진다.

## Data Contract

모델이 살아 있으려면 데이터 계약이 먼저 지켜져야 한다.

필수 조건:
- `crypto_data`에 market/hourly candle이 중복 없이 저장된다.
- `KRW-BTC`, `KRW-ETH`는 항상 우선 refresh 대상이다.
- timestamp는 UTC 기준으로 정렬된다.
- stale/empty market은 inference 전에 제거된다.
- 신규상장/저유동성/dead market은 selection 단계에서 과도하게 끼지 않는다.

운영 지표:
- per-market lag hours
- refresh success ratio
- selected vs kept ratio
- stale drop count
- DB latest lag

## Recommended Canonical Loop

표준 루프는 아래로 고정한다.

1. `refresh-db`
2. `freshness gate`
3. `intraday` 또는 `morning-report`
4. `record_run`
5. `telegram report`
6. `healthcheck`
7. stale abort 시 `watch-only rerun` 1회만 허용

즉 실행 단위는:

- `scheduled run = refresh + infer + record + alert`

## What To Implement Next

다음 구현 우선순위는 이 문서 기준으로 간다.

### Priority 1. Data Refresh Hardening — 부분 완료

**현황**: refresh-db는 매 inference 전 24개 마켓 대상으로 안정적으로 실행 중.

**남은 작업**:
- [ ] refresh 결과를 per-market 성공/실패로 남긴다 (현재는 전체 성공/실패만 기록)
- [ ] `KRW-BTC`, `KRW-ETH` refresh 실패 시 별도 강한 경고를 건다
- [ ] collector retry/backoff를 표준화한다

### Priority 2. Freshness Unification — 부분 완료

**현황**: freshness 판정 로직은 존재하나, exception bypass 버그로 stale 데이터가 gate를 통과할 수 있음.

**남은 작업**:
- [ ] freshness gate의 exception bypass 버그 수정 (stale 데이터로 inference 진행되는 경로 차단)
- [ ] intraday/morning의 freshness 로직을 공통 함수로 뺀다
- [ ] `kept/dropped/fallback/watch-only`를 동일 포맷으로 기록한다

### Priority 3. Output Contract — 부분 완료

**현황**: CSV 추천 파일(13,000+건)과 jsonl metrics는 안정적으로 생성 중. 다만 output contract 스키마와 실제 필드 간 불일치 있음.

**남은 작업**:
- [ ] 추천 파일/metrics의 필드를 output contract 스키마에 맞춰 통일한다
- [ ] run metrics에 누락된 필드(watch_only, reason_code 등) 추가한다

### Priority 4. Health API Alignment — 미완료

**현황**: `scripts/ops_healthcheck.py`는 구현되어 있으나 systemd timer가 시스템에 설치되지 않아 자동 실행되지 않음 (unit 파일은 `deploy/systemd/`에 존재). `/api/ops/overview` 등 Health API route는 gunicorn 재시작 전이라 404 반환.

**남은 작업**:
- [ ] systemd timer unit 파일을 시스템에 설치하고 활성화한다 (`systemctl enable --now aether-*.timer`)
- [ ] gunicorn 재시작하여 신규 route 활성화
- [ ] `app.py` health API와 `ops_healthcheck.py` 판정 기준을 통일한다
- [ ] file age 기반 판정과 run metrics 기반 판정의 이중화 해소

### Priority 5. Recovery Policy — 미완료

**현황**: watch-only fallback 실행 결과가 metrics에 기록되지 않으며, 실패 reason code도 없음.

**남은 작업**:
- [ ] stale fallback 후 watch-only 실행 결과를 명시적으로 기록한다
- [ ] timeout, offline, stale, zero-rec를 서로 다른 reason code로 남긴다
- [ ] watch_only 필드를 metrics jsonl에 추가한다

## Definition Of Done

운영 루프가 완성됐다고 부를 최소 기준은 아래다.

- [x] refresh-db가 매일 안정적으로 돈다 — 매 inference 전 24개 마켓 refresh 확인됨
- [x] intraday가 stale market을 자동 drop한다 — 163회 실행 이력에서 동작 확인
- [x] morning-report가 최소 1회/일 정상 산출물을 남긴다 — 매일 08:05 KST 실행, CSV 누적 중
- [ ] stale/timeout/offline 때 silent failure가 없다 — **freshness gate에 exception bypass 버그 있음, stale 데이터로 inference 진행 가능**
- [ ] watch-only fallback이 명시적으로 기록된다 — **watch_only 필드가 metrics에 기록되지 않음**
- [ ] healthcheck가 실제 실패를 놓치지 않는다 — **systemd timer가 설치되지 않아 자동 실행 안 됨; 수동 실행은 가능**

## Decision

기본 운영 루프는 가동 중이다. 다음 단계는 **안전성 보강**이다.

현재 상태:
- `refresh -> inference -> record -> health` 루프는 매일 정상 가동 (3/6 DoD 완료)
- 남은 핵심 리스크: freshness gate bypass 버그, watch-only 미기록, Health API 미연결, systemd timer 미설치

앞으로의 기준:
- freshness gate 버그 수정이 최우선 (stale 데이터로 추천이 나가는 경로 차단)
- Health API 활성화로 대시보드에서 운영 상태 즉시 확인 가능하게
- watch-only/reason_code 기록으로 실패 분석 기반 확보
- 새로운 모델이나 논문 작업보다 위 3개가 먼저
