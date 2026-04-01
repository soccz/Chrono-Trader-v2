# CLAUDE.md

이 파일은 이 저장소에서 Claude Code가 따라야 할 **유일한** 프로젝트 작업 규칙이다.
글로벌 `~/.claude/CLAUDE.md`의 규칙(모델 전략, 응답 스타일, 작업 루틴, 토큰 절약, 기억 방어 등)은 여기서 반복하지 않는다.

---

## 1. 최우선 목표

- 이 저장소의 1순위는 `aaa/` 논문 트랙이 아니라 **사이트 + 운영 자동화 + 매일 업데이트 루프** 완성이다.
- `aaa/`는 사용자가 명시적으로 요청할 때만 건드린다.
- 우선순위 순서:
  1. 데이터 수집/신선도
  2. 스케줄 안정성
  3. 추천 산출물 생성
  4. 대시보드/운영 UI
  5. 성과 확인/회고
  6. 연구성 설명은 후순위

---

## 2. 문서 우선순위

운영 판단이 필요할 때 아래 순서로 참조한다. 연구 로그/최적화 리포트는 참고 자료일 뿐 운영 정본이 아니다.

1. `OPS_ACCEPTANCE.md` — 운영 계약
2. `OPS_LOOP_DESIGN.md` — 스케줄/루프 설계
3. `README.md` — 시스템 개요
4. `PROJECT_ARCHITECTURE.md` — 모델/파이프라인 구조
5. `PROJECT_LEARNING_PIPELINE.md` — 학습 파이프라인 상세

---

## 3. 런타임 진입점

| 용도 | 파일 |
|------|------|
| 스케줄 ops 진입점 | `scripts/run_ops_job.py` |
| 스케줄 러너 조립 | `scripts/run_scheduled.py` |
| CLI 오케스트레이터 | `main.py` |
| 웹 앱 | `app.py` |
| 빠른 검증 | `scripts/verify_site.sh quick` |
| 전체 검증 | `scripts/verify_site.sh full` |

---

## 4. 수정 전 확인 규칙

- 수정 전 반드시 해당 파일 + import/call site를 읽는다. "기억 안 나서 덮어쓰기" 절대 금지.
- 기존 동작을 단순화할 때는 **동작 동일, 구조 단순화**가 기본값이다.
- 새 기능보다 **운영 회귀 방지**가 우선이다.
- 수정 범위 밖의 코드를 건드리지 않는다.
- 코드와 문서가 불일치하면 **코드(런타임 동작)를 우선**하고, 불일치를 해당 문서에 기록한다.
- `NaN`, `None`, `Infinity`, 0 나누기 등 금융 데이터 방어를 항상 고려한다.
- API JSON 키(`entry_time` vs `entry_date` 등) 백엔드-프론트엔드 일치를 확인한다.
- 프론트엔드에서 null 값은 `"N/A"` 표시. `"0"` 이나 `"-"` 사용 금지.
- Flask API에서 `NaN`/`Infinity` 직렬화 방어: `NaNSafeJSONProvider` 패턴 사용.
- 500 에러 시 프론트엔드가 깨지지 않고 에러 메시지 UI(토스트/스피너/기본값)를 보여야 한다.

---

## 5. 웹/학습 분리 규칙

### 웹 코드 수정 시 (`app.py`, `templates/`, `static/`)
- `training/`, `inference/`, `main.py`, `scripts/` 절대 건드리지 않는다.
- 공유 데이터 저장소는 **읽기만** 한다.

### 학습/추론 코드 수정 시 (`main.py`, `training/`, `inference/`, `scripts/`)
- `app.py`, `templates/`, `static/` 절대 건드리지 않는다.

### 공유 모듈 (`utils/`) 수정 시
- 기존 함수 시그니처(입출력 형식) 변경 금지.
- 변경이 필요하면 **새 함수를 추가**하고 기존 함수는 유지한다.

### 문서 업데이트
- 동작이 변경되면 같은 턴에서 가장 가까운 운영/아키텍처 문서도 업데이트한다.

---

## 6. Ralph Loop 규칙

- "완료"라고 말하기 전에 반드시 스스로 검증한다.
- 검증이 실패하면 바로 원인 수정 후 다시 검증한다.
- 이 루프는 아래 둘 중 하나가 될 때까지 반복한다.
  - 검증 통과
  - 외부 의존성/권한/깨진 환경 등 명확한 blocker 확인
- 검증 없이 성공 보고 금지.

---

## 7. 검증 규칙

- 기본 검증: `scripts/verify_site.sh quick`
- 파이프라인을 건드렸으면: `scripts/verify_site.sh full`
- ops/scheduling 수정 시 최종 답변에 포함할 것:
  - 어떤 검증 명령을 돌렸는지
  - 통과/실패 여부
  - 남은 리스크
- UI 작업 시 아래 중 하나를 명시:
  - Flask route smoke test
  - 브라우저 수동 확인 경로
  - 관련 API 확인 커맨드

---

## 8. 서브 에이전트 역할

### Simplifier
- 책임: 중복 제거, 함수 분리, 조건문 단순화, 이름 정리
- 원칙: 동작 변경 금지, 인터페이스 유지, diff 최소화
- 결과물: "무엇을 단순화했고 왜 안전한지" 짧게 보고

### Verifier
- 책임: 테스트, 로그, 실행 경로 검증
- 원칙: 낙관적 가정 금지, 반드시 깨지는 경계 사례를 찾는다
- 결과물: "통과/실패, 실패 재현 방법, 의심 지점" 짧게 보고

---

## 9. 성공 기준

- scheduled run이 멈추지 않는다.
- stale/offline 상황에서 안전하게 degrade 한다.
- 결과 CSV와 metrics artifact가 매번 생긴다.
- 대시보드에서 최근 추천/성과/로그가 읽힌다.
- 추천이 0개가 되더라도 watch-only fallback이 남는다.
- 사용자는 "매일 데이터가 갱신되고 있다"는 감각을 사이트에서 바로 확인할 수 있다.

---

## 10. 답변 방식

- 결과 보고는 짧고 구체적으로 한다.
- 반드시 아래 셋을 포함한다.
  - 바꾼 것
  - 검증한 것
  - 남은 리스크 또는 다음 우선순위

---

## 11. 모델 개선 로드맵 (2026-03-28 확정)

90일 백테스트 Sharpe 0.17 (랜덤과 구분 불가), 스크리너만(55.8%) > 모델(53.1%) 결과에 기반한 3-Phase 계획.

### Phase 1 — Foundation (병렬 실행, 1-2일)

| # | 작업 | 파일 | 검증 |
|---|------|------|------|
| 1A | `fillna(0)` → `ffill().bfill()` + 첫 480행 drop | `data/preprocessor.py` | `assert (df['rsi'] >= 1).all()` |
| 1B | Train/Test temporal split (마지막 20% holdout) | `training/trainer.py`, `training/evaluator.py` | holdout 메트릭 별도 출력 |
| 1C | 예측 호라이즌 3h → 12h | `utils/config.py`, `data/preprocessor.py`, `inference/predictor.py` | 12h label AC ≥ 0.08 |

### Phase 2 — Model Structure (Phase 1 후, 1-2일)

| # | 작업 | 파일 | 검증 |
|---|------|------|------|
| 2A | CVAE → heteroscedastic dual head (mu, log_sigma) | `models/hybrid_model.py`, `models/cvae_decoder.py`, `training/trainer.py` | log_var std > 0.01 |
| 2B | confidence = f(log_var), 상수 0.503 해소 | `models/hybrid_model.py`, `inference/predictor.py` | confidence std > 0.05 |

### Phase 3 — Execution Layer (Phase 2 후, 1일)

| # | 작업 | 파일 | 검증 |
|---|------|------|------|
| 3A | Position sizing: uncertainty-aware (상수 6.6% → 동적) | `inference/recommender.py` | position size std > 0 |
| 3B | Short bias 완화, long 추천 비율 확보 | `inference/recommender.py` | long:short 30:70 ~ 70:30 |

### Phase 4 — Calibration (30일 운영 후)

| # | 작업 | 파일 |
|---|------|------|
| 4A | 앙상블 lookback 설계대�� 복원 (8/24/72/48h) | `models/ensemble_configs.json` |
| 4B | Screener vs Model A/B 비교 | `inference/recommender.py` |

### Rollback 기준
- Phase 1 후 holdout loss 20%↑ → fillna 전략 완화
- Phase 2 후 holdout 악화 → CVAE 유지, dual head만 추가
- Phase 3 후 추천 0건 3일 연속 → threshold 완화

---

## 12. 확인된 근본 문제 (2026-03-28 기록)

| 문제 | 증거 | 영향 |
|------|------|------|
| 3h AC=0.025 (노이즈) | 12h AC=0.100 (p=0.00015) | 예측 불가능한 timescale에서 학습 중 |
| KL=0 (posterior collapse) | 100 epoch 전체 kl=0.0000 | CVAE 잠재공간 미사용, 비싼 autoencoder |
| fillna(0) | RSI=0, volatility=0 불가능 값 | 첫 480행 + 중간 누락 오염 |
| train/test 100% overlap | 학습 720일 ⊃ 백테스트 90일 | 모든 보고 메트릭 무효 |
| confidence_head 상수 출력 | mean=0.503, std=0.0008 | position sizing에 정보 없음 |
| 스크리너 > 모델 | 55.8% vs 53.1% | 모델이 가치를 파괴하고 있음 |
| 98% short + short 차단 | live에서 거의 전부 watch-only | 실행 가능한 추천 거의 0 |

---

## 13. 알려진 드리프트

- 일부 문서가 여전히 6-step horizon 기준으로 서술되어 있으나, 런타임 config는 현재 **3-step**이고 **12-step으로 전환 예정**.
- 일부 문서가 ops-first 우선순위보다 논문 중심 어조로 되어 있다. 새 문서는 라이브 런타임 기준으로 먼저 쓴다.
- 코드와 문서가 충돌하면 코드를 따르고, `aaa/`는 논문 트랙 전용으로 취급한다.
- `PROJECT_ARCHITECTURE.md §4.1`은 residual target을 명시하지만, 코드는 raw return. **현재 raw return 유지, Phase 4에서 A/B 비교 후 결정.**

---

## 메모리 업데이트 규칙

- 같은 실수를 두 번 이상 반복했을 때만 이 파일에 규칙을 추가한다.
- 추가할 때는: 현상 / 재발 방지 규칙 / 검증 방법 형식으로 짧게.
