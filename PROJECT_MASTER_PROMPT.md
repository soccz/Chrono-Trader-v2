# PROJECT MASTER PROMPT (절대 기준)

이 파일은 **AETHER: Chrono-Trader** 프로젝트의 헌법입니다.
모든 개발, 기획, 디자인 작업은 이 문서에 정의된 원칙을 **절대적으로 준수**해야 합니다.
AI 엔지니어는 작업 시작 전 반드시 이 파일을 정독하고, 스스로의 추론 과정(CoT)에 이 기준을 강제 적용해야 합니다.

---

## 1. 🧠 Core Philosophy & Motivation (핵심 철학)

### 1-1. Market Dynamics (시장 역학의 이해)
- **Non-Markovian Modeling (비마르코프적 역학)**:
    - 암호화폐 시장은 현재 상태(State)만으로 미래가 결정되지 않습니다. 전체 시계열의 맥락(Path-dependency)이 중요합니다.
    - 단순한 Price Action이 아니라, **"이 시점의 시장 분위기가 과거(예: S&P500 역사적 붕괴/상승기)와 얼마나 닮았는가?"**를 비교해야 합니다.
- **Uncertainty First (불확실성 우선)**:
    - 점 추정(Point Prediction)은 무의미합니다. **GAN의 생성적 특성**과 **MC-Dropout**을 통해 예측의 불확실성 범위(Confidence Interval)를 시각화하고, 이를 트레이딩의 근거로 삼아야 합니다.

### 1-2. User Experience (사용자 경험)
- **Deep Space Glass (심우주 유리)**:
    - 사용자는 '단순한 웹사이트'가 아니라 **'고도로 진보된 외계 기술의 인터페이스'**를 조작하는 느낌을 받아야 합니다.
    - **Premium Only**: 싼 티 나는 디자인(기본 부트스트랩, 생상된 원색)은 죄악입니다. 
    - **Alive System**: 정적인 페이지는 죽은 것입니다. 데이터는 실시간으로 흐르고, UI는 미세하게 숨쉬어야 합니다 (Micro-animations).

---

## 2. 🏗 Architecture Standards (아키텍처 절대 기준)

> **상세 정의서**: [PROJECT_ARCHITECTURE.md](./PROJECT_ARCHITECTURE.md) 참조 (수치는 여기가 정본)

### 2-1. Hybrid Model Structure (모델 구조)
모델은 반드시 다음 요소의 **앙상블 및 상호 검증** 구조를 갖춰야 합니다.

1.  **Global Attention (Transformer Encoder)**:
    - **역할**: 168시간(7일) 전체 시계열의 거시적 흐름(Trend) 파악.
    - **핵심 로직**: `Contextual Positional Encoding`을 통해 시장 지수(`market_index_return`)와 과거 유사도(`historical_similarity`)를 시간 인코딩에 주입.
2.  **Local Pattern (CNN)**:
    - **역할**: 미시적인 캔들 패턴(Micro-structure) 포착.
    - **핵심 로직**: 1D TCN (모델 1,2,3,5) 또는 2D GAF 변환(모델 4).
3.  **Explainable Gating (설명 가능한 통합)**:
    - **역할**: Prototype Bank 기반으로 Transformer vs CNN 비중 결정.
    - **출력**: Gate Value → "현재는 거시적 트렌드(80%)가 미시적 패턴(20%)보다 우세합니다."
4.  **GAN Decoder + Critic (WGAN-GP)**:
    - 5가지 Loss로 학습: WGAN-GP, Reconstruction(MSE), ECE, Direction Balance, Gate Regularization.
    - 동적 가중치 조정 + Auto-Stop 규칙.

### 2-2. Inference Pipeline (추론)
-  **MC-Dropout**: 5 앙상블 모델 × 20회 추론 = 100개 예측 → 가중 평균.
-  **Shrunk Beta**: `0.5×coin + 0.5×cross_mean` → 유동성 낮은 코인 노이즈 감소.
-  **Weighted Voting**: `ModelPerformanceTracker`의 rolling accuracy 기반.

### 2-3. Recommendation Funnel (추천)
-  **6단계 필터링**: Tradeable → 방향일관(66%) → 체제/Lead-Lag → 유동성 → 수익률 → 불확실성 → DTW.
-  **Forced Fallback**: 최소 3개 추천 보장.

### 2-4. Financial Logic (금융 로직)
- **FF3 (Fama-French 3 Factor) 적용**:
    - 모든 성과 분석은 단순히 수익률(Return)만 보지 않고, **시장 초과 수익(Alpha)**과 **베타(Beta)**를 분리하여 평가해야 합니다.
    - "시장이 좋아서 번 돈"과 "실력으로 번 돈"을 구분하십시오.
- **Risk Management**:
    - `0` 나누기 오류, `NaN` 데이터, `Infinity` 값은 금융 시스템에서 발생하면 즉시 파산입니다.
    - **Backend**: `NaNSafeJSONProvider`를 통해 API 단에서 `NaN`을 `null`로 변환하여 송출하십시오.
    - **Frontend**: 데이터가 `null`일 경우 `0`이나 `-`으로 표시하지 말고, 로딩 스피너나 "N/A" 처리를 통해 데이터 부재를 명확히 알리십시오.

---

## 3. 🛡 Development Protocol (개발 프로토콜)

### 3-1. "Verify Before Edit" (수정 전 확인)
- **원칙**: 코드를 1줄이라도 고치기 전에, 그 코드가 영향을 미치는 **전체 파일**과 **Import 관계**를 파악하십시오.
- **금지**: "기존 코드가 기억 안 나서 덮어썼습니다" -> **절대 금지**. `view_file`로 먼저 읽으십시오.

### 3-2. "No Regressions" (기능 퇴행 방지)
- 새로운 기능을 추가한다고 기존 기능(예: 백테스트 그래프, 실시간 소켓 연결)이 깨지면 안 됩니다.
- 특히 `app.py`나 `static/js/` 같은 공용 파일을 건드릴 때는 사이드 이펙트를 3번 고민하십시오.

### 3-3. "Explicit Error Handling" (명시적 에러 처리)
- 백엔드 오류(500)가 발생했을 때 프론트엔드가 멈춰 있으면 안 됩니다.
- 사용자에게 "서버 연결 중..." 또는 "데이터 로드 실패" 같은 피드백을 즉시 제공해야 합니다.

---

## 4. 🎨 Design Guidelines (Deep Space Glass)

| 요소 | 규칙 | 예시 값 |
| :--- | :--- | :--- |
| **Colors** | Neon Accents on Deep Dark Background | BG: `#050507`, Accent: `#00f2ff` (Trend), `#d946ef` (Pattern) |
| **Surfaces** | Glassmorphism Hierarchy | `backdrop-filter: blur(20px)`, `border: 1px solid rgba(255,255,255,0.08)` |
| **Typography** | Inter, Tabular Nums for Data | `font-variant-numeric: tabular-nums` (숫자가 춤추지 않게) |
| **Motion** | Smooth Transitions, No Jumps | `transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1)` |

---

## 5. 📝 Immediate Action Items (즉시 해결 과제)

1.  **자동화 스케줄링**: systemd(user timers) + `scripts/run_scheduled.py` 기반으로 `refresh-db` → `intraday`/`morning-report` 운영 자동화.
2.  **System Stability**: `NaNSafeJSONProvider`가 모든 API 응답에 적용되어 있는지 전수 조사.
3.  **Data Consistency**: 백엔드(`entry_time`)와 프론트엔드(`entry_date`)의 키 값 불일치 전수 조사 및 통일.
4.  **Performance**: 메인 대시보드 로딩 속도 1초 미만 목표 (CSV 로딩 최적화, 캐싱).
5.  **문서 동기화**: 코드 변경 시 `PROJECT_ARCHITECTURE.md`와 `.agent/project_context.md` 반드시 업데이트.

---
**이 문서는 프로젝트의 법입니다. 이 내용을 벗어난 제안이나 코드는 기각하십시오.**
