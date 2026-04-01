# 펌핑 예측 모델(XGBoost) 연구 및 발전 계획

본 문서는 `pump_classifier.joblib` (XGBoost) 모델의 성능을 분석하고 개선하기 위한 연구 및 실험 계획을 기록합니다.

## Phase 1: 현 모델 분석 (Understanding the Current Model)

-   **목표:** 현재 모델이 어떤 특징(Feature)을 기반으로 예측을 수행하는지 이해한다.
-   **실행 항목:**
    1.  **특징 중요도(Feature Importance) 분석:** 모델을 학습한 후, 어떤 특징이 예측에 가장 큰 영향을 미쳤는지 순위를 매겨 확인한다. 이를 통해 모델의 기본적인 판단 기준을 파악할 수 있다.

## Phase 2: 성능 최적화 (Performance Optimization)

-   **목표:** 모델의 예측 정확도, 정밀도, 재현율 등 전반적인 성능을 향상시킨다.
-   **실행 항목:**
    1.  **하이퍼파라미터 튜닝(Hyperparameter Tuning):** 현재 하드코딩된 `n_estimators`, `max_depth` 등의 파라미터를 Optuna와 같은 라이브러리를 사용하여 최적의 조합을 탐색한다.
    2.  **신규 특징 추가(New Feature Engineering):**
        *   **변동성 관련 지표:** ATR(Average True Range) 등 변동성 관련 지표를 추가하여 '조용한 상태'와 '폭발 직전'을 더 정교하게 구분한다.
        *   **거래대금 관련 지표:** 단순 거래량이 아닌, 거래대금의 변화량을 추적하여 더 의미 있는 자금의 유입을 포착한다.

## Phase 3: 예측 결과 해석력 강화 (Enhancing Explainability)

-   **목표:** 특정 예측(예: `KRW-API3` 급등 예측)이 왜 그렇게 나왔는지 개별 사례에 대해 심층적으로 이해한다.
-   **실행 항목:**
    1.  **SHAP(SHapley Additive exPlanations) 분석 도입:** `analysis/explainability.py`에 SHAP 분석 기능을 추가하여, 특정 예측 결과에 각 특징이 얼마나, 그리고 긍정적/부정적으로 기여했는지를 시각화하고 분석한다.

## 실험 기록

*(이곳에 Phase 1, 2, 3의 각 실행 항목에 대한 실험 결과와 분석 내용을 순차적으로 기록한다.)*