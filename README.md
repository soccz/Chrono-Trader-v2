# 🤖 Chrono-Trader v2: Development Roadmap & Action Plan

---

## 1. Current Status (2025-11-06)

This document outlines the strategic roadmap for Chrono-Trader v2. The project has successfully evolved from an unprofitable prototype to a stable, profitable baseline model through systematic debugging, data expansion, and architectural refactoring.

### 1.1. Current Baseline Model

-   **Training Data:** 365 days of hourly data for `KRW-BTC` and `KRW-ETH`.
-   **Architecture:** Ensemble of GAN-Transformer models using hyperparameters from `models/model_config.json`.
-   **Key Logic:** Pre-trained on the combined dataset, fine-tuned on live trending markets.

### 1.2. v2.0 Performance Baseline

The following metrics represent the current, official performance baseline for the v2 model, established via a 30-day backtest. All future improvements will be measured against this baseline.

| Metric | **v2.0 Baseline (BTC+ETH)** |
| :--- | :--- |
| **Win Rate** | **52.14%** |
| **Avg Return/Trade** | **+0.20%** |
| **Sharpe Ratio (Ann.)** | **+1.39** |
| **Max Drawdown (MDD)** | **-11.85%** |
| **Uncertainty Correlation** | **0.1204** |

### 1.3. Sample Daily Output

The model is confirmed to be generating valid recommendations for both trend/pattern and pump detection systems.

-   **Trend/Pattern Recommendations (2025-11-06):** 2 successful recommendations (`KRW-SIGN`, `KRW-IP`).
-   **Pump Predictions (2025-11-06):** 2 successful candidates (`KRW-W`, `KRW-ME`).

---

## 2. Immediate Next Step: Hyperparameter Optimization

**Goal:** The current model, while profitable, uses hyperparameters that were tuned on a much smaller, older dataset. To unlock the full potential of our new 365-day BTC+ETH dataset, we must find the optimal set of hyperparameters.

**Action:** Run the Optuna hyperparameter tuning process. This is a **very time-consuming task** but is the most critical step for performance maximization.

**Command:**
```bash
python main.py --mode train --tune
```

**Expected Outcome:** A new `models/model_config.json` file will be generated, containing the best-performing parameters (e.g., `learning_rate`, `dropout_p`, `n_layers`, `lambda_recon`) for the new dataset. This will serve as the foundation for the `v2.1` model.

---

## 3. Standard Development Workflow

After the initial tuning is complete, all future model improvements (e.g., adding new assets, changing model architecture) should follow this cycle to ensure rigorous, quantitative evaluation.

1.  **Implement Changes:** Modify the code (e.g., add `KRW-SOL` to `TARGET_MARKETS` in `config.py`).
2.  **Re-Train Base Model:** Run `python main.py --mode train` to create a new base model using the updated configuration/data.
3.  **Run Backtest & Establish New Baseline:** Execute `python main.py --mode backtest --days 30` to get the performance metrics for the new model.
4.  **Analyze & Compare:** Compare the new backtest results against the previous baseline to determine if the change was beneficial.
5.  **Commit & Document:** If the change is successful, commit it to the repository and update the documentation.

---

## 4. Long-Term Improvement Roadmap

Once a satisfactory v2.1 baseline is established post-tuning, the following long-term enhancements can be pursued.

### 4.1. Gradual Market Expansion

-   **Objective:** Increase the diversity of the training data to capture a wider range of market behaviors.
-   **Action:** Sequentially add other high-liquidity major altcoins (e.g., `KRW-SOL`, `KRW-XRP`, `KRW-ADA`) to the `TARGET_MARKETS` list in `config.py`. After each addition, follow the **Standard Development Workflow** (Step 3) to train and evaluate the performance impact.

### 4.2. Advanced Modeling Strategies

-   **Objective:** Move beyond a single base model to capture coin-specific characteristics.
-   **Action 1 (Coin-Specific Models):** Refactor the pipeline to train and manage separate base models for each individual target asset. This would allow for more specialized predictions but requires significant changes to the training and inference logic.
-   **Action 2 (Cluster-Based Models):** Implement a clustering algorithm (e.g., based on market cap, sector like DeFi/AI/GameFi) to group similar assets. Train a separate base model for each cluster, creating a hybrid approach between a single model and fully coin-specific models.

### 4.3. Automation & Deployment

-   **Objective:** Transition the system from a manually-run script to a fully automated MLOps pipeline and user-facing application.
-   **Action 1 (Automated Execution):** Use a scheduler like **Cron** (on a dedicated server) or a cloud-native service (e.g., **AWS EventBridge, Google Cloud Scheduler**) to run the `main.py --mode daily` script automatically at a set time each day.
-   **Action 2 (Web Dashboard):** Develop a simple web application using a Python backend framework (**Flask** or **FastAPI**) to serve the latest results from the `recommendations` and `predictions` directories. A simple HTML/JavaScript frontend with a library like **Bootstrap** can be used to display the data in a clean, auto-refreshing table format.
