# Chrono-Trader v3.2: Explainable Hybrid Framework for Crypto Prediction

---

## 1. System Overview

**Chrono-Trader v3.2** is an advanced algorithmic trading system designed for the cryptocurrency market. It addresses the key challenge of "black-box" AI models by implementing a fully **explainable architecture** that combines long-term trend analysis (Transformer) with short-term pattern recognition (CNN/TCN).

Key innovations in v3.2:
1.  **Context-Aware Time Perception**: Unlike standard models, it perceives time relative to market conditions (Contextual Positional Encoding).
2.  **Explainable Decision Making**: It explicitly compares current partial patterns to a learned "Prototype Bank" of historical success patterns.
3.  **Uncertainty Quantification**: It provides a confidence score based on the agreement of an ensemble of 5 diverse models.

---

## 2. Core Architecture: The Hybrid Model

The heart of the system is the `HybridModel` class, which processes a sequence of **168 hours (7 days)** of market data to predict the return over the next **6 hours**.

```mermaid
graph TD
    Input[Input Sequence<br>168 hours x 19 features] --> Enc[Encoders];
    
    subgraph "Global Path (Transformer)"
        Enc --> T_PE[Contextual Positional Encoding];
        T_PE -- "Context: Market Index + Similarity" --> T_Attn[Self-Attention Layers];
        T_Attn --> T_Out[Global Features];
    end
    
    subgraph "Local Path (CNN)"
        Enc --> C_Conv[1D TCN / 2D GAF];
        C_Conv --> C_Out[Local Features];
    end
    
    subgraph "Explainable Fusion"
        T_Out & C_Out --> Sim[Similarity Check];
        Sim -- "Compare vs" --> Proto[Prototype Bank<br>(16 Success Patterns)];
        Sim --> Gate[Dynamic Gating<br>σ(Global, Local, Sim_Score)];
        Gate --> Fused[Fused Context];
    end
    
    subgraph "Generative Decoder"
        Fused --> G_Dec[GAN Decoder];
        Noise[Noise Vector z] --> G_Dec;
        G_Dec --> Pred[Price Prediction];
    end

    style Input fill:#f9f,stroke:#333
    style Proto fill:#ff9,stroke:#f66
    style Pred fill:#9f9,stroke:#333
```

### 2.1. Input Features (19 Dimensions)

The model consumes 19 engineered features for every time step:

| Index | Feature | Description |
|---|---|---|
| 0 | `close` | Normalized closing price |
| 1 | `volume` | Trading volume |
| 2 | `rsi` | Relative Strength Index (14) |
| 3-5 | `macd`, `sig`, `hist` | MACD components |
| 6 | `adx` | Trend strength indicator |
| 7 | `obv` | On-Balance Volume |
| **8** | **`market_index`** | **Context Signal**: Weighted return of BTC+ETH |
| **9** | **`hist_sim`** | **Context Signal**: Similarity to past recurring patterns |
| 10-12 | `bb_upper/mid/low` | Bollinger Bands |
| 13 | `volume_ma` | 20-day Volume MA |
| 14-16 | `volatility` | Volatility at 24h, 7d, and volume-based |
| 17-18 | `alpha`, `beta` | Market-relative performance metrics |

### 2.2. Contextual Positional Encoding (CPE)

Standard Transformers use fixed sine/cosine waves for time. Our **CPE** module modifies this:
-   **Concept**: Time flows "differently" during a crash vs. a bull run.
-   **Implementation**: A linear projection of `[market_index, hist_sim]` is added to the positional embeddings.
-   **Effect**: The model learns to attend to "similar market conditions" rather than just "similar time lags".

### 2.3. Explainable Gated Fusion

This module solves the "Transformer vs. CNN" debate by dynamically choosing the best tool:
1.  **Prototype Bank**: The model learns 16 "canonical" market patterns (prototypes) during training.
2.  **Similarity Matching**:
    -   It compares the Transformer's output to the prototypes.
    -   It compares the CNN's output to the prototypes.
3.  **Decision**:
    -   If Transformer output matches a known success pattern better → Gate favors Transformer.
    -   If CNN output matches better → Gate favors CNN.
    -   **Result**: We can inspect *why* the model chose a specific path (e.g., "Transformer matched Prototype #5 (Bull Flag)").

---

## 3. Explainability & Analysis Tools

The system provides built-in tools to dissect model decisions:

### 3.1. Attention Maps
-   **What**: Heatmaps showing which past time steps the Transformer focused on.
-   **Usage**: Identify if the model is looking at recent momentum (last 24h) or weekly structural support (last 7d).

### 3.2. Prototype Visualization
-   **What**: t-SNE projection of the 16 learned prototypes.
-   **Usage**: Understand what the model considers "important structure".

### 3.3. Uncertainty (MC-Dropout)
-   **What**: Running the model 20 times with different noise/dropout masks.
-   **Usage**:
    -   **Low Spread**: High confidence (Trade!).
    -   **High Spread**: Model is guessing (Stay Cash).

---

## 4. Execution Pipeline

1.  **Data Collection**: 5-minute intervals, raw data from Upbit.
2.  **Preprocessing**: Indicators, Z-score normalization.
3.  **Inference (Ensemble)**:
    -   5 independent models predict.
    -   Results are averaged.
    -   Spread determines uncertainty.
4.  **Filtering**:
    -   `Uncertainty > Threshold`: Reject.
    -   `Expected Return < 1.0%`: Reject.
    -   `Liquidity < Top 20%`: Reject.

---

## 5. Development History

-   **v1.0 (Legacy)**: Single model, BTC-only, no context. Unprofitable.
-   **v2.0 (Stable)**: BTC+ETH training, Hybrid architecture. Profitable (+1.3 Sharpe).
-   **v3.2 (Current)**: Explainable Fusion, 19 Features, Research Lab Integration.
