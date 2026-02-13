
import sys
import os
import torch
import numpy as np
import pandas as pd
import json
from datetime import datetime

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import config
from utils.logger import logger
from inference.predictor import _get_cached_models
from data.preprocessor import get_intermediate_data, create_final_sequences_and_scale, get_market_index

def generate_report():
    print(">>> Loading Models...")
    models = _get_cached_models()
    if not models:
        print("CRITICAL: No models found!")
        return

    # Load Model Metadata (Role, Name)
    try:
        with open("models/ensemble_configs.json", "r") as f:
            ensemble_config = json.load(f)
            model_info_map = {m['id']: m for m in ensemble_config['models']}
    except:
        model_info_map = {}

    target_market = "KRW-BTC" # Standard Representative
    print(f">>> Analyzing {target_market}...")

    # Load Data
    market_index_df = get_market_index()
    df, scaler = get_intermediate_data(target_market, market_index_df)
    
    if df is None:
        print(f"Failed to load data for {target_market}")
        return

    # Prepare specific sequence
    X, y, scaler = create_final_sequences_and_scale(df, scaler)
    last_sequence = X[-1]
    sequence_tensor = torch.FloatTensor([last_sequence]).to(config.Device.DEVICE)
    current_price = df.iloc[-1]['close']

    # --- Run Ensemble Inference ---
    model_opinions = []

    for i, model in enumerate(models):
        model.eval()
        model_id = i + 1
        info = model_info_map.get(model_id, {"name": f"Model {model_id}", "role": "Unknown"})
        
        # Inference with Explainability
        try:
            # Check if model supports explainability
            prediction, explain_dict = model(sequence_tensor, return_explainability=True)
            
            # Extract metrics
            pred_pattern = prediction.detach().cpu().numpy().flatten()
            direction = "Short" if pred_pattern[-1] < pred_pattern[0] else "Long"
            magnitude = abs(pred_pattern[-1] - pred_pattern[0])
            
            gate_info = explain_dict.get('gate_info', {})
            if isinstance(gate_info, dict):
                gate_val = float(np.mean(gate_info.get('gate_values', 0.5)))
            else:
                gate_val = 0.5
            
            # Interpret Gate
            focus = "Macro (Transformer)" if gate_val > 0.6 else ("Micro (CNN)" if gate_val < 0.4 else "Hybrid")
            
            model_opinions.append({
                "id": model_id,
                "name": info.get('name', f"Model {model_id}"),
                "direction": direction,
                "magnitude": magnitude,
                "gate_val": gate_val,
                "focus": focus,
                "confidence": "High" if magnitude > 0.5 else "Low" # Simplified
            })

        except Exception as e:
            print(f"Error inferencing Model {model_id}: {e}")

    # --- Generate Markdown Report ---
    report_path = "ENSEMBLE_ANALYSIS_REPORT.md"
    
    with open(report_path, "w") as f:
        f.write(f"# 🧠 Deep Agent Ensemble Analysis Report\n")
        f.write(f"**Target**: {target_market} | **Time**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        
        f.write("## 1. Executive Summary (요약)\n")
        longs = [m for m in model_opinions if m['direction'] == 'Long']
        shorts = [m for m in model_opinions if m['direction'] == 'Short']
        
        consensus_score = max(len(longs), len(shorts)) / len(models) * 100
        direction_str = "Bullish (상승)" if len(longs) > len(shorts) else "Bearish (하락)"
        
        f.write(f"- **Consensus**: {direction_str} ({consensus_score:.0f}% Agreement)\n")
        f.write(f"- **Current Price**: {current_price:,.2f}\n")
        f.write(f"- **Strategy**: {'Aggressive' if consensus_score > 80 else 'Conservative'}\n\n")

        f.write("## 2. Model Roundtable (모델별 상세 의견)\n")
        f.write("각 모델은 서로 다른 데이터 특성(미시적 패턴 vs 거시적 트렌드)에 집중하도록 설계되었습니다.\n\n")

        for op in model_opinions:
            icon = "🐂" if op['direction'] == 'Long' else "🐻"
            f.write(f"### {icon} {op['name']}\n")
            f.write(f"- **Position**: **{op['direction']}**\n")
            f.write(f"- **Focus**: {op['focus']} (Gate: {op['gate_val']:.2f})\n")
            f.write(f"- **Reasoning**: 이 모델은 현재 **{op['focus']}** 정보에 {op['gate_val']*100:.0f}% 비중을 두고 있습니다. ")
            if op['focus'] == "Macro (Transformer)":
                f.write("전체적인 시장 흐름과 과거 유사 패턴(S&P500 등)을 기반으로 판단했습니다.\n")
            elif op['focus'] == "Micro (CNN)":
                f.write("최근 캔들의 변동성과 단기적인 기술적 패턴을 기반으로 판단했습니다.\n")
            else:
                f.write("장기 트렌드와 단기 패턴을 균형 있게 고려하여 판단했습니다.\n")
            f.write("\n")

        f.write("## 3. Synthesis & Recommendation (최종 결론)\n")
        if consensus_score >= 80:
             f.write("> [!IMPORTANT]\n")
             f.write("> **Strong Signal**: 모델 간 합의가 강력합니다. 진입을 적극 고려하십시오.\n")
        elif consensus_score >= 60:
             f.write("> [!NOTE]\n")
             f.write("> **Moderate Signal**: 방향성은 보이지만 모델 간 이견이 존재합니다. 분할 진입을 권장합니다.\n")
        else:
             f.write("> [!WARNING]\n")
             f.write("> **Weak/Confused Signal**: 모델 간 의견이 엇갈립니다. 관망하거나 변동성 돌파 전략을 사용하십시오.\n")

    print(f"Report generated at {report_path}")

if __name__ == "__main__":
    generate_report()
