"""
MC Dropout 진단 스크립트
- 실제로 dropout이 예측 다양성을 만드는지 확인
- Uncertainty와 Error의 관계 시각화
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import config
from utils.logger import logger
from data.preprocessor import get_processed_data_for_training, get_market_index

def diagnose_mc_dropout():
    logger.info("=== MC Dropout 진단 시작 ===")
    
    # 1. 모델 로드
    model_path = config.MODEL_PATH
    try:
        model = torch.load(model_path, map_location=config.DEVICE, weights_only=False)
        logger.info(f"모델 로드 성공: {model_path}")
    except Exception as e:
        logger.error(f"모델 로드 실패: {e}")
        return
    
    # 2. 검증 데이터 준비
    market_index_df = get_market_index()
    X, y, _ = get_processed_data_for_training(config.TARGET_MARKETS[0], market_index_df)
    
    if X is None:
        logger.error("데이터 로드 실패")
        return
    
    # 마지막 100개 샘플만 사용
    X_test = X[-100:]
    y_test = y[-100:]
    
    logger.info(f"테스트 샘플 수: {len(X_test)}")
    
    # 3. MC Dropout으로 예측 (30회)
    N_INFERENCES = 30
    model.train()  # Dropout 활성화
    
    # BatchNorm은 eval 모드로
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm1d):
            module.eval()
    
    all_predictions = []
    all_uncertainties = []
    all_errors = []
    
    logger.info("MC Dropout 예측 시작...")
    
    with torch.no_grad():
        for i in range(len(X_test)):
            x_sample = torch.FloatTensor(X_test[i:i+1]).to(config.DEVICE)
            y_true = y_test[i]
            
            # 30번 추론
            mc_preds = []
            for _ in range(N_INFERENCES):
                pred = model(x_sample)[0].cpu().numpy()
                mc_preds.append(pred)
            
            mc_preds = np.array(mc_preds)
            
            # 통계량 계산
            mean_pred = mc_preds.mean(axis=0)
            std_pred = mc_preds.std(axis=0)
            uncertainty = np.sum(std_pred)  # 총 불확실성
            error = np.mean(np.abs(mean_pred - y_true))  # MAE
            
            all_predictions.append(mc_preds)
            all_uncertainties.append(uncertainty)
            all_errors.append(error)
            
            if i % 20 == 0:
                logger.info(f"[{i}/{len(X_test)}] Uncertainty: {uncertainty:.4f}, Error: {error:.4f}")
    
    # 4. 통계 분석
    all_uncertainties = np.array(all_uncertainties)
    all_errors = np.array(all_errors)
    
    logger.info("\n=== 진단 결과 ===")
    logger.info(f"Uncertainty - Mean: {all_uncertainties.mean():.4f}, Std: {all_uncertainties.std():.4f}")
    logger.info(f"Uncertainty - Min: {all_uncertainties.min():.4f}, Max: {all_uncertainties.max():.4f}")
    logger.info(f"Error - Mean: {all_errors.mean():.4f}, Std: {all_errors.std():.4f}")
    
    # Correlation 계산
    from scipy.stats import spearmanr, pearsonr
    spearman_corr, spearman_p = spearmanr(all_errors, all_uncertainties)
    pearson_corr, pearson_p = pearsonr(all_errors, all_uncertainties)
    
    logger.info(f"\n상관관계 분석:")
    logger.info(f"Spearman Correlation: {spearman_corr:.4f} (p={spearman_p:.4f})")
    logger.info(f"Pearson Correlation: {pearson_corr:.4f} (p={pearson_p:.4f})")
    
    if spearman_corr < 0.3:
        logger.warning("⚠️  상관관계가 너무 낮습니다! MC Dropout이 제대로 작동하지 않을 수 있습니다.")
    elif spearman_corr < 0.5:
        logger.info("✓ 약한 상관관계 존재. 개선 여지가 있습니다.")
    else:
        logger.info("✅ 강한 상관관계! MC Dropout이 잘 작동하고 있습니다.")
    
    # 5. 시각화
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # (1) Scatter plot: Uncertainty vs Error
    axes[0, 0].scatter(all_uncertainties, all_errors, alpha=0.6)
    axes[0, 0].set_xlabel('Uncertainty (sum of std)')
    axes[0, 0].set_ylabel('Error (MAE)')
    axes[0, 0].set_title(f'Uncertainty vs Error (r={spearman_corr:.3f})')
    
    # 추세선 추가
    z = np.polyfit(all_uncertainties, all_errors, 1)
    p = np.poly1d(z)
    axes[0, 0].plot(all_uncertainties, p(all_uncertainties), "r--", alpha=0.8)
    
    # (2) Uncertainty 분포
    axes[0, 1].hist(all_uncertainties, bins=30, edgecolor='black')
    axes[0, 1].set_xlabel('Uncertainty')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Uncertainty Distribution')
    axes[0, 1].axvline(all_uncertainties.mean(), color='r', linestyle='--', label='Mean')
    axes[0, 1].legend()
    
    # (3) Error 분포
    axes[1, 0].hist(all_errors, bins=30, edgecolor='black')
    axes[1, 0].set_xlabel('Error (MAE)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Error Distribution')
    axes[1, 0].axvline(all_errors.mean(), color='r', linestyle='--', label='Mean')
    axes[1, 0].legend()
    
    # (4) 예시: 한 샘플의 30번 예측 분포
    sample_idx = np.argmax(all_uncertainties)  # 가장 불확실한 샘플
    sample_preds = all_predictions[sample_idx]
    
    # 6시간 각각의 예측 분포
    for t in range(6):
        axes[1, 1].scatter([t]*N_INFERENCES, sample_preds[:, t], alpha=0.3, s=20)
    
    # 평균과 표준편차
    mean_per_time = sample_preds.mean(axis=0)
    std_per_time = sample_preds.std(axis=0)
    axes[1, 1].plot(range(6), mean_per_time, 'r-', linewidth=2, label='Mean')
    axes[1, 1].fill_between(
        range(6), 
        mean_per_time - std_per_time, 
        mean_per_time + std_per_time,
        alpha=0.3, color='red', label='±1 std'
    )
    axes[1, 1].plot(range(6), y_test[sample_idx], 'g--', linewidth=2, label='True')
    axes[1, 1].set_xlabel('Time (hours)')
    axes[1, 1].set_ylabel('Predicted % change')
    axes[1, 1].set_title(f'Sample with Highest Uncertainty (uncertainty={all_uncertainties[sample_idx]:.4f})')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = "analysis/mc_dropout_diagnosis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"\n시각화 저장 완료: {output_path}")
    
    # 6. 추가 진단: Dropout이 실제로 켜져있는지 확인
    logger.info("\n=== Dropout 레이어 상태 확인 ===")
    dropout_layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout):
            dropout_layers.append((name, module.p, module.training))
    
    if not dropout_layers:
        logger.error("❌ Dropout 레이어를 찾을 수 없습니다!")
    else:
        logger.info(f"발견된 Dropout 레이어 수: {len(dropout_layers)}")
        for name, p, is_training in dropout_layers:
            status = "활성화" if is_training else "비활성화"
            logger.info(f"  - {name}: p={p:.3f}, {status}")

if __name__ == "__main__":
    diagnose_mc_dropout()
