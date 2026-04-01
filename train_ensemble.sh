#!/bin/bash
# =============================================================================
# Diverse Ensemble Training Script
# Run this with nohup or screen to train in the background
# =============================================================================
# Usage:
#   nohup ./train_ensemble.sh > logs/training_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#   OR use screen:
#   screen -S training
#   ./train_ensemble.sh
# =============================================================================

set -e

PROJECT_DIR="/home/soccz/.gemini/antigravity/scratch/mnt_20t/main/gan_t"
cd "$PROJECT_DIR"

# Create logs directory if not exists
mkdir -p logs

# Use system python3 (PyTorch with CUDA is installed system-wide)
PYTHON="python3"
$PYTHON --version

# Print GPU info
echo "============================================="
echo "Starting Diverse Ensemble Training"
echo "Time: $(date +'%Y-%m-%d %H:%M:%S')"
echo "============================================="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv || echo "GPU info not available"
echo ""

# Backup existing models (optional safety)
BACKUP_DIR="models/backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
cp models/model_*.pth "$BACKUP_DIR/" 2>/dev/null && echo "Backed up existing models to $BACKUP_DIR" || echo "No existing models to backup"

# Show ensemble config
echo ""
echo "Ensemble Configuration:"
cat models/ensemble_configs.json
echo ""

# Run training (full training mode, 100 epochs)
echo "============================================="
echo "Starting Training..."
echo "============================================="
$PYTHON main.py --mode train --epochs 100

# Send Telegram notification when done
echo ""
echo "============================================="
echo "Training Completed: $(date +'%Y-%m-%d %H:%M:%S')"
echo "============================================="

# Send Telegram notification
$PYTHON -c "
from utils.telegram_bot import send_alert
import os
msg = '''🎉 *다양한 앙상블 학습 완료!*

✅ 5개 모델 학습 성공
📍 학습 위치: $(hostname)
🕐 완료 시간: $(date +'%Y-%m-%d %H:%M:%S')

*다음 단계:*
- 새 모델로 예측 시작 가능
- \`--mode daily\` 또는 \`--mode continuous\` 실행
'''
send_alert(msg, bypass_dedup=True)
print('Telegram notification sent!')
" 2>/dev/null || echo "Telegram notification failed (not critical)"

# Check new model files
echo ""
echo "New model files:"
ls -la models/model_*.pth

echo ""
echo "Done! Models saved to models/"
