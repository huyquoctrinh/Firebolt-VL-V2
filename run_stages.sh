#!/usr/bin/env bash
# Automated: wait for current training to finish, then run stage1 + stage2
set -e

PROJECT_DIR="/home/mamba/ML_project/Testing/Huy/joint_vlm/FireboltVL"
cd "$PROJECT_DIR"

echo "=========================================="
echo "[$(date)] Waiting for current training to finish in firebolt tmux session..."
echo "=========================================="

# Wait for any torchrun/train.py process to finish
while pgrep -f "torchrun.*train.py" > /dev/null 2>&1 || pgrep -f "python.*train.py" > /dev/null 2>&1; do
    sleep 60
done

echo "=========================================="
echo "[$(date)] Current training finished! Starting Stage 1..."
echo "=========================================="

# Stage 1: 10 epochs with stage1.yaml config (connector-only training)
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
    train.py --config-name stage1 \
    training.ddp.enabled=True \
    training.stage=1

echo "=========================================="
echo "[$(date)] Stage 1 complete! Verifying epoch_10 checkpoint..."
echo "=========================================="

# Verify epoch_10 checkpoint exists
if [ ! -f "$PROJECT_DIR/fireboltvl_results1/stage1/epoch_10/model.safetensors" ]; then
    echo "ERROR: epoch_10 checkpoint not found! Checking available checkpoints:"
    ls "$PROJECT_DIR/fireboltvl_results1/stage1/"
    exit 1
fi

echo "[$(date)] Epoch 10 checkpoint confirmed. Starting Stage 2..."

# Stage 2: full fine-tune using epoch_10 pretrained weights
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
    train.py --config-name stage2 \
    training.ddp.enabled=True \
    training.stage=2 \
    training.resume_from_checkpoint=fireboltvl_results1/stage1/epoch_10

echo "=========================================="
echo "[$(date)] Stage 2 complete! All training finished."
echo "=========================================="
