#!/usr/bin/env bash
# GRPO post-training for FireboltVL
# Requires: conda activate llama

cd "$(dirname "$0")/.."

# Single GPU (GPU 1 - adjust as needed)
CUDA_VISIBLE_DEVICES=1 /home/mamba/anaconda3/envs/llama/bin/python post_training/train_grpo.py --config-name grpo

# Multi-GPU DDP example:
# CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 post_training/train_grpo.py --config-name grpo training.ddp.enabled=True
