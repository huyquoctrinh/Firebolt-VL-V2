#!/usr/bin/env bash
# FireboltVL training: single-GPU or multi-GPU DDP

# Single GPU using configs/stage1.yml
# python train.py --config-name stage1

# Multi-GPU with DDP using configs/stage1.yml
# Then launch with torchrun:
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py --config-name stage2 training.ddp.enabled=True

# Example: 4 GPUs, stage 1
# torchrun --nproc_per_node=4 train.py --config-name stage1 training.ddp.enabled=True training.stage=1

# Example: 2 GPUs, stage 2, resume from stage1
# torchrun --nproc_per_node=2 train.py --config-name stage1 training.ddp.enabled=True training.stage=2 training.resume_from_checkpoint=fireboltvl_results/stage1/epoch_3
