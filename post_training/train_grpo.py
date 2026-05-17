import os
import sys
import logging
from pathlib import Path

import torch

torch.backends.cudnn.enabled = False
torch.backends.cuda.enable_cudnn_sdp(False)

import hydra
from omegaconf import DictConfig
from transformers import AutoProcessor, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modeling import FireboltVLForCausalLM
from train import (
    setup_ddp,
    cleanup_ddp,
    is_distributed,
    is_main_process,
    resize_model_for_tokenizer,
    set_training_stage,
    save_checkpoint,
    build_optimizer_and_scheduler,
    count_trainable_parameters,
    summarize_trainable_parameters,
)

from post_training.grpo_dataset import create_grpo_dataloader
from post_training.grpo_trainer import GRPOTrainer


GRPO_SPECIAL_TOKENS = ["<think>", "</think>", "<answer>", "</answer>"]


def _add_special_tokens_compat(tokenizer, tokens):
    try:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": tokens},
            replace_additional_special_tokens=False,
        )
    except TypeError:
        tokenizer.add_special_tokens({"additional_special_tokens": tokens})


def prepare_grpo_tokenizer(tokenizer):
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    from train import MMPR_SPECIAL_TOKENS

    vocab = tokenizer.get_vocab()
    all_tokens = list(MMPR_SPECIAL_TOKENS) + list(GRPO_SPECIAL_TOKENS)
    missing = [t for t in all_tokens if t not in vocab]
    if missing:
        _add_special_tokens_compat(tokenizer, missing)
    return missing


def load_policy_model(checkpoint_path: str, device: torch.device, dtype: torch.dtype):
    model = FireboltVLForCausalLM.load_model(
        checkpoint_path, device=str(device), torch_dtype=dtype,
    )
    model.train()
    return model


def load_reference_model(
    checkpoint_path: str, device: torch.device, dtype: torch.dtype,
):
    model = FireboltVLForCausalLM.load_model(
        checkpoint_path, device=str(device), torch_dtype=dtype,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


@hydra.main(version_base=None, config_path="../configs", config_name="grpo")
def main(cfg: DictConfig):
    rank, local_rank, world_size = setup_ddp(cfg)
    device = torch.device(
        f"cuda:{local_rank}" if local_rank is not None else cfg.training.device
    )
    amp_dtype = (
        torch.bfloat16
        if getattr(cfg.training, "amp_dtype", "bf16") == "bf16"
        else torch.float16
    )

    if is_main_process(rank):
        logging.getLogger().handlers.clear()
        os.makedirs(cfg.training.results_dir, exist_ok=True)
        log_dir = os.path.dirname(cfg.logging.filename)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(
            filename=cfg.logging.filename,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

    checkpoint_path = cfg.training.resume_from_checkpoint
    if is_main_process(rank):
        print(f"Loading tokenizer from: {cfg.tokenizer_path}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path)
    added_tokens = prepare_grpo_tokenizer(tokenizer)
    tokenizer.padding_side = "left"

    if is_main_process(rank):
        print(f"Loading policy model from: {checkpoint_path}")
    policy_model = load_policy_model(checkpoint_path, device, amp_dtype)
    resize_model_for_tokenizer(policy_model, tokenizer)
    set_training_stage(policy_model, stage=2, vision_freeze_config=True)

    if is_main_process(rank):
        print(f"Loading reference model from: {checkpoint_path}")
    ref_model = load_reference_model(checkpoint_path, device, amp_dtype)
    resize_model_for_tokenizer(ref_model, tokenizer)

    if added_tokens and is_main_process(rank):
        print(f"Added special tokens: {added_tokens}")

    if is_main_process(rank):
        print(f"Loading processor from: {cfg.processor_path}")
    processor = AutoProcessor.from_pretrained(cfg.processor_path)

    if is_main_process(rank):
        print(f"Loading dataset: {cfg.data.dataset_name}")
    dl_result = create_grpo_dataloader(
        dataset_name=cfg.data.dataset_name,
        image_base_path=cfg.data.image_base_path,
        tokenizer=tokenizer,
        processor=processor,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        ddp=is_distributed(cfg),
        rank=rank if rank is not None else 0,
        world_size=world_size,
        max_prompt_length=getattr(cfg.data, "max_prompt_length", 256),
        filter_correct_only=getattr(cfg.data, "filter_correct_only", False),
    )
    dataloader = dl_result["dataloader"]
    train_sampler = dl_result["sampler"]

    trainable_named_params = [
        (name, p) for name, p in policy_model.named_parameters() if p.requires_grad
    ]

    stage_cfg = getattr(cfg.training, "stage3", None)
    lr = getattr(stage_cfg, "lr", None) if stage_cfg else None
    if lr is None:
        lr = cfg.training.optimizer.lr
    T_max = (
        getattr(stage_cfg, "num_epochs", cfg.training.scheduler.T_max)
        if stage_cfg
        else cfg.training.scheduler.T_max
    )

    optimizer, scheduler, optimizer_stats = build_optimizer_and_scheduler(
        cfg, trainable_named_params, lr=lr, t_max=T_max,
    )

    use_ddp = is_distributed(cfg)
    if use_ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP

        policy_model = DDP(
            policy_model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

    n_trainable = count_trainable_parameters(policy_model)
    if is_main_process(rank):
        print(f"GRPO: trainable parameters = {n_trainable}")
        print(f"Trainable groups: {summarize_trainable_parameters(policy_model)}")
        print(f"Optimizer: {getattr(cfg.training.optimizer, 'name', 'adamw')} groups = {optimizer_stats}")
        print(f"GRPO config: group_size={cfg.grpo.group_size}, clip_eps={cfg.grpo.clip_epsilon}, kl_coeff={cfg.grpo.kl_coeff}")
        logging.info(f"GRPO: trainable parameters = {n_trainable}")
        logging.info(f"GRPO config: group_size={cfg.grpo.group_size}")

    stage_cfg = getattr(cfg.training, "stage3", None)
    num_epochs = getattr(stage_cfg, "num_epochs", None) if stage_cfg else None
    if num_epochs is None:
        num_epochs = cfg.training.num_epochs

    results_dir = cfg.training.results_dir

    def save_fn(epoch):
        out_dir = os.path.join(results_dir, "grpo", f"epoch_{epoch + 1}")
        save_checkpoint(out_dir, policy_model, tokenizer, rank, cfg=cfg)

    trainer = GRPOTrainer(
        cfg=cfg,
        policy_model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        processor=processor,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        rank=rank,
    )

    trainer.train(
        dataloader=dataloader,
        num_epochs=num_epochs,
        train_sampler=train_sampler,
        tokenizer=tokenizer,
        save_fn=save_fn,
    )

    cleanup_ddp()
    if is_main_process(rank):
        print("GRPO training complete.")


if __name__ == "__main__":
    main()
