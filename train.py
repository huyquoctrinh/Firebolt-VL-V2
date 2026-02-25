import os
import math
import logging
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from safetensors.torch import load_file as safe_load_file

from dataset import create_dataloader
from modeling import FireboltVLForCausalLM, FireboltVLConfig
from transformers import AutoProcessor, AutoTokenizer


# -----------------------
# DDP helpers
# -----------------------
def is_distributed(cfg: DictConfig) -> bool:
    return bool(getattr(cfg.training.ddp, "enabled", False))


def setup_ddp(cfg: DictConfig):
    """Initialize process group. Must be launched with torchrun."""
    if not is_distributed(cfg):
        return None, None, 1
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return rank, local_rank, world_size


def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: Optional[int]) -> bool:
    return (rank is None) or (rank == 0)


def ddp_all_reduce_mean(value: torch.Tensor, device=None) -> torch.Tensor:
    """All-reduce a tensor across DDP processes (average)."""
    if dist.is_available() and dist.is_initialized():
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, device=device or torch.cuda.current_device())
        if not value.is_cuda:
            value = value.to(device or torch.cuda.current_device())
        was_scalar = value.dim() == 0
        if was_scalar:
            value = value.unsqueeze(0)
        dist.all_reduce(value, op=dist.ReduceOp.AVG)
        if was_scalar:
            value = value.squeeze(0)
    return value


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_training_stage(model: nn.Module, stage: int, vision_freeze_config: bool = True):
    """
    Stage 1: freeze LLM + vision encoder; train connector only
      (router, expert_connector, ssm, visual_proj).
    Stage 2: unfreeze all; vision encoder follows vision_freeze_config.
    """
    for p in model.parameters():
        p.requires_grad = False

    connector_prefixes = (
        "model.router.",
        "model.expert_connector.",
        "model.ssm.",
        "model.visual_proj.",
        "model.embed_dropout.",
    )
    vision_prefix = "model.vision_encoder."
    base_lm_prefix = "base_lm."

    for name, p in model.named_parameters():
        if stage == 1:
            if any(name.startswith(prefix) for prefix in connector_prefixes):
                p.requires_grad = True
        else:
            if any(name.startswith(prefix) for prefix in connector_prefixes):
                p.requires_grad = True
            if name.startswith(base_lm_prefix):
                p.requires_grad = True
            if name.startswith(vision_prefix) and not vision_freeze_config:
                p.requires_grad = True


def eval_model(model, val_dataloader, device, rank: Optional[int] = None):
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Evaluating", disable=not is_main_process(rank)):
            if batch is None:
                continue
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            attention_mask = batch.get("attention_mask").to(device, non_blocking=True)
            outputs = model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            if dist.is_initialized():
                loss = ddp_all_reduce_mean(loss.detach(), device=device)
            total_loss += loss.item()
            n += 1
    avg_loss = total_loss / max(n, 1)
    if math.isnan(avg_loss) or math.isinf(avg_loss):
        return float("inf"), float("inf")
    # PPL = exp(NLL); clamp exponent input for numeric safety.
    ppl = float(math.exp(min(avg_loss, 20.0)))
    return avg_loss, ppl


def save_checkpoint(output_dir: str, model, tokenizer, rank: Optional[int]):
    if not is_main_process(rank):
        return
    os.makedirs(output_dir, exist_ok=True)
    to_save = model.module if isinstance(model, DDP) else model
    to_save.save_pretrained(output_dir)
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)
    print(f"[Rank 0] Saved checkpoint to: {output_dir}")


def train(cfg, model, train_dl, val_dl, optimizer, scheduler, device, stage: int = 1, rank: Optional[int] = None, train_sampler=None, tokenizer=None):
    stage_cfg = getattr(cfg.training, f"stage{stage}", None)
    num_epochs = getattr(stage_cfg, "num_epochs", None) if stage_cfg else None
    if num_epochs is None:
        num_epochs = cfg.training.num_epochs
    amp_dtype = torch.bfloat16 if getattr(cfg.training, "amp_dtype", "bf16") == "bf16" else torch.float16
    results_dir = cfg.training.results_dir
    out_prefix = f"stage{stage}"
    for epoch in range(num_epochs):
        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        n_batches = 0
        pbar = tqdm(train_dl, desc=f"Stage {stage} Epoch {epoch+1}/{num_epochs}", disable=not is_main_process(rank))
        use_amp = (device.type == "cuda")
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            for batch in pbar:
                if batch is None:
                    continue
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                pixel_values = batch["pixel_values"].to(device, non_blocking=True)
                attention_mask = batch.get("attention_mask").to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=1.0
                )
                optimizer.step()
                total_loss += loss.detach().item()
                n_batches += 1
                if is_main_process(rank):
                    pbar.set_postfix(loss=total_loss / n_batches, lr=optimizer.param_groups[0]["lr"])
        scheduler.step()
        avg_train = total_loss / max(n_batches, 1)
        if dist.is_initialized():
            avg_train_t = torch.tensor(avg_train, device=device)
            ddp_all_reduce_mean(avg_train_t, device=device)
            avg_train = avg_train_t.item()
        if math.isnan(avg_train) or math.isinf(avg_train):
            avg_train = float("inf")
        avg_val_loss, avg_val_ppl = eval_model(model, val_dl, device, rank=rank)
        if is_main_process(rank):
            print(
                f"Stage {stage} Epoch [{epoch+1}/{num_epochs}] "
                f"Train Loss: {avg_train:.4f} Val Loss: {avg_val_loss:.4f} Val PPL: {avg_val_ppl:.4f}"
            )
            logging.info(
                f"Stage {stage} Epoch [{epoch+1}/{num_epochs}] "
                f"Train Loss: {avg_train:.4f} Val Loss: {avg_val_loss:.4f} Val PPL: {avg_val_ppl:.4f}"
            )
        out_dir = os.path.join(results_dir, out_prefix, f"epoch_{epoch+1}")
        save_checkpoint(out_dir, model, tokenizer, rank)


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig):
    rank, local_rank, world_size = setup_ddp(cfg)
    device = torch.device(f"cuda:{local_rank}" if local_rank is not None else cfg.training.device)

    if is_main_process(rank):
        logging.getLogger().handlers.clear()
        os.makedirs(cfg.training.results_dir, exist_ok=True)
        log_dir = os.path.dirname(cfg.logging.filename)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(filename=cfg.logging.filename, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    config = FireboltVLConfig(
        vision_encoder_type=cfg.model.vision_encoder_type,
        vision_ckpt_path=cfg.model.vision_ckpt_path,
        vision_hidden_size=cfg.model.vision_hidden_size,
        vision_freeze=cfg.model.vision_freeze,
        routing_top_k=cfg.model.routing_top_k,
        num_experts=cfg.model.num_experts,
        expert_top_k=getattr(cfg.model, "expert_top_k", 2),
        expert_hidden_dim=cfg.model.expert_hidden_dim,
        expert_dropout=cfg.model.expert_dropout,
        ssm_type=cfg.model.ssm_type,
        ssm_d_state=cfg.model.ssm_d_state,
        ssm_dropout=cfg.model.ssm_dropout,
        visual_proj_dim=getattr(cfg.model, "visual_proj_dim", 1024),
        visual_proj_layers=cfg.model.visual_proj_layers,
        visual_proj_dropout=cfg.model.visual_proj_dropout,
        lm_name_or_path=cfg.model.lm_name_or_path,
        freeze_llm=cfg.model.freeze_llm,
        image_token_id=cfg.model.image_token_id,
    )
    model = FireboltVLForCausalLM(config)
    resume_path = getattr(cfg.training, "resume_from_checkpoint", None)
    if resume_path:
        loaded = False
        safe_path = os.path.join(resume_path, "model.safetensors")
        bin_path = os.path.join(resume_path, "pytorch_model.bin")
        if os.path.isfile(safe_path):
            state = safe_load_file(safe_path, device="cpu")
            model.load_state_dict(state, strict=False)
            loaded = True
        elif os.path.isfile(bin_path):
            state = torch.load(bin_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state, strict=False)
            loaded = True
        if loaded and is_main_process(rank):
            print(f"Resumed from {resume_path}")
    model.to(device)

    stage = getattr(cfg.training, "stage", 1)
    set_training_stage(model, stage, vision_freeze_config=config.vision_freeze)
    if not config.freeze_llm or stage == 2:
        model.base_lm.resize_token_embeddings(config.vocab_size)
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path)
    processor = AutoProcessor.from_pretrained(cfg.processor_path)
    if dist.is_initialized():
        dist.barrier()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    raw_model = model
    raw_model.config.image_token_id = tokenizer.convert_tokens_to_ids("<image>") if "<image>" in tokenizer.get_vocab() else cfg.model.image_token_id

    use_ddp = is_distributed(cfg)
    dataloaders = create_dataloader(
        image_path=cfg.data.image_path,
        json_path=cfg.data.json_path,
        tokenizer=tokenizer,
        processor=processor,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        ddp=use_ddp,
        rank=rank if rank is not None else 0,
        world_size=world_size,
        train_val_split=cfg.data.train_val_split,
    )
    if dist.is_initialized():
        dist.barrier()

    if use_ddp:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
    (model.module if isinstance(model, DDP) else model).config.image_token_id = getattr(raw_model.config, "image_token_id", cfg.model.image_token_id)

    stage_cfg = getattr(cfg.training, f"stage{stage}", None)
    lr = getattr(stage_cfg, "lr", None) if stage_cfg else None
    if lr is None:
        lr = cfg.training.optimizer.lr
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    T_max = getattr(stage_cfg, "num_epochs", cfg.training.scheduler.T_max) if stage_cfg else cfg.training.scheduler.T_max
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=cfg.training.scheduler.eta_min)
    n_trainable = count_trainable_parameters(model)
    if is_main_process(rank):
        print(f"Stage {stage}: trainable parameters = {n_trainable}")
        logging.info(f"Stage {stage}: trainable parameters = {n_trainable}")

    train(
        cfg,
        model,
        dataloaders["train_dataloader"],
        dataloaders["val_dataloader"],
        optimizer,
        scheduler,
        device,
        stage=stage,
        rank=rank,
        train_sampler=dataloaders.get("train_sampler"),
        tokenizer=tokenizer,
    )
    cleanup_ddp()


if __name__ == "__main__":
    main()
