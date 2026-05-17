import os
import math
import logging
from typing import List, Optional, Sequence, Tuple

import torch
torch.backends.cudnn.enabled = False
torch.backends.cuda.enable_cudnn_sdp(False)
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
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


def build_lm_labels(input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Build labels for AutoCausalLM loss.
    - Keep normal token ids as targets.
    - Mask padded positions with -100 so they are ignored by HF causal loss.
    """
    labels = input_ids.clone()
    if attention_mask is not None:
        labels = labels.masked_fill(attention_mask == 0, -100)
    return labels


MMPR_SPECIAL_TOKENS = [
    "<image>",
    "<SUMMARY>",
    "</SUMMARY>",
    "<CAPTION>",
    "</CAPTION>",
    "<REASONING>",
    "</REASONING>",
    "<CONCLUSION>",
    "</CONCLUSION>",
]


def prepare_tokenizer(tokenizer):
    """Ensure tokenizer can encode the section tags emitted by merge_mmpr_sharegpt.py."""
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    vocab = tokenizer.get_vocab()
    missing = [token for token in MMPR_SPECIAL_TOKENS if token not in vocab]
    if missing:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": missing},
            replace_additional_special_tokens=False,
        )
    return missing


def sync_config_with_tokenizer(config: FireboltVLConfig, tokenizer) -> FireboltVLConfig:
    config.vocab_size = len(tokenizer)
    if tokenizer.bos_token_id is not None:
        config.bos_token_id = tokenizer.bos_token_id
    if tokenizer.eos_token_id is not None:
        config.eos_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token_id is not None:
        config.pad_token_id = tokenizer.pad_token_id
    if "<image>" in tokenizer.get_vocab():
        config.image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    return config


def resize_model_for_tokenizer(model: FireboltVLForCausalLM, tokenizer) -> None:
    current_size = model.get_input_embeddings().num_embeddings
    target_size = len(tokenizer)
    if current_size != target_size:
        model.resize_token_embeddings(target_size)
    sync_config_with_tokenizer(model.config, tokenizer)


def set_training_stage(model: nn.Module, stage: int, vision_freeze_config: bool = True):
    """
    Stage 1: always freeze LLM + vision encoder; train connector/alignment heads only
      (router, expert_connector, ssm, visual_proj, optional aux heads).
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
        "recon_head.",
        "image_proj.",
        "text_proj.",
        "dino_head.",
        "contrastive_log_temp",
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


def _raw_model(model):
    return model.module if isinstance(model, DDP) else model


def summarize_trainable_parameters(model: nn.Module):
    summary = {
        "base_lm": 0,
        "vision_encoder": 0,
        "connector": 0,
        "aux_heads": 0,
        "other": 0,
    }
    raw = _raw_model(model)
    connector_prefixes = (
        "model.router.",
        "model.expert_connector.",
        "model.ssm.",
        "model.visual_proj.",
    )
    aux_prefixes = ("recon_head.", "image_proj.", "text_proj.", "dino_head.", "contrastive_log_temp")
    for name, param in raw.named_parameters():
        if not param.requires_grad:
            continue
        n = param.numel()
        if name.startswith("base_lm."):
            summary["base_lm"] += n
        elif name.startswith("model.vision_encoder."):
            summary["vision_encoder"] += n
        elif any(name.startswith(prefix) for prefix in connector_prefixes):
            summary["connector"] += n
        elif any(name.startswith(prefix) for prefix in aux_prefixes):
            summary["aux_heads"] += n
        else:
            summary["other"] += n
    return summary


class OptimizerBundle:
    """Small adapter for training with multiple optimizers."""

    def __init__(self, optimizers: Sequence[torch.optim.Optimizer]):
        self.optimizers = list(optimizers)
        if not self.optimizers:
            raise ValueError("OptimizerBundle requires at least one optimizer.")

    @property
    def param_groups(self):
        groups = []
        for optimizer in self.optimizers:
            groups.extend(optimizer.param_groups)
        return groups

    def zero_grad(self, set_to_none: bool = True):
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()


class SchedulerBundle:
    """Step a scheduler for each optimizer in an OptimizerBundle."""

    def __init__(self, schedulers: Sequence[torch.optim.lr_scheduler.LRScheduler]):
        self.schedulers = list(schedulers)

    def step(self):
        for scheduler in self.schedulers:
            scheduler.step()


def is_muon_parameter(name: str, param: torch.nn.Parameter) -> bool:
    """Muon is intended for 2D hidden-layer weights, not embeddings/bias/norms."""
    if param.ndim != 2:
        return False
    lowered = name.lower()
    excluded_fragments = (
        "embed",
        "embedding",
        "lm_head",
        "output",
        "norm",
        "ln",
        "bias",
    )
    return not any(fragment in lowered for fragment in excluded_fragments)


def split_muon_adamw_params(
    named_params: Sequence[Tuple[str, torch.nn.Parameter]],
) -> Tuple[List[Tuple[str, torch.nn.Parameter]], List[torch.nn.Parameter]]:
    muon_params = []
    adamw_params = []
    for name, param in named_params:
        if not param.requires_grad:
            continue
        if is_muon_parameter(name, param):
            muon_params.append((name, param))
        else:
            adamw_params.append(param)
    return muon_params, adamw_params


def build_optimizer_and_scheduler(
    cfg: DictConfig,
    named_trainable_params: Sequence[Tuple[str, torch.nn.Parameter]],
    lr: float,
    t_max: int,
):
    opt_cfg = cfg.training.optimizer
    optimizer_name = str(getattr(opt_cfg, "name", "adamw")).lower()
    weight_decay = float(getattr(opt_cfg, "weight_decay", 0.01))
    eta_min = float(cfg.training.scheduler.eta_min)

    if optimizer_name in {"adamw", "adam"}:
        params = [param for _, param in named_trainable_params if param.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
        return optimizer, scheduler, {"adamw": sum(p.numel() for p in params), "muon": 0}

    if optimizer_name not in {"muon_adamw", "muon+adamw", "muon"}:
        raise ValueError(f"Unsupported optimizer name: {optimizer_name}")
    if not hasattr(torch.optim, "Muon"):
        raise RuntimeError("torch.optim.Muon is not available in this PyTorch build.")

    muon_named_params, adamw_params = split_muon_adamw_params(named_trainable_params)
    optimizers = []
    schedulers = []
    if muon_named_params:
        muon_optimizer = torch.optim.Muon(
            muon_named_params,
            lr=float(getattr(opt_cfg, "muon_lr", lr)),
            weight_decay=float(getattr(opt_cfg, "muon_weight_decay", weight_decay)),
            momentum=float(getattr(opt_cfg, "muon_momentum", 0.95)),
            nesterov=bool(getattr(opt_cfg, "muon_nesterov", True)),
            ns_steps=int(getattr(opt_cfg, "muon_ns_steps", 5)),
            eps=float(getattr(opt_cfg, "muon_eps", 1e-7)),
            adjust_lr_fn=getattr(opt_cfg, "muon_adjust_lr_fn", "match_rms_adamw"),
        )
        optimizers.append(muon_optimizer)
        schedulers.append(torch.optim.lr_scheduler.CosineAnnealingLR(muon_optimizer, T_max=t_max, eta_min=eta_min))

    if adamw_params:
        adamw_optimizer = torch.optim.AdamW(
            adamw_params,
            lr=float(getattr(opt_cfg, "adamw_lr", lr)),
            weight_decay=float(getattr(opt_cfg, "adamw_weight_decay", weight_decay)),
        )
        optimizers.append(adamw_optimizer)
        schedulers.append(torch.optim.lr_scheduler.CosineAnnealingLR(adamw_optimizer, T_max=t_max, eta_min=eta_min))

    stats = {
        "muon": sum(param.numel() for _, param in muon_named_params),
        "adamw": sum(param.numel() for param in adamw_params),
    }
    return OptimizerBundle(optimizers), SchedulerBundle(schedulers), stats


def uses_dino_alignment_loss(model, stage: int) -> bool:
    cfg = _raw_model(model).config
    return (
        stage == 1
        and bool(getattr(cfg, "aux_dino_loss_enabled", False))
        and bool(getattr(cfg, "stage1_dino_only", False))
    )


def eval_model(
    model,
    val_dataloader,
    device,
    amp_dtype: torch.dtype,
    rank: Optional[int] = None,
    stage: int = 1,
    dino_teacher=None,
):
    model.eval()
    total_loss = 0.0
    n = 0
    bad_batches = 0
    use_amp = (device.type == "cuda")
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Evaluating", disable=not is_main_process(rank)):
            if batch is None:
                continue
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            image_counts = batch.get("image_counts")
            if image_counts is not None:
                image_counts = image_counts.to(device, non_blocking=True)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device, non_blocking=True)
            pixel_values_dino = batch.get("pixel_values_dino")
            if pixel_values_dino is not None:
                pixel_values_dino = pixel_values_dino.to(device, non_blocking=True)
            labels = None
            if not uses_dino_alignment_loss(model, stage):
                labels = batch.get("labels")
                labels = labels.to(device, non_blocking=True) if labels is not None else build_lm_labels(input_ids, attention_mask)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                outputs = model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    image_counts=image_counts,
                    attention_mask=attention_mask,
                    labels=labels,
                    pixel_values_dino=pixel_values_dino,
                    dino_teacher=dino_teacher,
                )
            loss = outputs.loss
            if loss is None:
                bad_batches += 1
                continue
            if not torch.isfinite(loss):
                bad_batches += 1
                continue
            if dist.is_initialized():
                loss = ddp_all_reduce_mean(loss.detach(), device=device)
            total_loss += loss.item()
            n += 1
    if is_main_process(rank) and bad_batches > 0:
        print(f"[Eval] Skipped non-finite batches: {bad_batches}")
        logging.warning(f"[Eval] Skipped non-finite batches: {bad_batches}")
    avg_loss = total_loss / max(n, 1)
    if math.isnan(avg_loss) or math.isinf(avg_loss):
        return float("inf"), float("inf")
    if uses_dino_alignment_loss(model, stage):
        return avg_loss, float("nan")
    # PPL = exp(NLL); clamp exponent input for numeric safety.
    ppl = float(math.exp(min(avg_loss, 20.0)))
    return avg_loss, ppl


def save_checkpoint(output_dir: str, model, tokenizer, rank: Optional[int], cfg: Optional[DictConfig] = None):
    if not is_main_process(rank):
        return
    os.makedirs(output_dir, exist_ok=True)
    to_save = model.module if isinstance(model, DDP) else model
    if tokenizer is not None:
        sync_config_with_tokenizer(to_save.config, tokenizer)
    to_save.save_pretrained(output_dir, safe_serialization=True)
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)
    if cfg is not None:
        OmegaConf.save(config=cfg, f=os.path.join(output_dir, "training_config.yaml"))
    print(f"[Rank 0] Saved checkpoint to: {output_dir}")


def train(
    cfg,
    model,
    train_dl,
    val_dl,
    optimizer,
    scheduler,
    device,
    stage: int = 1,
    rank: Optional[int] = None,
    train_sampler=None,
    tokenizer=None,
    vjepa_teacher=None,
    dino_teacher=None,
):
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
        for batch in pbar:
            if batch is None:
                continue
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            image_counts = batch.get("image_counts")
            if image_counts is not None:
                image_counts = image_counts.to(device, non_blocking=True)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device, non_blocking=True)
            pixel_values_vjepa = batch.get("pixel_values_vjepa")
            if pixel_values_vjepa is not None:
                pixel_values_vjepa = pixel_values_vjepa.to(device, non_blocking=True)
            mask_flags = batch.get("mask_flags")
            if mask_flags is not None:
                mask_flags = mask_flags.to(device, non_blocking=True)
            pixel_values_dino = batch.get("pixel_values_dino")
            if pixel_values_dino is not None:
                pixel_values_dino = pixel_values_dino.to(device, non_blocking=True)
            labels = None
            if not uses_dino_alignment_loss(model, stage):
                labels = batch.get("labels")
                labels = labels.to(device, non_blocking=True) if labels is not None else build_lm_labels(input_ids, attention_mask)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                outputs = model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    image_counts=image_counts,
                    attention_mask=attention_mask,
                    labels=labels,
                    pixel_values_vjepa=pixel_values_vjepa,
                    pixel_values_dino=pixel_values_dino,
                    mask_flags=mask_flags,
                    vjepa_teacher=vjepa_teacher,
                    dino_teacher=dino_teacher,
                )
                loss = outputs.loss
            if loss is None:
                raise RuntimeError("Model returned loss=None. Check stage loss configuration.")
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
        avg_val_loss, avg_val_ppl = eval_model(
            model,
            val_dl,
            device,
            amp_dtype=amp_dtype,
            rank=rank,
            stage=stage,
            dino_teacher=dino_teacher,
        )
        if is_main_process(rank):
            val_ppl_str = "N/A" if math.isnan(avg_val_ppl) else f"{avg_val_ppl:.4f}"
            print(
                f"Stage {stage} Epoch [{epoch+1}/{num_epochs}] "
                f"Train Loss: {avg_train:.4f} Val Loss: {avg_val_loss:.4f} Val PPL: {val_ppl_str}"
            )
            logging.info(
                f"Stage {stage} Epoch [{epoch+1}/{num_epochs}] "
                f"Train Loss: {avg_train:.4f} Val Loss: {avg_val_loss:.4f} Val PPL: {val_ppl_str}"
            )
        out_dir = os.path.join(results_dir, out_prefix, f"epoch_{epoch+1}")
        save_checkpoint(out_dir, model, tokenizer, rank, cfg=cfg)


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig):
    rank, local_rank, world_size = setup_ddp(cfg)
    device = torch.device(f"cuda:{local_rank}" if local_rank is not None else cfg.training.device)
    stage = getattr(cfg.training, "stage", 1)
    stage1_dino_alignment = stage == 1
    stage2_lm_only = stage == 2

    if is_main_process(rank):
        logging.getLogger().handlers.clear()
        os.makedirs(cfg.training.results_dir, exist_ok=True)
        log_dir = os.path.dirname(cfg.logging.filename)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(filename=cfg.logging.filename, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path)
    added_tokens = prepare_tokenizer(tokenizer)

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
        aux_balance_loss_enabled=False if stage2_lm_only else getattr(cfg.model, "aux_balance_loss_enabled", False),
        aux_balance_loss_weight=getattr(cfg.model, "aux_balance_loss_weight", 0.01),
        aux_recon_loss_enabled=False,
        aux_recon_loss_weight=getattr(cfg.model, "aux_recon_loss_weight", 1.0),
        recon_mask_ratio=getattr(cfg.model, "recon_mask_ratio", 0.5),
        recon_mask_prob=getattr(cfg.model, "recon_mask_prob", 0.5),
        recon_teacher_model=getattr(cfg.model, "recon_teacher_model", "facebook/vjepa2-vitg-fpc64-384"),
        recon_teacher_hidden_size=getattr(cfg.model, "recon_teacher_hidden_size", 1408),
        aux_contrastive_loss_enabled=False,
        aux_contrastive_loss_weight=getattr(cfg.model, "aux_contrastive_loss_weight", 0.1),
        contrastive_embed_dim=getattr(cfg.model, "contrastive_embed_dim", 512),
        contrastive_temperature=getattr(cfg.model, "contrastive_temperature", 0.07),
        contrastive_learnable_temp=getattr(cfg.model, "contrastive_learnable_temp", True),
        aux_dino_loss_enabled=stage1_dino_alignment and getattr(cfg.model, "aux_dino_loss_enabled", True),
        aux_dino_loss_weight=getattr(cfg.model, "aux_dino_loss_weight", 1.0),
        stage1_dino_only=getattr(cfg.model, "stage1_dino_only", False),
        dino_teacher_model=getattr(cfg.model, "dino_teacher_model", "facebook/dinov2-base"),
        dino_teacher_hidden_size=getattr(cfg.model, "dino_teacher_hidden_size", 768),
        dino_student_temperature=getattr(cfg.model, "dino_student_temperature", 0.1),
        dino_teacher_temperature=getattr(cfg.model, "dino_teacher_temperature", 0.04),
    )
    sync_config_with_tokenizer(config, tokenizer)
    model = FireboltVLForCausalLM(config)
    resize_model_for_tokenizer(model, tokenizer)
    if added_tokens and is_main_process(rank):
        print(f"Added tokenizer special tokens: {added_tokens}")

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

    set_training_stage(model, stage, vision_freeze_config=config.vision_freeze)
    trainable_named_params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    trainable_params = [p for _, p in trainable_named_params]

    vjepa_teacher = None
    vjepa_processor = None
    if config.aux_recon_loss_enabled:
        from transformers import AutoModel as HFAutoModel
        if is_main_process(rank):
            print(f"Loading V-JEPA 2 teacher: {config.recon_teacher_model}")
        vjepa_teacher = HFAutoModel.from_pretrained(
            config.recon_teacher_model,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to(device)
        vjepa_teacher.eval()
        for p in vjepa_teacher.parameters():
            p.requires_grad = False
        try:
            from transformers import AutoVideoProcessor
            vjepa_processor = AutoVideoProcessor.from_pretrained(config.recon_teacher_model)
        except Exception:
            from transformers import AutoImageProcessor
            vjepa_processor = AutoImageProcessor.from_pretrained(config.recon_teacher_model)
        if is_main_process(rank):
            print("V-JEPA 2 teacher loaded and frozen.")

    dino_teacher = None
    dino_processor = None
    if config.aux_dino_loss_enabled:
        from transformers import AutoImageProcessor, AutoModel as HFAutoModel
        if is_main_process(rank):
            print(f"Loading DINO teacher: {config.dino_teacher_model}")
        dino_teacher = HFAutoModel.from_pretrained(
            config.dino_teacher_model,
            torch_dtype=torch.bfloat16 if getattr(cfg.training, "amp_dtype", "bf16") == "bf16" else torch.float16,
            low_cpu_mem_usage=True,
        ).to(device)
        dino_teacher.eval()
        for p in dino_teacher.parameters():
            p.requires_grad = False
        dino_processor = AutoImageProcessor.from_pretrained(config.dino_teacher_model)
        if is_main_process(rank):
            print("DINO teacher loaded and frozen.")

    processor = AutoProcessor.from_pretrained(cfg.processor_path)
    if dist.is_initialized():
        dist.barrier()
    raw_model = model
    sync_config_with_tokenizer(raw_model.config, tokenizer)

    use_ddp = is_distributed(cfg)
    siglip_patch_size = 16 if config.vision_hidden_size <= 768 else 14
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
        max_length=getattr(cfg.data, "max_length", None),
        mask_ratio=config.recon_mask_ratio if config.aux_recon_loss_enabled else 0.0,
        mask_prob=config.recon_mask_prob if config.aux_recon_loss_enabled else 0.0,
        patch_size=siglip_patch_size,
        vjepa_processor=vjepa_processor,
        dino_processor=dino_processor,
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
    T_max = getattr(stage_cfg, "num_epochs", cfg.training.scheduler.T_max) if stage_cfg else cfg.training.scheduler.T_max
    optimizer, scheduler, optimizer_stats = build_optimizer_and_scheduler(
        cfg,
        trainable_named_params,
        lr=lr,
        t_max=T_max,
    )
    n_trainable = count_trainable_parameters(model)
    if is_main_process(rank):
        print(f"Stage {stage}: trainable parameters = {n_trainable}")
        print(f"Trainable parameter groups: {summarize_trainable_parameters(model)}")
        print(f"Optimizer: {getattr(cfg.training.optimizer, 'name', 'adamw')} groups = {optimizer_stats}")
        logging.info(f"Stage {stage}: trainable parameters = {n_trainable}")
        logging.info(f"Trainable parameter groups: {summarize_trainable_parameters(model)}")
        logging.info(f"Optimizer: {getattr(cfg.training.optimizer, 'name', 'adamw')} groups = {optimizer_stats}")

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
        vjepa_teacher=vjepa_teacher,
        dino_teacher=dino_teacher,
    )
    cleanup_ddp()


if __name__ == "__main__":
    main()
