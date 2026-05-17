import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union


def compute_load_balancing_loss(
    router_logits: torch.Tensor,
    num_experts: int,
    expert_top_k: int,
) -> torch.Tensor:
    """Switch Transformer load-balancing loss for MoE expert connector.

    Args:
        router_logits: (B, k, num_experts)
        num_experts: total number of experts
        expert_top_k: number of experts selected per token
    """
    B, k, E = router_logits.shape
    logits_flat = router_logits.reshape(-1, E)
    routing_probs = F.softmax(logits_flat, dim=-1)
    _, top_idx = torch.topk(logits_flat, k=expert_top_k, dim=-1)
    dispatch_mask = torch.zeros_like(logits_flat)
    dispatch_mask.scatter_(-1, top_idx, 1.0)
    fraction_per_expert = dispatch_mask.mean(dim=0)
    mean_prob_per_expert = routing_probs.mean(dim=0)
    return num_experts * (fraction_per_expert * mean_prob_per_expert).sum()


def compute_reconstruction_loss(
    lm_hidden_states: torch.Tensor,
    num_visual_tokens: int,
    teacher_features: torch.Tensor,
    recon_head: nn.Module,
    router_indices: torch.Tensor,
    mask_flags: torch.Tensor,
    siglip_grid_size: int,
    vjepa_grid_size: int,
    siglip_patch_size: int,
    vjepa_patch_size: int,
) -> torch.Tensor:
    """MSE reconstruction loss between LM visual hidden states and V-JEPA 2 teacher features.

    Args:
        lm_hidden_states: (B, seq_len, D_lm) last-layer hidden states from LM
        num_visual_tokens: number of visual tokens prepended (routing_top_k)
        teacher_features: (B, N_vjepa, D_vjepa) from frozen V-JEPA 2 encoder
        recon_head: projects D_lm -> D_vjepa
        router_indices: (B, k) SigLIP patch indices selected by TopKRouter
        mask_flags: (B,) bool indicating which samples had masked patches
        siglip_grid_size: SigLIP spatial grid side (e.g. 27 for 384/14)
        vjepa_grid_size: V-JEPA 2 spatial grid side (e.g. 24 for 384/16)
        siglip_patch_size: SigLIP patch pixel size (e.g. 14)
        vjepa_patch_size: V-JEPA 2 patch pixel size (e.g. 16)
    """
    if not mask_flags.any():
        return torch.tensor(0.0, device=lm_hidden_states.device, requires_grad=True)

    vis_hidden = lm_hidden_states[:, :num_visual_tokens, :]
    predicted = recon_head(vis_hidden)

    siglip_row = router_indices // siglip_grid_size
    siglip_col = router_indices % siglip_grid_size
    center_y = siglip_row * siglip_patch_size + siglip_patch_size // 2
    center_x = siglip_col * siglip_patch_size + siglip_patch_size // 2
    vjepa_row = center_y // vjepa_patch_size
    vjepa_col = center_x // vjepa_patch_size
    vjepa_idx = (vjepa_row * vjepa_grid_size + vjepa_col).clamp(
        0, vjepa_grid_size * vjepa_grid_size - 1
    )

    D_vjepa = teacher_features.size(-1)
    target = torch.gather(
        teacher_features, 1,
        vjepa_idx.unsqueeze(-1).expand(-1, -1, D_vjepa),
    )

    return F.mse_loss(predicted[mask_flags], target[mask_flags].detach())


def compute_contrastive_loss(
    lm_hidden_states: torch.Tensor,
    num_visual_tokens: int,
    attention_mask: torch.Tensor,
    image_proj: nn.Module,
    text_proj: nn.Module,
    temperature: Union[float, torch.Tensor],
) -> torch.Tensor:
    """CLIP-style symmetric InfoNCE contrastive loss.

    Args:
        lm_hidden_states: (B, seq_len, D_lm) last-layer hidden states
        num_visual_tokens: visual tokens at positions [0, num_visual_tokens)
        attention_mask: (B, seq_len) including visual positions
        image_proj: projects D_lm -> embed_dim
        text_proj: projects D_lm -> embed_dim
        temperature: scalar or learnable parameter
    """
    vis_hidden = lm_hidden_states[:, :num_visual_tokens, :]
    image_emb = vis_hidden.mean(dim=1)

    text_hidden = lm_hidden_states[:, num_visual_tokens:, :]
    text_mask = attention_mask[:, num_visual_tokens:]
    mask_sum = text_mask.sum(dim=1, keepdim=True).clamp(min=1)
    text_emb = (text_hidden * text_mask.unsqueeze(-1)).sum(dim=1) / mask_sum

    image_z = F.normalize(image_proj(image_emb), dim=-1)
    text_z = F.normalize(text_proj(text_emb), dim=-1)

    if isinstance(temperature, torch.Tensor):
        temperature = temperature.clamp(min=0.01, max=100.0)
    else:
        temperature = max(temperature, 0.01)
    logits = (image_z @ text_z.T) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
    return loss


def compute_dino_alignment_loss(
    student_visual_tokens: torch.Tensor,
    teacher_features: torch.Tensor,
    dino_head: nn.Module,
    student_temperature: float = 0.1,
    teacher_temperature: float = 0.04,
) -> torch.Tensor:
    """DINO-style distillation from a frozen DINO image teacher to visual tokens.

    The student is the Firebolt visual stream after projection into the LLM
    hidden space. The teacher is a frozen DINO/DINOv2 encoder. Both sides are
    pooled to one image representation and trained with soft-label cross entropy.
    """
    student_logits = dino_head(student_visual_tokens.mean(dim=1))
    if teacher_features.dim() == 3:
        teacher_logits = teacher_features[:, 0, :]
    else:
        teacher_logits = teacher_features

    student_logits = F.normalize(student_logits, dim=-1)
    teacher_logits = F.normalize(teacher_logits.detach(), dim=-1)
    teacher_logits = teacher_logits - teacher_logits.mean(dim=0, keepdim=True)

    teacher_probs = F.softmax(teacher_logits / max(teacher_temperature, 1e-6), dim=-1)
    student_log_probs = F.log_softmax(student_logits / max(student_temperature, 1e-6), dim=-1)
    return -(teacher_probs * student_log_probs).sum(dim=-1).mean()
