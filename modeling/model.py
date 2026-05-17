# FireboltVL: V-JEPA + Routing top-k + Expert Connector + SSM + concat -> LLM
import math
import os
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union

from safetensors.torch import load_file as safe_load_file
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from .config import FireboltVLConfig
from .vision_encoder import VisionEncoderVJEPA
from .routing import TopKRouter
from .expert_connector import ExpertConnector
from .ssm_module import build_ssm
from .losses import (
    compute_load_balancing_loss,
    compute_reconstruction_loss,
    compute_contrastive_loss,
    compute_dino_alignment_loss,
)


class VisualProjector(nn.Module):
    """Project visual tokens (B, k, D_vision) -> (B, k, D_llm)."""

    def __init__(self, d_vision: int, d_llm: int, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_d = d_vision if i == 0 else d_llm
            out_d = d_llm
            layers.append(nn.Linear(in_d, out_d))
            layers.append(nn.LayerNorm(out_d))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        self.proj = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class FireboltVLModel(nn.Module):
    """
    Image -> VisionEncoder -> TopKRouter -> ExpertConnector -> SSM -> VisualProjector
    -> replace <image> token embeddings. If no <image> token is present, visual
    tokens are prepended as a compatibility fallback.
    """

    def __init__(self, config: FireboltVLConfig, lm_hidden_size: int):
        super().__init__()
        self.config = config
        d_vision = config.vision_hidden_size

        self.vision_encoder = VisionEncoderVJEPA(
            ckpt_path=config.vision_ckpt_path or "",
            encoder_type=config.vision_encoder_type,
            freeze=config.vision_freeze,
        )
        self.router = TopKRouter(d_vision, config.routing_top_k)
        self.expert_connector = ExpertConnector(
            d_model=d_vision,
            num_experts=config.num_experts,
            top_k=getattr(config, "expert_top_k", 2),
            expert_hidden_dim=config.expert_hidden_dim,
            dropout=config.expert_dropout,
        )
        self.ssm = build_ssm(
            config.ssm_type,
            d_vision,
            d_state=config.ssm_d_state,
            dropout=config.ssm_dropout,
        )
        self.visual_proj = VisualProjector(
            d_vision,
            config.visual_proj_dim if config.visual_proj_dim else lm_hidden_size,
            num_layers=config.visual_proj_layers,
            dropout=config.visual_proj_dropout,
        )
        self.embed_dropout = nn.Dropout(0.0)

    def _encode_single_image_batch(self, image_inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_i = self.vision_encoder(image_inputs)
        z_top, router_indices = self.router(z_i)
        z_expert, expert_router_logits = self.expert_connector(z_top)
        ssm_out, _ = self.ssm(z_expert)
        if ssm_out is None:
            ssm_out = z_expert
        return self.visual_proj(ssm_out), expert_router_logits, router_indices

    def _encode_images(self, image_inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if image_inputs.dim() == 5:
            batch_size, num_images = image_inputs.shape[:2]
            flat_inputs = image_inputs.reshape(batch_size * num_images, *image_inputs.shape[2:])
            visual_tokens, expert_router_logits, router_indices = self._encode_single_image_batch(flat_inputs)
            visual_tokens = visual_tokens.reshape(batch_size, num_images, *visual_tokens.shape[1:])
            expert_router_logits = expert_router_logits.reshape(batch_size, num_images, *expert_router_logits.shape[1:])
            router_indices = router_indices.reshape(batch_size, num_images, *router_indices.shape[1:])
            return visual_tokens, expert_router_logits, router_indices
        return self._encode_single_image_batch(image_inputs)

    def _flatten_valid_visual_tokens(
        self,
        vision_embeds: torch.Tensor,
        image_counts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if vision_embeds.dim() == 3:
            return vision_embeds
        if image_counts is None:
            return vision_embeds.flatten(0, 1)
        valid = []
        for row_idx, count in enumerate(image_counts.tolist()):
            if count > 0:
                valid.append(vision_embeds[row_idx, :count])
        if not valid:
            return vision_embeds[:, :1].flatten(0, 1)
        return torch.cat(valid, dim=0)

    def forward(
        self,
        input_ids: torch.LongTensor,
        input_embedding_layer: nn.Embedding,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_inputs: Optional[torch.Tensor] = None,
        vision_embeds: Optional[torch.Tensor] = None,
        image_counts: Optional[torch.Tensor] = None,
        image_token_id: Optional[int] = None,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        int,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        text_embeds = self.embed_dropout(input_embedding_layer(input_ids))
        batch_size = text_embeds.size(0)
        num_prepended_visual_tokens = 0
        expert_router_logits = None
        router_indices = None

        image_tensor = image_inputs if image_inputs is not None else pixel_values
        if image_tensor is not None and vision_embeds is None:
            vision_embeds, expert_router_logits, router_indices = self._encode_images(image_tensor)

        if vision_embeds is not None:
            vision_embeds = vision_embeds.to(device=text_embeds.device, dtype=text_embeds.dtype)
            student_visual_tokens = self._flatten_valid_visual_tokens(vision_embeds, image_counts=image_counts)
            if image_token_id is not None:
                image_token_mask = input_ids.eq(image_token_id)
                if image_token_mask.any():
                    fused_embeds = text_embeds.clone()
                    if vision_embeds.dim() == 4:
                        pooled_vision = vision_embeds.mean(dim=2)
                        for batch_idx in range(input_ids.size(0)):
                            positions = image_token_mask[batch_idx].nonzero(as_tuple=False).flatten()
                            if positions.numel() == 0:
                                continue
                            count = int(image_counts[batch_idx].item()) if image_counts is not None else pooled_vision.size(1)
                            count = max(1, min(count, pooled_vision.size(1)))
                            for token_idx, position in enumerate(positions.tolist()):
                                image_idx = min(token_idx, count - 1)
                                fused_embeds[batch_idx, position] = pooled_vision[batch_idx, image_idx]
                    else:
                        pooled_vision = vision_embeds.mean(dim=1)
                        rows = image_token_mask.nonzero(as_tuple=False)[:, 0]
                        fused_embeds[image_token_mask] = pooled_vision.index_select(0, rows)
                    return (
                        fused_embeds,
                        attention_mask,
                        num_prepended_visual_tokens,
                        expert_router_logits,
                        router_indices,
                        student_visual_tokens,
                    )

            prepend_embeds = vision_embeds.flatten(1, 2) if vision_embeds.dim() == 4 else vision_embeds
            num_prepended_visual_tokens = prepend_embeds.size(1)
            fused_embeds = torch.cat([prepend_embeds, text_embeds], dim=1)
            if attention_mask is not None:
                vision_mask = torch.ones(
                    (batch_size, num_prepended_visual_tokens),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([vision_mask, attention_mask], dim=1)
            return (
                fused_embeds,
                attention_mask,
                num_prepended_visual_tokens,
                expert_router_logits,
                router_indices,
                student_visual_tokens,
            )

        return text_embeds, attention_mask, 0, None, None, None


class FireboltVLForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = FireboltVLConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    main_input_name = "input_ids"

    def __init__(self, config: FireboltVLConfig):
        super().__init__(config)
        local_files_only = os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"
        self.base_lm = AutoModelForCausalLM.from_pretrained(
            config.lm_name_or_path,
            local_files_only=local_files_only,
        )
        lm_hidden = self.base_lm.get_input_embeddings().embedding_dim
        base_vocab_size = self.base_lm.get_input_embeddings().num_embeddings
        config_vocab_size = getattr(config, "vocab_size", None)
        target_vocab_size = config_vocab_size or base_vocab_size
        if target_vocab_size != base_vocab_size:
            self.base_lm.resize_token_embeddings(target_vocab_size)
        self.config.vocab_size = target_vocab_size
        if getattr(config, "visual_proj_dim", None) in (None, 0):
            config.visual_proj_dim = lm_hidden
        self.model = FireboltVLModel(config, lm_hidden_size=lm_hidden)

        if getattr(config, "aux_recon_loss_enabled", False):
            teacher_dim = config.recon_teacher_hidden_size
            self.recon_head = nn.Sequential(
                nn.Linear(lm_hidden, lm_hidden),
                nn.GELU(),
                nn.Linear(lm_hidden, teacher_dim),
            )
        else:
            self.recon_head = None

        if getattr(config, "aux_contrastive_loss_enabled", False):
            embed_dim = config.contrastive_embed_dim
            self.image_proj = nn.Sequential(
                nn.Linear(lm_hidden, lm_hidden),
                nn.GELU(),
                nn.Linear(lm_hidden, embed_dim),
            )
            self.text_proj = nn.Sequential(
                nn.Linear(lm_hidden, lm_hidden),
                nn.GELU(),
                nn.Linear(lm_hidden, embed_dim),
            )
            if getattr(config, "contrastive_learnable_temp", True):
                self.contrastive_log_temp = nn.Parameter(
                    torch.tensor(math.log(1.0 / config.contrastive_temperature))
                )
            else:
                self.contrastive_log_temp = None
        else:
            self.image_proj = None
            self.text_proj = None
            self.contrastive_log_temp = None

        if getattr(config, "aux_dino_loss_enabled", False):
            teacher_dim = config.dino_teacher_hidden_size
            self.dino_head = nn.Sequential(
                nn.Linear(lm_hidden, lm_hidden),
                nn.GELU(),
                nn.Linear(lm_hidden, teacher_dim),
            )
        else:
            self.dino_head = None

        self.post_init()

    @classmethod
    def load_model(
        cls,
        model_dir: str,
        device: Optional[Union[str, torch.device]] = "cuda",
        torch_dtype: Optional[torch.dtype] = torch.bfloat16,
        strict: bool = False,
    ) -> "FireboltVLForCausalLM":
        config = FireboltVLConfig.from_pretrained(model_dir)
        tokenizer = None
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        except Exception:
            tokenizer = None

        if tokenizer is not None:
            config.vocab_size = len(tokenizer)
            if tokenizer.bos_token_id is not None:
                config.bos_token_id = tokenizer.bos_token_id
            if tokenizer.eos_token_id is not None:
                config.eos_token_id = tokenizer.eos_token_id
            if tokenizer.pad_token_id is not None:
                config.pad_token_id = tokenizer.pad_token_id
            if "<image>" in tokenizer.get_vocab():
                config.image_token_id = tokenizer.convert_tokens_to_ids("<image>")

        model = cls(config)
        safe_path = os.path.join(model_dir, "model.safetensors")
        bin_path = os.path.join(model_dir, "pytorch_model.bin")
        if os.path.isfile(safe_path):
            state = safe_load_file(safe_path, device="cpu")
        elif os.path.isfile(bin_path):
            state = torch.load(bin_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
        else:
            raise FileNotFoundError(f"No checkpoint found in {model_dir}")

        model.load_state_dict(state, strict=strict)
        if device is not None and torch_dtype is not None:
            model.to(device=device, dtype=torch_dtype)
        elif device is not None:
            model.to(device=device)
        elif torch_dtype is not None:
            model.to(dtype=torch_dtype)
        model.eval()
        return model

    def get_input_embeddings(self) -> nn.Embedding:
        return self.base_lm.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings: nn.Embedding) -> None:
        self.base_lm.set_input_embeddings(new_embeddings)

    def get_output_embeddings(self) -> nn.Module:
        return self.base_lm.get_output_embeddings()

    def set_output_embeddings(self, new_lm_head: nn.Module) -> None:
        if hasattr(self.base_lm, "set_output_embeddings"):
            self.base_lm.set_output_embeddings(new_lm_head)

    def tie_weights(self, **kwargs) -> None:
        if hasattr(self.base_lm, "tie_weights"):
            self.base_lm.tie_weights(**kwargs)

    def prepare_inputs_for_generation(self, input_ids: torch.LongTensor, past_key_values=None, **kwargs):
        if past_key_values:
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "attention_mask": kwargs.get("attention_mask", None),
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
            "pixel_values": kwargs.get("pixel_values", None) if past_key_values is None else None,
            "image_inputs": kwargs.get("image_inputs", None) if past_key_values is None else None,
            "vision_embeds": kwargs.get("vision_embeds", None) if past_key_values is None else None,
            "image_counts": kwargs.get("image_counts", None) if past_key_values is None else None,
        }

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_inputs: Optional[torch.Tensor] = None,
        vision_embeds: Optional[torch.Tensor] = None,
        image_counts: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        image_token_id: Optional[int] = None,
        pixel_values_vjepa: Optional[torch.Tensor] = None,
        pixel_values_dino: Optional[torch.Tensor] = None,
        mask_flags: Optional[torch.Tensor] = None,
        vjepa_teacher: Optional[nn.Module] = None,
        dino_teacher: Optional[nn.Module] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        num_visual_tokens = 0
        expert_router_logits = None
        router_indices = None
        student_visual_tokens = None

        if inputs_embeds is None:
            (
                inputs_embeds,
                attention_mask,
                num_visual_tokens,
                expert_router_logits,
                router_indices,
                student_visual_tokens,
            ) = self.model(
                input_ids=input_ids,
                input_embedding_layer=self.get_input_embeddings(),
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_inputs=image_inputs,
                vision_embeds=vision_embeds,
                image_counts=image_counts,
                image_token_id=image_token_id or getattr(self.config, "image_token_id", None),
            )
            if labels is not None and num_visual_tokens > 0 and labels.dim() == 2:
                pad_labels = torch.full(
                    (labels.size(0), num_visual_tokens),
                    -100,
                    dtype=labels.dtype,
                    device=labels.device,
                )
                labels = torch.cat([pad_labels, labels], dim=1)

        dino_loss = None
        if (
            getattr(self.config, "aux_dino_loss_enabled", False)
            and self.dino_head is not None
            and dino_teacher is not None
            and pixel_values_dino is not None
            and student_visual_tokens is not None
        ):
            dino_inputs = pixel_values_dino
            if dino_inputs.dim() == 5:
                batch_size, num_images = dino_inputs.shape[:2]
                flat_dino_inputs = dino_inputs.reshape(batch_size * num_images, *dino_inputs.shape[2:])
                if image_counts is not None:
                    image_idx = torch.arange(num_images, device=image_counts.device).unsqueeze(0)
                    valid_mask = image_idx < image_counts.unsqueeze(1)
                    flat_dino_inputs = flat_dino_inputs[valid_mask.reshape(-1).to(flat_dino_inputs.device)]
                dino_inputs = flat_dino_inputs
            with torch.no_grad():
                teacher_out = dino_teacher(pixel_values=dino_inputs)
                if getattr(teacher_out, "last_hidden_state", None) is not None:
                    teacher_features = teacher_out.last_hidden_state
                elif getattr(teacher_out, "pooler_output", None) is not None:
                    teacher_features = teacher_out.pooler_output
                elif isinstance(teacher_out, dict) and "last_hidden_state" in teacher_out:
                    teacher_features = teacher_out["last_hidden_state"]
                elif isinstance(teacher_out, dict) and "pooler_output" in teacher_out:
                    teacher_features = teacher_out["pooler_output"]
                elif isinstance(teacher_out, (tuple, list)) and len(teacher_out) > 0:
                    teacher_features = teacher_out[0]
                else:
                    raise RuntimeError("Cannot extract DINO teacher features.")
            dino_loss = compute_dino_alignment_loss(
                student_visual_tokens=student_visual_tokens,
                teacher_features=teacher_features,
                dino_head=self.dino_head,
                student_temperature=getattr(self.config, "dino_student_temperature", 0.1),
                teacher_temperature=getattr(self.config, "dino_teacher_temperature", 0.04),
            )

        if labels is None and dino_loss is not None:
            total_loss = self.config.aux_dino_loss_weight * dino_loss
            if getattr(self.config, "aux_balance_loss_enabled", False) and expert_router_logits is not None:
                balance_loss = compute_load_balancing_loss(
                    expert_router_logits,
                    num_experts=self.config.num_experts,
                    expert_top_k=self.config.expert_top_k,
                )
                total_loss = total_loss + self.config.aux_balance_loss_weight * balance_loss
            return CausalLMOutputWithPast(loss=total_loss, logits=None)

        need_hidden = (
            self.training
            and labels is not None
            and num_visual_tokens > 0
            and (
                getattr(self.config, "aux_recon_loss_enabled", False)
                or getattr(self.config, "aux_contrastive_loss_enabled", False)
            )
        )

        lm_outputs = self.base_lm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=need_hidden,
            **kwargs,
        )

        if not self.training or labels is None or num_visual_tokens == 0:
            if dino_loss is not None:
                return CausalLMOutputWithPast(
                    loss=lm_outputs.loss + self.config.aux_dino_loss_weight * dino_loss,
                    logits=lm_outputs.logits,
                    past_key_values=lm_outputs.past_key_values,
                    hidden_states=lm_outputs.hidden_states,
                    attentions=lm_outputs.attentions,
                )
            return lm_outputs

        total_loss = lm_outputs.loss

        if dino_loss is not None:
            total_loss = total_loss + self.config.aux_dino_loss_weight * dino_loss

        if getattr(self.config, "aux_balance_loss_enabled", False) and expert_router_logits is not None:
            balance_loss = compute_load_balancing_loss(
                expert_router_logits,
                num_experts=self.config.num_experts,
                expert_top_k=self.config.expert_top_k,
            )
            total_loss = total_loss + self.config.aux_balance_loss_weight * balance_loss

        if (
            getattr(self.config, "aux_recon_loss_enabled", False)
            and self.recon_head is not None
            and vjepa_teacher is not None
            and pixel_values_vjepa is not None
            and mask_flags is not None
            and mask_flags.any()
            and need_hidden
        ):
            lm_hidden = lm_outputs.hidden_states[-1]
            with torch.no_grad():
                vjepa_input = pixel_values_vjepa.unsqueeze(1).repeat(1, 2, 1, 1, 1)
                teacher_out = vjepa_teacher(pixel_values_videos=vjepa_input)
                teacher_features = teacher_out.last_hidden_state

            siglip_patch_size = 14
            siglip_grid_size = 27
            vision_h = getattr(self.config, "vision_hidden_size", 1152)
            if vision_h <= 768:
                siglip_patch_size = 16
                siglip_grid_size = 16

            recon_loss = compute_reconstruction_loss(
                lm_hidden_states=lm_hidden,
                num_visual_tokens=num_visual_tokens,
                teacher_features=teacher_features,
                recon_head=self.recon_head,
                router_indices=router_indices,
                mask_flags=mask_flags,
                siglip_grid_size=siglip_grid_size,
                vjepa_grid_size=24,
                siglip_patch_size=siglip_patch_size,
                vjepa_patch_size=16,
            )
            total_loss = total_loss + self.config.aux_recon_loss_weight * recon_loss

        if (
            getattr(self.config, "aux_contrastive_loss_enabled", False)
            and self.image_proj is not None
            and need_hidden
        ):
            lm_hidden = lm_outputs.hidden_states[-1]
            temperature = (
                self.contrastive_log_temp.exp()
                if self.contrastive_log_temp is not None
                else self.config.contrastive_temperature
            )
            contrastive_loss = compute_contrastive_loss(
                lm_hidden_states=lm_hidden,
                num_visual_tokens=num_visual_tokens,
                attention_mask=attention_mask,
                image_proj=self.image_proj,
                text_proj=self.text_proj,
                temperature=temperature,
            )
            total_loss = total_loss + self.config.aux_contrastive_loss_weight * contrastive_loss

        return CausalLMOutputWithPast(
            loss=total_loss,
            logits=lm_outputs.logits,
            past_key_values=lm_outputs.past_key_values,
            hidden_states=lm_outputs.hidden_states,
            attentions=lm_outputs.attentions,
        )

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        emb = self.base_lm.resize_token_embeddings(new_num_tokens)
        self.config.vocab_size = new_num_tokens
        return emb

    def _set_gradient_checkpointing(self, module, value: bool = False):
        if hasattr(self.base_lm, "_set_gradient_checkpointing"):
            self.base_lm._set_gradient_checkpointing(module, value)
