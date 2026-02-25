# FireboltVL: V-JEPA + Routing top-k + Expert Connector + SSM + concat -> LLM
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union

from transformers import AutoModelForCausalLM, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerationMixin

from .config import FireboltVLConfig
from .vision_encoder import VisionEncoderVJEPA
from .routing import TopKRouter
from .expert_connector import ExpertConnector
from .ssm_module import build_ssm


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
    Luồng: Image -> V-JEPA(SigLIP) -> Z_I -> Routing top-k -> Expert Connector -> SSM -> visual tokens (B,k,D_llm).
    Text -> embed -> text_embeds. Concat [visual_tokens; text_embeds] -> LLM.
    """
    def __init__(self, config: FireboltVLConfig, lm_hidden_size: int):
        super().__init__()
        self.config = config
        d_vision = config.vision_hidden_size
        k = config.routing_top_k
        d_llm = lm_hidden_size

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
            config.visual_proj_dim if config.visual_proj_dim else d_llm,
            num_layers=config.visual_proj_layers,
            dropout=config.visual_proj_dropout,
        )
        self.embed_dropout = nn.Dropout(0.0)

    def encode_vision(
        self,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns:
            visual_tokens: (B, k, D_llm) để concat với text.
        """
        z_i = self.vision_encoder(pixel_values)
        z_top, _ = self.router(z_i)
        z_expert = self.expert_connector(z_top)
        ssm_out, _ = self.ssm(z_expert)
        if ssm_out is None:
            ssm_out = z_expert
        visual_tokens = self.visual_proj(ssm_out)
        return visual_tokens

    def forward(
        self,
        input_ids: torch.LongTensor,
        input_embedding_layer: nn.Embedding,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_token_id: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], int]:
        """
        Returns:
            inputs_embeds: (B, L_total, D_llm) với L_total = k + T_text.
            attention_mask: (B, L_total) nếu có.
            num_visual_tokens: int k.
        """
        text_embeds = self.embed_dropout(input_embedding_layer(input_ids))
        B, T_text, D = text_embeds.shape
        num_visual_tokens = 0

        if pixel_values is not None:
            visual_tokens = self.encode_vision(pixel_values)
            num_visual_tokens = visual_tokens.size(1)
            inputs_embeds = torch.cat([visual_tokens, text_embeds], dim=1)
            if attention_mask is not None:
                vis_mask = torch.ones(B, num_visual_tokens, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([vis_mask, attention_mask], dim=1)
        else:
            inputs_embeds = text_embeds

        return inputs_embeds, attention_mask, num_visual_tokens


class FireboltVLForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = FireboltVLConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    main_input_name = "input_ids"

    def __init__(self, config: FireboltVLConfig):
        super().__init__(config)
        self.base_lm = AutoModelForCausalLM.from_pretrained(config.lm_name_or_path)
        lm_hidden = self.base_lm.get_input_embeddings().embedding_dim
        self.config.vocab_size = getattr(self.base_lm.config, "vocab_size", None)
        if self.config.vocab_size is None:
            self.config.vocab_size = self.base_lm.config.vocab_size if hasattr(self.base_lm.config, "vocab_size") else 50304
        if getattr(config, "visual_proj_dim", None) is None or config.visual_proj_dim == 0:
            config.visual_proj_dim = lm_hidden
        self.model = FireboltVLModel(config, lm_hidden_size=lm_hidden)
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.base_lm.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.base_lm.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.base_lm.get_output_embeddings()

    def prepare_inputs_for_generation(self, input_ids: torch.LongTensor, past_key_values=None, **kwargs):
        if past_key_values:
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "attention_mask": kwargs.get("attention_mask", None),
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
            "pixel_values": kwargs.get("pixel_values", None) if past_key_values is None else None,
        }

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        image_token_id: Optional[int] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        image_token_id = image_token_id or getattr(self.config, "image_token_id", None)
        inputs_embeds, attention_mask, num_visual_tokens = self.model(
            input_ids=input_ids,
            input_embedding_layer=self.get_input_embeddings(),
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_token_id=image_token_id,
        )
        ignore_index = -100
        if labels is not None and num_visual_tokens > 0 and labels.dim() == 2:
            pad_labels = torch.full(
                (labels.size(0), num_visual_tokens),
                ignore_index,
                dtype=labels.dtype,
                device=labels.device,
            )
            labels = torch.cat([pad_labels, labels], dim=1)
        lm_outputs = self.base_lm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=None,
            output_hidden_states=False,
            **kwargs,
        )
        logits = lm_outputs.logits
        loss = None
        if labels is not None:
            loss_text = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=ignore_index,
            )
            loss = loss_text
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=lm_outputs.past_key_values,
            hidden_states=lm_outputs.hidden_states,
            attentions=lm_outputs.attentions,
        )

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        emb = self.base_lm.resize_token_embeddings(new_num_tokens)
        self.config.vocab_size = new_num_tokens
        return emb


if __name__ == "__main__":
    # Test model instantiation and forward pass with dummy data
    config = FireboltVLConfig(
        vision_hidden_size=512,
        vision_encoder_type="resnet",
        vision_freeze=True,
        routing_top_k=4,
        num_experts=2,
        expert_hidden_dim=256,
        expert_dropout=0.1,
        ssm_type="linear",
        ssm_d_state=128,
        ssm_dropout=0.1,
        visual_proj_dim=512,
        visual_proj_layers=2,
        visual_proj_dropout=0.1,
        lm_name_or_path="gpt2",
    )
    model = FireboltVLForCausalLM(config)
    batch_size = 2
    seq_len = 5
    vocab_size = config.vocab_size
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    pixel_values = torch.randn(batch_size, 3, 224, 224)  # Dummy image input
    outputs = model(input_ids=input_ids, pixel_values=pixel_values)
    print("Logits shape:", outputs.logits.shape)
