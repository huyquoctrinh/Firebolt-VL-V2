from typing import Optional
from transformers import PretrainedConfig


class FireboltVLConfig(PretrainedConfig):
    model_type = "fireboltvl"

    def __init__(
        self,
        vision_encoder_type: str = "siglip",
        vision_ckpt_path: Optional[str] = None,
        vision_hidden_size: int = 768,
        vision_freeze: bool = True,
        routing_top_k: int = 32,
        num_experts: int = 4,
        expert_top_k: int = 2,
        expert_hidden_dim: int = 512,
        expert_dropout: float = 0.1,
        ssm_type: str = "s4d",
        ssm_d_state: int = 64,
        ssm_dropout: float = 0.1,
        visual_proj_dim: int = 1024,
        visual_proj_layers: int = 1,
        visual_proj_dropout: float = 0.1,
        lm_name_or_path: str = "LiquidAI/LFM2-350M",
        freeze_llm: bool = False,
        image_token_id: Optional[int] = 64400,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: int = 0,
        **kwargs,
    ):
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs,
        )
        self.vision_encoder_type = vision_encoder_type
        self.vision_ckpt_path = vision_ckpt_path
        self.vision_hidden_size = vision_hidden_size
        self.vision_freeze = vision_freeze
        self.routing_top_k = routing_top_k
        self.num_experts = num_experts
        self.expert_top_k = expert_top_k
        self.expert_hidden_dim = expert_hidden_dim
        self.expert_dropout = expert_dropout
        self.ssm_type = ssm_type
        self.ssm_d_state = ssm_d_state
        self.ssm_dropout = ssm_dropout
        self.visual_proj_dim = visual_proj_dim
        self.visual_proj_layers = visual_proj_layers
        self.visual_proj_dropout = visual_proj_dropout
        self.lm_name_or_path = lm_name_or_path
        self.freeze_llm = freeze_llm
        self.image_token_id = image_token_id
