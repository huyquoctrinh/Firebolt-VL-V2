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
        # MoE load-balancing loss
        aux_balance_loss_enabled: bool = False,
        aux_balance_loss_weight: float = 0.01,
        # Masked image reconstruction loss (V-JEPA 2 teacher)
        aux_recon_loss_enabled: bool = False,
        aux_recon_loss_weight: float = 1.0,
        recon_mask_ratio: float = 0.5,
        recon_mask_prob: float = 0.5,
        recon_teacher_model: str = "facebook/vjepa2-vitg-fpc64-384",
        recon_teacher_hidden_size: int = 1408,
        # CLIP-style contrastive loss
        aux_contrastive_loss_enabled: bool = False,
        aux_contrastive_loss_weight: float = 0.1,
        contrastive_embed_dim: int = 512,
        contrastive_temperature: float = 0.07,
        contrastive_learnable_temp: bool = True,
        # DINO-style visual alignment loss
        aux_dino_loss_enabled: bool = False,
        aux_dino_loss_weight: float = 1.0,
        stage1_dino_only: bool = False,
        dino_teacher_model: str = "facebook/dinov2-base",
        dino_teacher_hidden_size: int = 768,
        dino_student_temperature: float = 0.1,
        dino_teacher_temperature: float = 0.04,
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
        self.aux_balance_loss_enabled = aux_balance_loss_enabled
        self.aux_balance_loss_weight = aux_balance_loss_weight
        self.aux_recon_loss_enabled = aux_recon_loss_enabled
        self.aux_recon_loss_weight = aux_recon_loss_weight
        self.recon_mask_ratio = recon_mask_ratio
        self.recon_mask_prob = recon_mask_prob
        self.recon_teacher_model = recon_teacher_model
        self.recon_teacher_hidden_size = recon_teacher_hidden_size
        self.aux_contrastive_loss_enabled = aux_contrastive_loss_enabled
        self.aux_contrastive_loss_weight = aux_contrastive_loss_weight
        self.contrastive_embed_dim = contrastive_embed_dim
        self.contrastive_temperature = contrastive_temperature
        self.contrastive_learnable_temp = contrastive_learnable_temp
        self.aux_dino_loss_enabled = aux_dino_loss_enabled
        self.aux_dino_loss_weight = aux_dino_loss_weight
        self.stage1_dino_only = stage1_dino_only
        self.dino_teacher_model = dino_teacher_model
        self.dino_teacher_hidden_size = dino_teacher_hidden_size
        self.dino_student_temperature = dino_student_temperature
        self.dino_teacher_temperature = dino_teacher_temperature
