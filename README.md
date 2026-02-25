# FireboltVL

Kiến trúc **Joint Vision-Language** theo sơ đồ: **V-JEPA (vision encoder)** → **Routing top-k** → **Expert Connector** → **State-Space Model** → concat với text embeddings → **LLM** với loss **L_NLL** (text).

## Cấu trúc thư mục

```
FireboltVL/
  configs/default.yaml   # Cấu hình Hydra
  modeling/
    config.py           # FireboltVLConfig
    model.py            # FireboltVLModel, FireboltVLForCausalLM (LLM loss)
    vision_encoder.py   # VisionEncoderVJEPA (SigLIP / Grid SigLIP)
    routing.py          # TopKRouter
    expert_connector.py # ExpertConnector (MoE)
    ssm_module.py       # SSM (S4/S4D/self_attn, optional từ Viper-LM)
  dataset.py           # CCDataset, create_dataloader
  train.py              # Training với L_NLL
  infer.py              # Inference
  README.md
```

## Luồng kiến trúc

1. **Image** → Vision Encoder (V-JEPA style; mặc định SigLIP) → **Z_I** (B, T×N, D).
2. **Routing top-k** → chọn k tokens → **Z_I** (B, k, D).
3. **Expert Connector** (nhiều expert, router) → **Z_I** (B, k, D).
4. **State-Space Model** (S4 / S4D / self_attn) trên visual stream → visual token embeddings.
5. **Project** visual tokens lên không gian LLM → (B, k, D_llm).
6. **Text** → tokenizer → embedding → text_embeds (B, T, D_llm).
7. **Concat** [visual_tokens; text_embeds] → LLM (frozen hoặc trainable).
8. **Loss:** causal LM loss (NLL) trên phần text.

## Cấu hình chính (`configs/default.yaml`)

- `model.vision_encoder_type`: `"siglip"` | `"siglip_grid"`
- `model.routing_top_k`: số vision tokens sau routing.
- `model.num_experts`, `model.expert_hidden_dim`: Expert Connector.
- `model.ssm_type`: `"s4"` | `"s4d"` | `"self_attn"` (S4/S4D cần Viper-LM trong path).

## Training 2 giai đoạn

- **Stage 1:** Đóng băng LLM + vision encoder, chỉ train connector (router, expert_connector, ssm, visual_proj).  
  Trong `configs/default.yaml`: `training.stage: 1`. Có thể chỉnh `training.stage1.num_epochs`, `training.stage1.lr`.
- **Stage 2:** Train toàn bộ (unfreeze LLM; vision encoder vẫn theo `model.vision_freeze`).  
  Set `training.stage: 2`, `training.resume_from_checkpoint: "fireboltvl_results/stage1/epoch_3"` (hoặc thư mục stage1 bất kỳ). Có thể chỉnh `training.stage2.num_epochs`, `training.stage2.lr`.

## Chạy

```bash
# Single GPU
python train.py training.stage=1

# Multi-GPU DDP: bật training.ddp.enabled=True và chạy bằng torchrun
torchrun --nproc_per_node=NUM_GPUS train.py training.ddp.enabled=True

# Stage 2: full fine-tune (resume từ stage 1)
python train.py training.stage=2 training.resume_from_checkpoint=fireboltvl_results/stage1/epoch_3

# Inference (cần set evaluation.model_dir và evaluation.image_path trong config)
python infer.py
```

## Phụ thuộc

- PyTorch, transformers, hydra-core, omegaconf, PIL, tqdm.
- Để dùng S4/S4D: đặt FireboltVL và Viper-LM cùng cấp trong repo (ví dụ `joint_vlm/FireboltVL`, `joint_vlm/Viper-LM`) để `ssm_module` import được từ Viper-LM.
