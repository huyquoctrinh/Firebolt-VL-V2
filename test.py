#!/usr/bin/env python3
import torch
from transformers import AutoModel, GPT2Config, GPT2LMHeadModel

import modeling.model as firebolt_model
import modeling.vision_encoder as vision_encoder
from modeling import FireboltVLConfig, FireboltVLForCausalLM


def _mock_lm_loader(*args, **kwargs):
    cfg = GPT2Config(
        vocab_size=64,
        n_positions=64,
        n_embd=32,
        n_layer=2,
        n_head=4,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    return GPT2LMHeadModel(cfg)


def _load_siglip_vision_only(ckpt_path: str, dtype=torch.float32, device=None):
    model = AutoModel.from_pretrained(ckpt_path, dtype=dtype, low_cpu_mem_usage=True)
    if hasattr(model, "vision_model"):
        model = model.vision_model
    if device is not None:
        model = model.to(device)
    return model


def run_test():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    siglip_ckpt = "/home/mamba/ML_project/Testing/Huy/joint_vlm/viper-vlm/pretrained/siglip384"

    original_lm_loader = firebolt_model.AutoModelForCausalLM.from_pretrained
    original_vision_loader = vision_encoder._load_siglip_encoder
    firebolt_model.AutoModelForCausalLM.from_pretrained = _mock_lm_loader
    vision_encoder._load_siglip_encoder = _load_siglip_vision_only

    try:
        cfg = FireboltVLConfig(
            vision_encoder_type="siglip",
            vision_ckpt_path=siglip_ckpt,
            vision_hidden_size=1152,
            routing_top_k=2,
            num_experts=2,
            expert_top_k=1,
            expert_hidden_dim=32,
            ssm_type="self_attn",
            visual_proj_dim=32,
            visual_proj_layers=1,
            lm_name_or_path="mock-lm",
            freeze_llm=False,
            image_token_id=10,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )

        model = FireboltVLForCausalLM(cfg).to(device)
        model.train()

        batch_size, seq_len = 2, 6
        input_ids = torch.randint(0, 64, (batch_size, seq_len), device=device)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
        pixel_values = torch.randn(batch_size, 3, 384, 384, device=device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=input_ids,
        )

        expected_len = seq_len + cfg.routing_top_k
        assert outputs.logits.shape == (batch_size, expected_len, 64), (
            f"Unexpected logits shape: {outputs.logits.shape}"
        )
        assert outputs.loss is not None, "Loss should not be None."
        assert torch.isfinite(outputs.loss).item(), f"Loss is not finite: {outputs.loss}"

        model.eval()
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids[:1],
                pixel_values=pixel_values[:1],
                max_new_tokens=4,
                do_sample=False,
                use_cache=False,
            )
        assert generated.shape[0] == 1, f"Unexpected generated batch: {generated.shape}"

        print("FireboltVL test with real SigLIP vision encoder passed.")
        print(f"loss={outputs.loss.item():.6f}, logits_shape={tuple(outputs.logits.shape)}, generated_shape={tuple(generated.shape)}")
    finally:
        firebolt_model.AutoModelForCausalLM.from_pretrained = original_lm_loader
        vision_encoder._load_siglip_encoder = original_vision_loader


if __name__ == "__main__":
    run_test()
