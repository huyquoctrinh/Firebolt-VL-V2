# Inference FireboltVL
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, GenerationConfig
import hydra
from omegaconf import DictConfig

from modeling import FireboltVLForCausalLM, FireboltVLConfig


@torch.inference_mode()
def run_inference(cfg: DictConfig, model, tokenizer, processor, prompt: str, image_path: str = None):
    device = cfg.training.device
    image_inputs = None
    if image_path:
        try:
            img = Image.open(image_path).convert("RGB")
            image_inputs = processor(images=[img], return_tensors="pt")["pixel_values"].to(device, dtype=model.dtype)
            user_content = f"<image>\n{prompt}"
        except FileNotFoundError:
            user_content = prompt
    else:
        user_content = prompt
    messages = [{"role": "user", "content": user_content}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_tensors="pt",
    ).to(device)
    if isinstance(inputs, dict):
        inputs = inputs["input_ids"]
    gen_cfg = GenerationConfig(
        max_new_tokens=cfg.evaluation.max_new_tokens,
        do_sample=cfg.evaluation.do_sample,
        temperature=cfg.evaluation.get("temperature", 0.7),
        top_p=cfg.evaluation.get("top_p", 0.9),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=False,
    )
    gen_kwargs = {}
    if image_inputs is not None:
        gen_kwargs["pixel_values"] = image_inputs.to(device, dtype=next(model.parameters()).dtype)
    model_dtype = next(model.parameters()).dtype
    with torch.autocast(device_type=device, dtype=model_dtype):
        gen_ids = model.generate(inputs, generation_config=gen_cfg, **gen_kwargs)
    prompt_len = inputs.shape[1]
    new_tokens = gen_ids[0, prompt_len:]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
    print("\n--- Prompt ---")
    print(prompt)
    if image_path:
        print(f"(image: {image_path})")
    print("\n--- Response ---")
    print(decoded)
    return decoded


@hydra.main(config_path="configs", config_name="default")
def main(cfg: DictConfig):
    device = cfg.training.device
    torch_dtype = getattr(torch, getattr(cfg.training, "amp_dtype", "bf16"), torch.bfloat16)
    model = FireboltVLForCausalLM.from_pretrained(
        cfg.evaluation.model_dir,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    tokenizer_dir = cfg.evaluation.get("tokenizer_dir") or cfg.evaluation.model_dir
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    processor = AutoProcessor.from_pretrained(cfg.processor_path)
    run_inference(
        cfg,
        model,
        tokenizer,
        processor,
        prompt=cfg.evaluation.prompt,
        image_path=cfg.evaluation.get("image_path") or "",
    )


if __name__ == "__main__":
    main()
