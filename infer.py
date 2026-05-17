#!/usr/bin/env python3
# Inference FireboltVL
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch

torch.backends.cudnn.enabled = False
if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
    torch.backends.cuda.enable_cudnn_sdp(False)

import yaml
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedTokenizerFast,
)

from modeling import FireboltVLForCausalLM


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "stage2.yaml"
DEFAULT_CKPT = ROOT / "fireboltvl_results1" / "stage2" / "epoch_1"
DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


def read_yaml(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def deep_get(data: Dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    cur: Any = data
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def apply_override(data: Dict[str, Any], dotted_key: str, value: str) -> None:
    cur = data
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = yaml.safe_load(value)


def unique_existing(candidates: Iterable[Optional[str]]) -> List[str]:
    seen = set()
    out = []
    for candidate in candidates:
        if not candidate:
            continue
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def load_tokenizer(candidates: Iterable[Optional[str]], local_files_only: bool = True) -> AutoTokenizer:
    errors = []
    for candidate in unique_existing(candidates):
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                candidate,
                use_fast=True,
                local_files_only=local_files_only,
            )
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")
            tokenizer_json = Path(candidate) / "tokenizer.json"
            if not tokenizer_json.is_file():
                continue
            tokenizer_config = read_json(Path(candidate) / "tokenizer_config.json")
            try:
                tokenizer = PreTrainedTokenizerFast(
                    tokenizer_file=str(tokenizer_json),
                    bos_token=tokenizer_config.get("bos_token"),
                    eos_token=tokenizer_config.get("eos_token"),
                    pad_token=tokenizer_config.get("pad_token"),
                    unk_token=tokenizer_config.get("unk_token"),
                )
            except Exception as fallback_exc:
                errors.append(f"{candidate} tokenizer.json fallback: {fallback_exc}")
                continue

            chat_template = Path(candidate) / "chat_template.jinja"
            if chat_template.is_file():
                tokenizer.chat_template = chat_template.read_text(encoding="utf-8")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        return tokenizer

    details = "\n".join(errors) if errors else "No tokenizer paths were provided."
    raise RuntimeError(f"Could not load a tokenizer.\n{details}")


def load_processor(candidates: Iterable[Optional[str]], local_files_only: bool = True) -> AutoProcessor:
    errors = []
    for candidate in unique_existing(candidates):
        try:
            return AutoProcessor.from_pretrained(candidate, local_files_only=local_files_only)
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")
    details = "\n".join(errors) if errors else "No processor paths were provided."
    raise RuntimeError(f"Could not load an image processor.\n{details}")


def load_model(model_dir: str, device: torch.device, torch_dtype: torch.dtype):
    return FireboltVLForCausalLM.load_model(
        model_dir,
        device=str(device),
        torch_dtype=torch_dtype,
    )


def load_images(image_paths: List[str]) -> List[Image.Image]:
    images = []
    for image_path in image_paths:
        path = Path(image_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        images.append(Image.open(path).convert("RGB"))
    return images


@torch.inference_mode()
def run_inference(
    model,
    tokenizer,
    processor,
    prompt: str,
    image_paths: List[str],
    device: torch.device,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
):
    images = load_images(image_paths)
    image_prefix = "".join("<image>\n" for _ in images)
    messages = [{"role": "user", "content": f"{image_prefix}{prompt}"}]
    tokenized = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    )
    if isinstance(tokenized, torch.Tensor):
        input_ids = tokenized.to(device)
        attention_mask = torch.ones_like(input_ids, device=device)
    else:
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized.get("attention_mask")
        attention_mask = (
            attention_mask.to(device)
            if attention_mask is not None
            else torch.ones_like(input_ids, device=device)
        )

    gen_kwargs = {"attention_mask": attention_mask}
    if images:
        pixel_values = processor(images=images, return_tensors="pt").get("pixel_values")
        if pixel_values is None:
            raise ValueError("Processor did not return 'pixel_values'.")
        model_dtype = next(model.parameters()).dtype
        gen_kwargs["pixel_values"] = pixel_values.unsqueeze(0).to(device, dtype=model_dtype)
        gen_kwargs["image_counts"] = torch.tensor([len(images)], dtype=torch.long, device=device)

    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=False,
    )

    model_dtype = next(model.parameters()).dtype
    with torch.autocast(
        device_type="cuda",
        dtype=model_dtype,
        enabled=device.type == "cuda" and model_dtype in (torch.bfloat16, torch.float16),
    ):
        gen_ids = model.generate(input_ids, generation_config=gen_cfg, **gen_kwargs)

    new_tokens = gen_ids[0, input_ids.shape[1] :]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    print("\n--- Prompt ---")
    print(prompt)
    for image_path in image_paths:
        print(f"(image: {image_path})")
    print("\n--- Response ---")
    print(decoded)
    return decoded


def parse_args():
    parser = argparse.ArgumentParser(description="Run FireboltVL inference.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML config path")
    parser.add_argument("--ckpt", "--model_dir", dest="model_dir", default=None, help="Checkpoint directory")
    parser.add_argument("--tokenizer", "--tokenizer_dir", dest="tokenizer_dir", default=None)
    parser.add_argument("--processor", "--processor_path", dest="processor_path", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--image", "--image_path", dest="image_paths", action="append", default=[])
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None, choices=sorted(DTYPE_MAP))
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument(
        "--allow_remote",
        action="store_true",
        help="Allow Hugging Face downloads/checks instead of forcing local cached files.",
    )
    args, overrides = parser.parse_known_args()
    return args, overrides


def main():
    args, overrides = parse_args()
    cfg_path = Path(args.config).expanduser() if args.config else None
    cfg = read_yaml(cfg_path)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Unknown argument: {override}")
        key, value = override.split("=", 1)
        apply_override(cfg, key, value)

    model_dir = args.model_dir or deep_get(cfg, "evaluation.model_dir") or str(DEFAULT_CKPT)
    model_dir = str(Path(model_dir).expanduser())
    ckpt_cfg = read_yaml(Path(model_dir) / "training_config.yaml")
    ckpt_model_config = read_json(Path(model_dir) / "config.json")
    local_files_only = not args.allow_remote
    if local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    device_name = args.device or deep_get(cfg, "training.device") or deep_get(ckpt_cfg, "training.device") or "cuda"
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available; running on CPU.", file=sys.stderr)
        device_name = "cpu"
    device = torch.device(device_name)

    dtype_name = args.dtype or deep_get(cfg, "training.amp_dtype") or deep_get(ckpt_cfg, "training.amp_dtype") or "bf16"
    torch_dtype = DTYPE_MAP[dtype_name]
    if device.type == "cpu" and torch_dtype in (torch.bfloat16, torch.float16):
        torch_dtype = torch.float32

    tokenizer_dir = args.tokenizer_dir or deep_get(cfg, "evaluation.tokenizer_dir")
    processor_path = args.processor_path or deep_get(cfg, "processor_path")
    prompt = args.prompt or deep_get(cfg, "evaluation.prompt") or "Describe the image."
    image_paths = args.image_paths or [deep_get(cfg, "evaluation.image_path")]
    image_paths = [p for p in image_paths if p]

    max_new_tokens = args.max_new_tokens or deep_get(cfg, "evaluation.max_new_tokens") or 96
    temperature = args.temperature if args.temperature is not None else deep_get(cfg, "evaluation.temperature", 1.0)
    top_p = args.top_p if args.top_p is not None else deep_get(cfg, "evaluation.top_p", 1.0)
    repetition_penalty = (
        args.repetition_penalty
        if args.repetition_penalty is not None
        else deep_get(cfg, "evaluation.repetition_penalty", 1.05)
    )
    do_sample = args.do_sample or bool(deep_get(cfg, "evaluation.do_sample", False))

    print(f"Loading model: {model_dir}", flush=True)
    model = load_model(model_dir, device=device, torch_dtype=torch_dtype)
    tokenizer = load_tokenizer(
        [
            tokenizer_dir,
            deep_get(ckpt_cfg, "evaluation.tokenizer_dir"),
            deep_get(cfg, "tokenizer_path"),
            deep_get(ckpt_cfg, "tokenizer_path"),
            model_dir,
        ],
        local_files_only=local_files_only,
    )
    processor = load_processor(
        [
            processor_path,
            deep_get(ckpt_cfg, "processor_path"),
            ckpt_model_config.get("vision_ckpt_path"),
        ],
        local_files_only=local_files_only,
    )

    run_inference(
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        prompt=prompt,
        image_paths=image_paths,
        device=device,
        max_new_tokens=int(max_new_tokens),
        do_sample=do_sample,
        temperature=float(temperature),
        top_p=float(top_p),
        repetition_penalty=float(repetition_penalty),
    )


if __name__ == "__main__":
    main()
