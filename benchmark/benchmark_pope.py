#!/usr/bin/env python3
# ----------------------------------------------------------------------------
# Benchmark FireboltVL on lmms-lab/POPE
# - Dataset: https://huggingface.co/datasets/lmms-lab/POPE
# - Configs: default (~9k test) or Full (adversarial/popular/random)
# - Task: binary yes/no object existence (hallucination check)
# - Prompt: <image> + question; force short yes/no
# - Metrics: overall accuracy + per-category accuracy (same as viper benchmark)
# - Optional CSV logging
# ----------------------------------------------------------------------------

import os
import re
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoProcessor, GenerationConfig

# Ensure project root is on path for FireboltVL imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modeling import FireboltVLForCausalLM

IMAGE_TOKEN_ID_FALLBACK = 64400


# =========================
# Normalization helpers (same as viper benchmark)
# =========================
PUNCT = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
ARTICLES = {"a", "an", "the"}
YES_SET = {"yes", "y", "yeah", "yep", "true", "correct", "right"}
NO_SET = {"no", "n", "nope", "false", "incorrect", "wrong"}


def _remove_punct(s: str) -> str:
    return s.translate(str.maketrans({c: " " for c in PUNCT}))


def normalize_free(s: str) -> str:
    """Lowercase, strip tags/punct/articles, collapse spaces."""
    s = (s or "").strip().lower()
    s = re.sub(r"<[^>]+>", " ", s)
    s = _remove_punct(s)
    s = " ".join([w for w in s.split() if w not in ARTICLES])
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_yesno(s: str) -> Optional[str]:
    """Map any variant to 'yes' or 'no'; return None if undecidable."""
    t = normalize_free(s)
    first = (t.split()[0] if t else "")
    if first in YES_SET:
        return "yes"
    if first in NO_SET:
        return "no"
    if t in YES_SET:
        return "yes"
    if t in NO_SET:
        return "no"
    return None


def final_line_token(raw: str) -> str:
    """Take the last non-empty line as the answer (for yes/no)."""
    if not raw:
        return ""
    lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
    return lines[-1].strip().strip(" .,:;!?)(") if lines else ""


# =========================
# Prompt builder
# =========================
def build_messages_pope(question: str) -> List[Dict]:
    """Single <image> token; keep instruction tight for yes/no."""
    q = question.strip()
    user = (
        f"<image> Question: {q}\n"
        "Answer with only 'yes' or 'no'."
    )
    return [{"role": "user", "content": user}]


# =========================
# Robust question extractor
# =========================
def extract_question(ex: Dict) -> str:
    """POPE may expose 'question' or similar; fallback to object-based synthesis."""
    for k in ("question", "query", "prompt", "text"):
        if k in ex and isinstance(ex[k], str) and ex[k].strip():
            return ex[k]
    if "object" in ex and isinstance(ex["object"], str) and ex["object"].strip():
        return f"Is there a {ex['object'].strip()} in the image?"
    raise KeyError("Could not locate a question field in this POPE example.")


# =========================
# Inference (FireboltVL)
# =========================
@torch.inference_mode()
def generate_answer(
    model: FireboltVLForCausalLM,
    tokenizer: AutoTokenizer,
    processor: AutoProcessor,
    image: Image.Image,
    messages: List[Dict],
    device: torch.device,
    amp_dtype: Optional[torch.dtype] = None,
    max_new_tokens: int = 8,
    temperature: float = 0.0,
    top_p: float = 0.9,
    repetition_penalty: float = 1.0,
) -> str:
    enc = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
    )
    if isinstance(enc, torch.Tensor):
        input_ids = enc
    else:
        input_ids = enc["input_ids"]
    input_ids = input_ids.to(device)

    image_rgb = image.convert("RGB")
    proc_out = processor(images=[image_rgb], return_tensors="pt")
    pixel_values = proc_out.get("pixel_values")
    if pixel_values is None:
        raise ValueError("Processor did not return 'pixel_values'.")
    model_dtype = next(model.parameters()).dtype
    pixel_values = pixel_values.to(device, dtype=model_dtype)

    image_token_id = getattr(model.config, "image_token_id", None)
    if image_token_id is None:
        if "<image>" in tokenizer.get_vocab():
            image_token_id = tokenizer.convert_tokens_to_ids("<image>")
        else:
            image_token_id = IMAGE_TOKEN_ID_FALLBACK

    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0.0,
        temperature=max(temperature, 1e-6),
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=False,
    )

    gen_kwargs = {"pixel_values": pixel_values}
    if amp_dtype is not None and device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            gen_ids = model.generate(
                input_ids, generation_config=gen_cfg, **gen_kwargs
            )
    else:
        gen_ids = model.generate(
            input_ids, generation_config=gen_cfg, **gen_kwargs
        )

    prompt_len = input_ids.size(1)
    new_tokens = gen_ids[0, prompt_len:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return answer.strip()


# =========================
# Eval harness
# =========================
def evaluate_dataset(
    ds,
    model,
    tokenizer,
    processor,
    device,
    amp_dtype,
    max_new_tokens,
    temperature,
    top_p,
    repetition_penalty,
    show_progress: bool = False,
    save_rows: bool = False,
    verbose: bool = False,
):
    gen_args = dict(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )

    total = 0
    correct = 0
    by_cat = {}
    rows = []

    iterator = tqdm(ds, desc="Evaluating", total=len(ds)) if show_progress else ds

    for ex in iterator:
        img = ex.get("image", None)
        if not isinstance(img, Image.Image):
            img = Image.new("RGB", (224, 224), (255, 255, 255))

        gt_raw = ex.get("answer", "")
        gt = normalize_yesno(gt_raw)
        category = ex.get("category", "")

        try:
            q = extract_question(ex)
        except Exception:
            q = "Is the mentioned object present in the image?"

        messages = build_messages_pope(q)
        raw = generate_answer(
            model, tokenizer, processor, img, messages, device, amp_dtype, **gen_args
        )
        pred = final_line_token(raw)
        pred_norm = normalize_yesno(pred)

        if verbose:
            print(f"Q: {q}\nGT: {gt_raw} | Pred: {pred} | Norm: {pred_norm}\n")

        is_ok = int(
            pred_norm is not None and gt is not None and pred_norm == gt
        )
        total += 1
        correct += is_ok

        if category not in by_cat:
            by_cat[category] = [0, 0]
        by_cat[category][0] += is_ok
        by_cat[category][1] += 1

        if show_progress:
            iterator.set_postfix({"acc": f"{(correct / total) * 100:.2f}%"})

        if save_rows:
            rows.append({
                "question": q,
                "answer_gt_raw": gt_raw,
                "answer_gt": gt or "",
                "pred_raw": raw,
                "pred_final": pred,
                "pred_norm": pred_norm or "",
                "category": category,
                "is_correct": is_ok,
                "image_source": ex.get("image_source", ""),
            })

    overall_acc = (correct / total) if total else 0.0
    cat_stats = {
        k: (v[0] / v[1] if v[1] else 0.0, v[1])
        for k, v in by_cat.items()
    }
    return overall_acc, cat_stats, rows


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser(
        description="Benchmark FireboltVL on POPE (yes/no object hallucination)"
    )
    parser.add_argument(
        "--ckpt",
        required=True,
        help="Path to FireboltVL checkpoint (dir with config + safetensors/bin)",
    )
    parser.add_argument("--tokenizer", required=True, help="Tokenizer path or HF id")
    parser.add_argument("--processor", required=True, help="Processor path (e.g. SigLIP)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)

    parser.add_argument("--hf_repo", default="lmms-lab/POPE")
    parser.add_argument("--config", default="default", choices=["default", "Full"])
    parser.add_argument(
        "--split",
        default="test",
        help="For Full: adversarial|popular|random; for default: test",
    )
    parser.add_argument("--limit", type=int, default=0, help="0 = all examples")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument(
        "--save_csv",
        type=str,
        default="",
        help="Path to save per-sample results CSV",
    )
    parser.add_argument("--show_progress", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="Print each Q/A")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_bf16 = args.dtype.lower() == "bf16"
    use_fp16 = args.dtype.lower() == "fp16"
    amp_dtype = (
        torch.bfloat16
        if use_bf16
        else (torch.float16 if use_fp16 else None)
    )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    processor = AutoProcessor.from_pretrained(args.processor)

    model = FireboltVLForCausalLM.from_pretrained(
        args.ckpt,
        torch_dtype=(
            amp_dtype if (amp_dtype and device.type == "cuda") else torch.float32
        ),
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    ds = load_dataset(args.hf_repo, args.config, split=args.split)
    if args.shuffle:
        ds = ds.shuffle(seed=args.seed)
    if args.limit and args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))

    print(
        f"Loaded {len(ds)} examples from {args.hf_repo} — config='{args.config}' split='{args.split}'"
    )

    overall_acc, cat_stats, rows = evaluate_dataset(
        ds,
        model,
        tokenizer,
        processor,
        device,
        amp_dtype,
        args.max_new_tokens,
        args.temperature,
        args.top_p,
        args.repetition_penalty,
        show_progress=args.show_progress,
        save_rows=bool(args.save_csv),
        verbose=args.verbose,
    )

    print("\n=== POPE Accuracy (yes/no) ===")
    print(f"Config: {args.config} | Split: {args.split}")
    print(f"Overall: {overall_acc * 100:.2f}% over {len(ds)} examples")
    if cat_stats:
        print("\nPer-category:")
        for cat, (acc, n) in sorted(cat_stats.items()):
            print(f"  - {cat:12s}: {acc * 100:6.2f}%  (n={n})")

    if args.save_csv and rows:
        try:
            import pandas as pd
            Path(args.save_csv).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(args.save_csv, index=False)
            print(f"\nSaved detailed results to: {args.save_csv}")
        except Exception as e:
            print(f"\n[WARN] Could not save CSV: {e}")


if __name__ == "__main__":
    main()
