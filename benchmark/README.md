# FireboltVL Benchmarks

## POPE (Object Hallucination)

Benchmark FireboltVL on [lmms-lab/POPE](https://huggingface.co/datasets/lmms-lab/POPE): yes/no object existence (hallucination check). Metrics: **overall accuracy** and **per-category accuracy** (adversarial, popular, random when using Full config).

### Usage

Run from the **project root** (FireboltVL):

```bash
# Default config, test split (~9k examples), with progress bar
python benchmark/benchmark_pope.py \
  --ckpt /path/to/fireboltvl/checkpoint \
  --tokenizer /path/to/tokenizer \
  --processor /path/to/processor \
  --show_progress

# Limit to N examples (e.g. quick sanity check)
python benchmark/benchmark_pope.py \
  --ckpt ... --tokenizer ... --processor ... \
  --limit 100 --show_progress

# Save per-sample results to CSV
python benchmark/benchmark_pope.py \
  --ckpt ... --tokenizer ... --processor ... \
  --save_csv results/pope_results.csv --show_progress

# Full dataset config, specific split
python benchmark/benchmark_pope.py \
  --ckpt ... --tokenizer ... --processor ... \
  --config Full --split adversarial --show_progress
```

### Arguments

| Argument | Description |
|----------|-------------|
| `--ckpt` | Path to FireboltVL checkpoint directory |
| `--tokenizer` | Tokenizer path (e.g. same as in config) |
| `--processor` | Vision processor path (e.g. SigLIP) |
| `--device` | `cuda` or `cpu` |
| `--dtype` | `bf16`, `fp16`, or `fp32` |
| `--max_new_tokens` | Max tokens for yes/no answer (default 32) |
| `--temperature` | 0 = greedy (default) |
| `--hf_repo` | Dataset repo (default `lmms-lab/POPE`) |
| `--config` | `default` or `Full` |
| `--split` | `test` (default) or for Full: `adversarial` / `popular` / `random` |
| `--limit` | Cap number of examples (0 = all) |
| `--save_csv` | Path to save per-sample CSV |
| `--show_progress` | Show tqdm progress bar |
| `--verbose` | Print each Q/A pair |

### Example (using your config paths)

```bash
python benchmark/benchmark_pope.py \
  --ckpt /home/mamba/ML_project/Testing/Huy/joint_vlm/FireboltVL/outputs/2026-02-06/11-13-38/fireboltvl_results/stage1/epoch_3 \
  --tokenizer /home/mamba/ML_project/Testing/Huy/joint_vlm/liquid_tokenizer_cot_fixed \
  --processor /home/mamba/ML_project/Testing/Huy/joint_vlm/viper-vlm/pretrained/siglip2_base_16_256 \
  --limit 50 \
  --show_progress
```
