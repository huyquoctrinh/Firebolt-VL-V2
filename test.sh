# Quick run (e.g. 50 samples)
python benchmark/benchmark_pope.py \
  --ckpt /home/mamba/ML_project/Testing/Huy/joint_vlm/FireboltVL/fireboltvl_results/stage2/epoch_1 \
  --tokenizer /home/mamba/ML_project/Testing/Huy/joint_vlm/liquid_tokenizer_cot_fixed \
  --processor /home/mamba/ML_project/Testing/Huy/joint_vlm/viper-vlm/pretrained/siglip2_base_16_256 \
  --limit 50 \
  --show_progress \
  --verbose

# Full test set
# python benchmark/benchmark_pope.py \
#   --ckpt ... --tokenizer ... --processor ... \
#   --show_progress

# # Save results to CSV
# python benchmark/benchmark_pope.py \
#   --ckpt ... --tokenizer ... --processor ... \
#   --save_csv ./benchmark/pope_results.csv --show_progress