CUDA_VISIBLE_DEVICES=0 python predict.py \
    --image /home/mamba/ML_project/Testing/Huy/joint_vlm/Viper-LM/examples/messi-1805.jpg \
    --prompt "What are in the image?" \
    --max_tokens 1024 \
    --top_p 0.9 \
    --temperature 0.7 \