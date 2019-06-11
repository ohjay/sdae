#!/usr/bin/env bash

# experiment30
# ------------
# Generate samples using the VAE trained during experiment 20.

python3 generate_samples.py \
    --model_class MNISTVAE \
    --restore_path ./ckpt/vae.pth \
    --num 10 \
    --sample_h 28 \
    --sample_w 28 \
    --fig_save_path mnist_vae_samples.png
