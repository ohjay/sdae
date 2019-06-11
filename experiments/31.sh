#!/usr/bin/env bash

# experiment31
# ------------
# Generate samples using the DVAE trained during experiment 22.

python3 generate_samples.py \
    --model_class MNISTVAE \
    --restore_path ./ckpt/dvae.pth \
    --num 10 \
    --sample_h 28 \
    --sample_w 28 \
    --fig_save_path mnist_dvae_samples.png
