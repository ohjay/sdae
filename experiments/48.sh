#!/usr/bin/env bash

# experiment48
# ------------
# Generate samples using the SDAE trained during experiment 47.

python3 generate_samples.py \
    --model_class DandelionSAE \
    --restore_path ./ckpt/dandelion_sae.pth \
    --dataset_key interp \
    --num 10 \
    --sample_h 128 \
    --sample_w 128 \
    --fig_save_path interp_gen_samples.png \
    --lower -1 \
    --upper 2
