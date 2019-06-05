#!/usr/bin/env bash

# experiment18
# ------------
# Generate samples using the SDAE trained during experiment 12.

python3 generate_samples.py \
    --model_class MNISTSAE2 \
    --dataset_key mnist \
    --restore_path ./ckpt/stage1_sdae.pth \
    --num_originals 10 \
    --num_variations 15 \
    --fig_save_path mnist_sdae_samples.png
