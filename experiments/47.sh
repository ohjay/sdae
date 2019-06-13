#!/usr/bin/env bash

# experiment47
# ------------
# Pretrains an SDAE on the grayscale interpolation dataset.

python3 sdae.py \
    --batch_size 128 \
    --learning_rate 0.002 \
    --num_epochs 50 \
    --model_class DandelionSAE \
    --dataset_key interp \
    --weight_decay 0.0000001 \
    --save_path ./ckpt/dandelion_sae.pth
