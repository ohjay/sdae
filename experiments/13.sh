#!/usr/bin/env bash

# experiment13
# ------------
# Pretrains a stacked REGULAR autoencoder for future use.
# The trained SAE should be saved in ./ckpt/stage1_sae.pth.

python3 sdae.py \
    --batch_size 128 \
    --learning_rate 0.005 \
    --num_epochs 100 \
    --model_class MNISTSAE2 \
    --dataset mnist \
    --noise_type n/a \
    --weight_decay 0.0000001 \
    --save_path ./ckpt/stage1_sae.pth
