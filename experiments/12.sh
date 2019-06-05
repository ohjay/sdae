#!/usr/bin/env bash

# experiment12
# ------------
# Pretrains a stacked denoising autoencoder for future use.
# The trained SDAE should be saved in ./ckpt/stage1_sdae.pth.

python3 sdae.py \
    --batch_size 128 \
    --learning_rate 0.005 \
    --num_epochs 100 \
    --model_class MNISTSAE2 \
    --dataset mnist \
    --noise_type gs \
    --gaussian_stdev 0.4 \
    --weight_decay 0.0000001 \
    --save_path ./ckpt/stage1_sdae.pth
