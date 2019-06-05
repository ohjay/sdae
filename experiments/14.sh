#!/usr/bin/env bash

# experiment14
# ------------
# Pretrains a stacked denoising autoencoder on MNIST rot.
# The trained SDAE should be saved in ./ckpt/stage1_sdae_rot.pth.

python3 sdae.py \
    --batch_size 128 \
    --learning_rate 0.005 \
    --num_epochs 100 \
    --model_class MNISTSAE2 \
    --dataset_key rot \
    --noise_type gs \
    --gaussian_stdev 0.4 \
    --weight_decay 0.0000001 \
    --save_path ./ckpt/stage1_sdae_rot.pth
