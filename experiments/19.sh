#!/usr/bin/env bash

# experiment19
# ------------
# Pretrains a stacked denoising autoencoder on the Olshausen dataset.
# The trained SDAE should be saved in ./ckpt/stage1_sdae_olshausen.pth.
# Also runs sample generation using the pretrained Olshausen autoencoder.

python3 sdae.py \
    --batch_size 128 \
    --learning_rate 0.005 \
    --num_epochs 20 \
    --model_class OlshausenSAE3 \
    --dataset_key olshausen \
    --olshausen_path /home/owen/workspace/sdae/data/natural/images.mat \
    --olshausen_step_size 1 \
    --noise_type gs \
    --gaussian_stdev 0.75 \
    --weight_decay 0.0000001 \
    --save_path ./ckpt/stage1_sdae_olshausen.pth

python3 generate_samples.py \
    --model_class OlshausenSAE3 \
    --dataset_key olshausen \
    --restore_path ./ckpt/stage1_sdae_olshausen.pth \
    --olshausen_path /home/owen/workspace/sdae/data/natural/images.mat \
    --olshausen_step_size 1 \
    --num_originals 10 \
    --num_variations 15 \
    --fig_save_path olshausen_sdae_samples.png
