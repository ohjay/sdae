#!/usr/bin/env bash

# experiment03
# ------------
# Reproduces results from section 5.1 in Vincent et al.
# Trains a single-layer denoising autoencoder on natural image patches.
# This experiment uses salt-and-pepper noise (instead of additive Gaussian noise).

python3 sdae.py \
    --batch_size 128 \
    --learning_rate 0.006 \
    --num_epochs 100 \
    --model_class OlshausenAE \
    --dataset_key olshausen \
    --olshausen_path /home/owen/workspace/sdae/data/natural/images.mat \
    --olshausen_step_size 6 \
    --noise_type sp \
    --sp_frac 0.25
