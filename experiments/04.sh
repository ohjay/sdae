#!/usr/bin/env bash

# experiment04
# ------------
# Reproduces results from section 5.1 in Vincent et al.
# Trains a single-layer denoising autoencoder on natural image patches.
# This experiment uses zero-masking noise (as opposed to Gaussian/S&P noise).

python3 sdae.py \
    --batch_size 128 \
    --learning_rate 0.004 \
    --num_epochs 500 \
    --model_class OlshausenAE \
    --dataset_key olshausen \
    --olshausen_path /home/owen/workspace/sdae/data/natural/images.mat \
    --olshausen_step_size 6 \
    --noise_type mn \
    --zero_frac 0.65
