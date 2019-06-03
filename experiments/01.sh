#!/usr/bin/env bash

# experiment01
# ------------
# Reproduces results from section 5.1 in Vincent et al.
# Trains a single-layer denoising autoencoder on natural image patches.
# The learned weights end up resembling Gabor filters (/edge detectors).

python3 sdae.py \
    --batch_size 128 \
    --learning_rate 0.01 \
    --num_epochs 100 \
    --model_key olshausen_ae \
    --dataset olshausen \
    --olshausen_path /home/owen/workspace/sdae/data/natural/images.mat \
    --olshausen_step_size 1 \
    --noise_type gs \
    --gaussian_stdev 0.4
