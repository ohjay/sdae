#!/usr/bin/env bash

# experiment06
# ------------
# Reproduces results from section 5.2 in Vincent et al.
# Trains a single-layer denoising autoencoder on MNIST digits.
# In this experiment, I use zero-masking noise.

python3 sdae.py \
    --batch_size 128 \
    --learning_rate 0.004 \
    --num_epochs 100 \
    --model_key mnist_ae \
    --dataset mnist \
    --noise_type mn \
    --zero_frac 0.65 \
    --weight_decay 0.0000001
