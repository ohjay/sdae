#!/usr/bin/env bash

# experiment05
# ------------
# Reproduces results from section 5.2 in Vincent et al.
# Trains a single-layer denoising autoencoder on MNIST digits.
# In this experiment, I use additive Gaussian noise.

python3 sdae.py \
    --batch_size 128 \
    --learning_rate 0.01 \
    --num_epochs 100 \
    --model_class MNISTAE \
    --dataset_key mnist \
    --noise_type gs \
    --gaussian_stdev 0.4 \
    --weight_decay 0.0000001
