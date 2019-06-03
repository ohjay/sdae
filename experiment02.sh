#!/usr/bin/env bash

# experiment02
# ------------
# Reproduces results from section 5.1 in Vincent et al.
# Trains a single-layer regular autoencoder on natural image patches.
# The learned weights won't end up having much meaningful structure.

python3 sdae.py \
    --batch_size 128 \
    --learning_rate 0.01 \
    --num_epochs 100 \
    --model_key olshausen_ae \
    --dataset olshausen \
    --noise_type n/a
