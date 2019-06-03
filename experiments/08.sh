#!/usr/bin/env bash

# experiment08
# ------------
# 1. Pretrains a stacked denoising autoencoder layer by layer.
# 2. Trains the stacked denoising autoencoder on the MNIST classification task. (todo)

python3 sdae.py \
    --batch_size 128 \
    --learning_rate 0.005 \
    --num_epochs 50 \
    --model_key mnist_sae2 \
    --dataset mnist \
    --noise_type gs \
    --gaussian_stdev 0.4 \
    --weight_decay 0.0000001
