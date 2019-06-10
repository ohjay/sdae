#!/usr/bin/env bash

# experiment22
# ------------
# Train a denoising variational autoencoder on MNIST.
# The trained DVAE should be saved in ./ckpt/dvae.pth.

python3 dvae.py \
    --batch_size 128 \
    --learning_rate 0.001 \
    --num_epochs 100 \
    --model_class MNISTVAE \
    --dataset_key mnist \
    --noise_type gs \
    --gaussian_stdev 0.4 \
    --save_path ./ckpt/dvae.pth \
    --weight_decay 0.0000001 \
    --reconstruction_loss_type bce
