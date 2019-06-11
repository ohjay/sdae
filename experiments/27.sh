#!/usr/bin/env bash

# experiment27
# ------------
# Train a stacked denoising variational autoencoder on MNIST.
# The trained SDVAE should be saved in ./ckpt/sdvae.pth.

python3 sdae.py \
    --batch_size 128 \
    --learning_rate 0.001 \
    --num_epochs 50 \
    --model_class MNISTSVAE \
    --dataset_key mnist \
    --noise_type gs \
    --gaussian_stdev 0.4 \
    --save_path ./ckpt/sdvae.pth \
    --weight_decay 0.0000001 \
    --vae_reconstruction_loss_type bce
