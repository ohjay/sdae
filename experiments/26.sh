#!/usr/bin/env bash

# experiment26
# ------------
# Train a stacked regular variational autoencoder on MNIST.
# The trained SVAE should be saved in ./ckpt/svae.pth.

python3 sdae.py \
    --batch_size 128 \
    --learning_rate 0.001 \
    --num_epochs 50 \
    --model_class MNISTSVAE \
    --dataset_key mnist \
    --noise_type n/a \
    --save_path ./ckpt/svae.pth \
    --weight_decay 0.0000001 \
    --vae_reconstruction_loss_type bce
