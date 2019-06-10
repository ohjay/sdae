#!/usr/bin/env bash

# experiment20
# ------------
# Train a regular variational autoencoder on MNIST.
# The trained VAE should be saved in ./ckpt/vae.pth.

python3 dvae.py \
    --batch_size 128 \
    --learning_rate 0.001 \
    --num_epochs 100 \
    --model_class MNISTVAE \
    --dataset_key mnist \
    --noise_type n/a \
    --save_path ./ckpt/vae.pth \
    --weight_decay 0.0000001 \
    --reconstruction_loss_type mse
