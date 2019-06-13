#!/usr/bin/env bash

# experiment50
# ------------
# Plots the t-SNE visualization for an MNIST SAE (no denoising).

python3 sdae.py \
    --batch_size 128 \
    --learning_rate 0.005 \
    --num_epochs 50 \
    --model_class MNISTSAE2 \
    --dataset_key mnist \
    --noise_type n/a \
    --weight_decay 0.0000001 \
    --save_path ./ckpt/mnist_sregae.pth

python3 manifold.py \
    --model_class MNISTSAE2 \
    --restore_path ./ckpt/mnist_sregae.pth \
    --dataset_key mnist \
    --batch_size 1000 \
    --fig_save_path mnist_tsne_reg.png
