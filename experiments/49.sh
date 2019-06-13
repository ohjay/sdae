#!/usr/bin/env bash

# experiment49
# ------------
# Plots the t-SNE visualization for the MNIST SDAE from before.

python3 manifold.py \
    --model_class MNISTSAE2 \
    --restore_path ./ckpt/mnist_sae.pth \
    --dataset_key mnist \
    --batch_size 1000 \
    --fig_save_path mnist_tsne.png
