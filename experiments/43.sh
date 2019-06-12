#!/usr/bin/env bash

# experiment43
# ------------
# Pretrains a convolutional denoising autoencoder (CDAE) on the CIFAR-10 dataset.

python3 sdae.py \
    --batch_size 128 \
    --learning_rate 0.0005 \
    --num_epochs 100 \
    --model_class CIFARCAE \
    --dataset_key cifar10 \
    --noise_type gs \
    --gaussian_stdev 0.4 \
    --save_path ./ckpt/cifarcdae.pth \
    --weight_decay 0.0000001
