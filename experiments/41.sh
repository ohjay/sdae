#!/usr/bin/env bash

# experiment41
# ------------
# Pretrains a convolutional autoencoder (CAE) on the CIFAR-10 dataset.

python3 sdae.py \
    --batch_size 128 \
    --learning_rate 0.0005 \
    --num_epochs 100 \
    --model_class CIFARCAE \
    --dataset_key cifar10 \
    --noise_type n/a \
    --save_path ./ckpt/cifarcae.pth \
    --weight_decay 0.0000001
