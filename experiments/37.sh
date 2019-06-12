#!/usr/bin/env bash

# experiment37
# ------------
# Pretrains a convolutional denoising autoencoder (CDAE) on the CUB dataset.

python3 sdae.py \
    --batch_size 128 \
    --learning_rate 0.001 \
    --num_epochs 50 \
    --model_class CUBCAE2 \
    --dataset_key cub \
    --noise_type gs \
    --gaussian_stdev 0.4 \
    --save_path ./ckpt/cubcdae.pth \
    --weight_decay 0.0000001 \
    --cub_folder /home/owen/workspace/CUB_200_2011
