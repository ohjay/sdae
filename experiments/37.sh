#!/usr/bin/env bash

# experiment37
# ------------
# Pretrains a convolutional autoencoder (CAE) on the CUB dataset.

python3 sdae.py \
    --batch_size 128 \
    --learning_rate 0.0002 \
    --num_epochs 100 \
    --model_class CUBCAE2 \
    --dataset_key cub \
    --noise_type n/a \
    --save_path ./ckpt/cubcae.pth \
    --weight_decay 0.0000001 \
    --cub_folder /home/owen/workspace/CUB_200_2011
