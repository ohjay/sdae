#!/usr/bin/env bash

# experiment45
# ------------
# Pretrains an SDAE using learned noise transformations.

python3 sdae.py \
    --batch_size 128 \
    --learning_rate 0.005 \
    --num_epochs 50 \
    --model_class MNISTSAE2 \
    --dataset_key mnist \
    --weight_decay 0.0000001 \
    --save_path ./stage1_sae.pth \
    --learned_noise_wt 0.01 \
    --nt_save_prefix ./ckpt/mnist_sae_nt
