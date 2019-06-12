#!/usr/bin/env bash

# experiment38
# ------------
# Trains experiment 37's CDAE (+ a classifier) to classify birds.

python3 classification.py \
    --batch_size 128 \
    --learning_rate 0.0001 \
    --num_epochs 50 \
    --sae_model_class CUBCAE2 \
    --sae_restore_path ./ckpt/cubcdae.pth \
    --sae_save_path ./stage2_cubcdae.pth \
    --classifier_model_class CUBDenseClassifier3 \
    --classifier_save_path ./stage2_classifier.pth \
    --weight_decay 0.0000001 \
    --loss_type nll \
    --dataset_key cub \
    --cub_folder /home/owen/workspace/CUB_200_2011
