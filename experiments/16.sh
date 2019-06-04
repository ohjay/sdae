#!/usr/bin/env bash

# experiment16
# ------------
# Trains a stacked encoder + classifier on the MNIST rot classification task.

python3 mnist_classification.py \
    --batch_size 128 \
    --learning_rate 0.005 \
    --num_epochs 50 \
    --sae_model_key mnist_sae2 \
    --sae_restore_path ./ckpt/stage1_sdae_rot.pth \
    --sae_save_path ./stage2_sae.pth \
    --classifier_model_key mnist_dense_classifier2 \
    --classifier_save_path ./stage2_classifier.pth \
    --weight_decay 0.0000001 \
    --loss_type nll \
    --mnist_variant rot
