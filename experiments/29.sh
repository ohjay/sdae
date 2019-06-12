#!/usr/bin/env bash

# experiment29
# ------------
# Using the pretrained SDVAE from experiment 27,
# trains and evaluates a SDVAE + classifier on the MNIST bg_rand classification task.

python3 classification.py \
    --batch_size 128 \
    --learning_rate 0.001 \
    --num_epochs 75 \
    --sae_model_class MNISTSVAE \
    --sae_restore_path ./ckpt/sdvae.pth \
    --sae_save_path ./stage2_sdvae.pth \
    --classifier_model_class MNISTDenseClassifier2 \
    --classifier_save_path ./stage2_classifier.pth \
    --weight_decay 0.0000001 \
    --loss_type nll \
    --dataset_key bg_rand
