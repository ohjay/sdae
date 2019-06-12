#!/usr/bin/env bash

# experiment42
# ------------
# Trains experiment 41's CAE (+ a classifier) to perform CIFAR-10 classification.

python3 classification.py \
    --batch_size 128 \
    --learning_rate 0.001 \
    --num_epochs 100 \
    --sae_model_class CIFARCAE \
    --sae_restore_path ./ckpt/cifarcae.pth \
    --sae_save_path ./stage2_cifarcae.pth \
    --classifier_model_class MNISTDenseClassifier2 \
    --classifier_save_path ./stage2_classifier.pth \
    --weight_decay 0.0000001 \
    --loss_type nll \
    --dataset_key cifar10
