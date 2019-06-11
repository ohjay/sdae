#!/usr/bin/env bash

# experiment28
# ------------
# Using the pretrained SVAE from experiment 26,
# trains and evaluates a SVAE + classifier on the MNIST bg_rand classification task.

python3 mnist_classification.py \
    --batch_size 128 \
    --learning_rate 0.001 \
    --num_epochs 75 \
    --sae_model_class MNISTSVAE \
    --sae_restore_path ./ckpt/svae.pth \
    --sae_save_path ./stage2_svae.pth \
    --classifier_model_class MNISTDenseClassifier2 \
    --classifier_save_path ./stage2_classifier.pth \
    --weight_decay 0.0000001 \
    --loss_type nll \
    --mnist_variant bg_rand
