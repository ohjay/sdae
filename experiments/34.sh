#!/usr/bin/env bash

# experiment34
# ------------
# Trains experiment 32's CAE (+ a classifier) on the MNIST bg_rand classification task.

python3 mnist_classification.py \
    --batch_size 128 \
    --learning_rate 0.001 \
    --num_epochs 50 \
    --sae_model_class MNISTCAE2 \
    --sae_restore_path ./ckpt/cae.pth \
    --sae_save_path ./stage2_cae.pth \
    --classifier_model_class MNISTDenseClassifier2 \
    --classifier_save_path ./stage2_classifier.pth \
    --weight_decay 0.0000001 \
    --loss_type nll \
    --mnist_variant bg_rand
