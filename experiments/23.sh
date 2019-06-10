#!/usr/bin/env bash

# experiment23
# ------------
# Using the pretrained DVAE from experiment 22,
# trains and evaluates a DVAE + classifier on the MNIST classification task.

python3 mnist_classification.py \
    --batch_size 128 \
    --learning_rate 0.005 \
    --num_epochs 50 \
    --sae_model_class MNISTVAE \
    --sae_restore_path ./ckpt/dvae.pth \
    --sae_save_path ./stage2_dvae.pth \
    --classifier_model_class MNISTDenseClassifier2 \
    --classifier_save_path ./stage2_classifier.pth \
    --weight_decay 0.0000001 \
    --loss_type nll
