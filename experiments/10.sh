#!/usr/bin/env bash

# experiment10
# ------------
# Trains a classifier on the MNIST classification task.
# Differences between this experiment and experiment08:
# - This has no SAE architecture at all
# - This has no unsupervised pretraining

python3 classification.py \
    --batch_size 128 \
    --learning_rate 0.005 \
    --num_epochs 50 \
    --sae_restore_path n/a \
    --classifier_model_class MNISTDenseClassifier2 \
    --classifier_save_path ./stage2_classifier.pth \
    --weight_decay 0.0000001 \
    --loss_type nll \
    --no_sae \
    --dataset_key mnist
