#!/usr/bin/env bash

# experiment46
# ------------
# Trains (experiment 45's SDAE) + (a classifier) on the MNIST classification task.

python3 classification.py \
    --batch_size 128 \
    --learning_rate 0.005 \
    --num_epochs 50 \
    --sae_model_class MNISTSAE2 \
    --sae_restore_path ./ckpt/mnist_sae.pth \
    --sae_save_path ./stage2_mnist_sae.pth \
    --classifier_model_class MNISTDenseClassifier2 \
    --classifier_save_path ./stage2_classifier.pth \
    --weight_decay 0.0000001 \
    --loss_type nll \
    --dataset_key mnist
