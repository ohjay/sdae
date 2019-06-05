#!/usr/bin/env bash

# experiment11
# ------------
# 1. Pretrains a stacked REGULAR autoencoder in a layer-by-layer fashion.
# 2. Trains the stacked encoder + classifier on the MNIST classification task.

python3 sdae.py \
    --batch_size 128 \
    --learning_rate 0.005 \
    --num_epochs 50 \
    --model_class MNISTSAE2 \
    --dataset_key mnist \
    --noise_type n/a \
    --weight_decay 0.0000001 \
    --save_path ./stage1_sae.pth

python3 mnist_classification.py \
    --batch_size 128 \
    --learning_rate 0.005 \
    --num_epochs 20 \
    --sae_model_class MNISTSAE2 \
    --sae_restore_path ./stage1_sae.pth \
    --sae_save_path ./stage2_sae.pth \
    --classifier_model_class MNISTDenseClassifier2 \
    --classifier_save_path ./stage2_classifier.pth \
    --weight_decay 0.0000001 \
    --loss_type nll
