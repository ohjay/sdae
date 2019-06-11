#!/usr/bin/env bash

# experiment32
# ------------
# Pretrains a convolutional autoencoder (CAE) on MNIST digits.
# Then trains the CAE, along with a supervised output head, to perform MNIST classification.

python3 sdae.py \
    --batch_size 128 \
    --learning_rate 0.005 \
    --num_epochs 50 \
    --model_class MNISTCAE2 \
    --dataset_key mnist \
    --noise_type n/a \
    --save_path ./ckpt/cae.pth \
    --weight_decay 0.0000001

python3 mnist_classification.py \
    --batch_size 128 \
    --learning_rate 0.001 \
    --num_epochs 50 \
    --sae_model_class MNISTCAE2 \
    --sae_restore_path ./ckpt/cae.pth \
    --sae_save_path ./stage2_sae.pth \
    --classifier_model_class MNISTDenseClassifier2 \
    --classifier_save_path ./stage2_classifier.pth \
    --weight_decay 0.0000001 \
    --loss_type nll
