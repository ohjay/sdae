#!/usr/bin/env bash

# experiment33
# ------------
# Pretrains a convolutional denoising autoencoder (CDAE) on MNIST digits.
# Then trains the CDAE, along with a supervised output head, to perform MNIST classification.

python3 sdae.py \
    --batch_size 128 \
    --learning_rate 0.005 \
    --num_epochs 50 \
    --model_class MNISTCAE2 \
    --dataset_key mnist \
    --noise_type gs \
    --gaussian_stdev 0.4 \
    --save_path ./ckpt/cdae.pth \
    --weight_decay 0.0000001

python3 classification.py \
    --batch_size 128 \
    --learning_rate 0.001 \
    --num_epochs 50 \
    --sae_model_class MNISTCAE2 \
    --sae_restore_path ./ckpt/cdae.pth \
    --sae_save_path ./stage2_cdae.pth \
    --classifier_model_class MNISTDenseClassifier2 \
    --classifier_save_path ./stage2_classifier.pth \
    --weight_decay 0.0000001 \
    --loss_type nll \
    --dataset_key mnist
