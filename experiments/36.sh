#!/usr/bin/env bash

# experiment36
# ------------
# Like experiment 35, but with a convolutional classifier.

python3 classification.py \
    --batch_size 128 \
    --learning_rate 0.001 \
    --num_epochs 50 \
    --sae_model_class MNISTCAE2 \
    --sae_restore_path ./ckpt/cdae.pth \
    --sae_save_path ./stage2_cdae.pth \
    --classifier_model_class MNISTConvClassifier4 \
    --classifier_save_path ./stage2_classifier.pth \
    --weight_decay 0.0000001 \
    --loss_type nll \
    --dataset_key bg_rand
