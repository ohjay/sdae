#!/usr/bin/env bash

# experiment07
# ------------
# Reproduces results from section 5.2 in Vincent et al.
# Trains a single-layer regular autoencoder on MNIST digits.
# The learned filters are less interesting than the ones from the previous two experiments.

python3 sdae.py \
    --batch_size 128 \
    --learning_rate 0.01 \
    --num_epochs 100 \
    --model_class MNISTAE \
    --dataset_key mnist \
    --noise_type n/a \
    --weight_decay 0.0000001
