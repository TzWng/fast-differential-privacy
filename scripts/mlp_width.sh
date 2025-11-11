#!/bin/bash
PYTHON=python3.10

PROJECT_ROOT=/content/fast-differential-privacy
export PYTHONPATH="$PROJECT_ROOT"


LRS=(-7.5 -7 -6.5 -6 -5.5) # SGD

for wid in 256 512 1024 2048 4096; do
  for lr in "${LRS[@]}"; do
    sig=$(awk "BEGIN {print sqrt(256.0/$wid)}")
    echo "Running width=$wid, lr=$lr, noise=$sig"
    $PYTHON -m scripts.MLP_unifed \
      --width "$wid" \
      --lr "$lr" \
      --epochs 20\
      --bs 500 \
      --mini_bs 500 \
      --epsilon 2 \
      --noise "$sig" \
      --clipping_mode BK-MixOpt \
      --clipping_style layer-wise \
      --cifar_data CIFAR10 \
      --dimension 32 \
      --optimizer SGD \
      --log_path "/content/drive/MyDrive/DP_muP/logs/MLP_SGD_diffwidth_truenorm_ratio_1.txt"
  done
done
