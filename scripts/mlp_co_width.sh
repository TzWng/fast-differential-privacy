#!/bin/bash
PYTHON=python3.10

PROJECT_ROOT=/content/fast-differential-privacy
export PYTHONPATH="$PROJECT_ROOT"


LRS=(-1.5 -1 -0.5 0) # SGD

for wid in 256; do
  for lr in "${LRS[@]}"; do
    echo "Running width=$wid, lr=$lr"
    $PYTHON -m scripts.MLP_clipping_only \
      --width "$wid" \
      --lr "$lr" \
      --epochs 20 \
      --bs 500 \
      --mini_bs 500 \
      --epsilon 2 \
      --noise 0 \
      --clipping_mode BK-MixOpt \
      --clipping_style layer-wise \
      --cifar_data CIFAR10 \
      --dimension 32 \
      --optimizer SGD \
      --log_path "/content/drive/MyDrive/DP_muP/logs/MLP_SGD_diffwid_co.txt"
  done
done
