#!/bin/bash
PYTHON=python3.10

PROJECT_ROOT=/content/fast-differential-privacy
export PYTHONPATH="$PROJECT_ROOT"


LRS=(0) # SGD
# 256 512 1024 2048 4096 8192
for wid in 1024; do
  for lr in "${LRS[@]}"; do 
    echo "Running width=$wid, lr=$lr, dim=32"
    $PYTHON -m scripts.MLP_sp \
      --width "$wid" \
      --lr "$lr" \
      --epochs 10 \
      --bs 500 \
      --mini_bs 500 \
      --noise 2 \
      --cifar_data CIFAR10 \
      --clipping_mode BK-MixOpt \
      --clipping_style layer-wise \
      --dimension 32 \
      --optimizer Adam \
      --log_path "/content/drive/MyDrive/DP_muP/logs/temp.txt"
  done
done
