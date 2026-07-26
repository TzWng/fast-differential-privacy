#!/bin/bash
PYTHON=python3.10

PROJECT_ROOT=/content/fast-differential-privacy
export PYTHONPATH="$PROJECT_ROOT"


LRS=(-8)
for s in 5; do
  for lr in "${LRS[@]}"; do
    echo "Running seed=$seed, scale=$s, lr=$lr"
    $PYTHON -m scripts.vit_sp \
      --lr "$lr" \
      --epochs 3 \
      --bs 125 \
      --mini_bs 125 \
      --epsilon 2 \
      --noise 2 \
      --scale "$s" \
      --clipping_mode BK-MixOpt \
      --clipping_style layer-wise \
      --dataset CIFAR10 \
      --dimension 224 \
      --optimizer Adam \
      --log_path "/content/drive/MyDrive/DP_muP/logs_rebuttal/temp.txt"
  done
done
