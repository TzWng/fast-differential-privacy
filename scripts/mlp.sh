#!/bin/bash

PYTHON=python3.10
BS=1024  # 你要的 batch size

LRS=(-8 -7.5 -7 -6.5 -6 -5.5 -5 -4.5 -4 -3.5)
# LRS = (-5 -4.5 -4 -3.5 -3 -2.5 -2 -1.5 -1 -0.5 0 0.5 1 1.5 2)
# LRS = (-3 -2.5 -2 -1.5 -1 -0.5 0 0.5 1 1.5 2)

PROJECT_ROOT=/content/fast-differential-privacy
export PYTHONPATH="$PROJECT_ROOT"

for wid in 8192; do
  for lr in "${LRS[@]}"; do
    $PYTHON -m scripts.MLP_unifed \
      --width "$wid" \
      --lr "$lr" \
      --epochs 20 \
      --bs "$BS" \
      --mini_bs "$BS" \
      --epsilon 2 \
      --noise 0 \
      --clipping_mode BK-MixOpt \
      --clipping_style layer-wise \
      --cifar_data CIFAR10 \
      --dimension 32 \
      --optimizer Adam \
      --log_path "/content/drive/MyDrive/DP_muP/logs/MLP_Adam_diffbs_clipping_only.txt"
  done
done
