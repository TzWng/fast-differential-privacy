#!/bin/bash

PYTHON=python
BS=1024  # 你要的 batch size

# 手写一组 log2lr，对应 2^{lr}
LRS=(-16, -15.5, -15, -14.5, -14)

for wid in 512 1024 2048 4096; do
  for lr in "${LRS[@]}"; do
    python scripts/MLP_unifed.py \
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
      --log_path "/content/drive/MyDrive/DP_muP/logs/MLP_SGD_diffbs_clipping_only.txt"
  done
done
