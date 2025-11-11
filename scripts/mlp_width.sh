#!/bin/bash
PYTHON=python3.10

PROJECT_ROOT=/content/fast-differential-privacy
export PYTHONPATH="$PROJECT_ROOT"

LRS=(-13 -12.5 -12 -11.5 -11)
# LRS=(-13)
LRS=(-16.5 -16 -15.5)

LRS=(-6 -5 -4 -3)
for wid in 256; do
  for lr in "${LRS[@]}"; do
    $PYTHON -m scripts.MLP_unifed \
      --width "$wid" \
      --lr "$lr" \
      --epochs 20\
      --bs 500 \
      --mini_bs 500 \
      --epsilon 2 \
      --noise 0 \
      --clipping_mode BK-MixOpt \
      --clipping_style layer-wise \
      --cifar_data CIFAR10 \
      --dimension 32 \
      --optimizer SGD \
      --log_path "/content/drive/MyDrive/DP_muP/logs/temp.txt"
  done
done
