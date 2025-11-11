#!/bin/bash
PYTHON=python3.10

PROJECT_ROOT=/content/fast-differential-privacy
export PYTHONPATH="$PROJECT_ROOT"


LRS=(-16.5 -17) # Adam

for wid in 512 1024; do
  for lr in "${LRS[@]}"; do
    sig=$(awk "BEGIN {print 256.0/$wid}")
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
      --optimizer Adam \
      --log_path "/content/drive/MyDrive/DP_muP/logs/MLP_Adam_diffwidth_truenorm_ratio.txt"
  done
done

LRS=(-17 -16.5 -16 -15.5 -15) # Adam

for wid in 2048 4096; do
  for lr in "${LRS[@]}"; do
    sig=$(awk "BEGIN {print 256.0/$wid}")
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
      --optimizer Adam \
      --log_path "/content/drive/MyDrive/DP_muP/logs/MLP_Adam_diffwidth_truenorm_ratio.txt"
  done
done
