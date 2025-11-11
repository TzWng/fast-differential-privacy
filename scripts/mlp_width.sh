#!/bin/bash
PYTHON=python3.10

PROJECT_ROOT=/content/fast-differential-privacy
export PYTHONPATH="$PROJECT_ROOT"


LRS=(-8.5 -8 -7.5 -7 -6.5) # SGD

for wid in 512 800 1152 1568 2048; do
  for lr in "${LRS[@]}"; do
    sig=$(awk "BEGIN {print sqrt(128.0/$wid)}")
    dim=$(awk "BEGIN {print sqrt($wid/128.0)*8.0}")
    echo "Running width=$wid, lr=$lr, noise=$sig, dim=$dim"
    $PYTHON -m scripts.MLP_unifed \
      --width "$wid" \
      --lr "$lr" \
      --epochs 20 \
      --bs 500 \
      --mini_bs 500 \
      --epsilon 2 \
      --noise "$sig" \
      --clipping_mode BK-MixOpt \
      --clipping_style layer-wise \
      --cifar_data CIFAR10 \
      --dimension "$dim" \
      --optimizer SGD \
      --log_path "/content/drive/MyDrive/DP_muP/logs/MLP_SGD_diffwidth_truenorm_lastlayer10_ratio.txt"
  done
done
