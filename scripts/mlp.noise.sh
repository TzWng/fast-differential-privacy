#!/bin/bash

PYTHON=python3.10
BS=1024  # 你要的 batch size

128 
LRS=(-13)
for sigma in 2 4 8 16; do
  epoch=$(( 4 * BS / 125 ))
  for lr in "${LRS[@]}"; do
    dim=$(awk "BEGIN {print sqrt($wid/128.0)*8.0}")
    wid=$(awk "BEGIN {print sqrt($wid/128.0)*8.0}")
    echo "Running BS=$BS, lr=$lr, noise=$sig" 
    # scaled_lr=$(echo "$lr + $ratio" | bc -l)
    $PYTHON -m scripts.MLP_unifed \
      --width 256 \
      --lr "$lr" \
      --epochs "$epoch"\
      --bs 500 \
      --mini_bs 500 \
      --epsilon 2 \
      --noise "$sig" \
      --clipping_mode BK-MixOpt \
      --clipping_style layer-wise \
      --cifar_data CIFAR10 \
      --dimension 32 \
      --optimizer Adam \
      --log_path "/content/drive/MyDrive/DP_muP/logs/MLP_Adam_diffbs_fnorm_ratio.txt"
  done
done
