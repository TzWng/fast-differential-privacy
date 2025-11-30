#!/bin/bash

PYTHON=python3.10
BS=1024  # 你要的 batch size


LRS=(-7.5 -6.5 -5.5)
# 128 512 2048 8192
for wid in 512; do
  for lr in "${LRS[@]}"; do
    dim=$(awk "BEGIN {print sqrt($wid/128.0)*8.0}")
    bs=$(awk "BEGIN {print sqrt($wid/128.0)*125.0}")
    epoch=$(( 4 * $bs / 125 ))
    echo "Running width=$wid, dim=$dim, BS=$bs, lr=$lr" 
    # scaled_lr=$(echo "$lr + $ratio" | bc -l)
    $PYTHON -m scripts.MLP_approx \
      --width "$wid" \
      --lr "$lr" \
      --epochs "$epoch"\
      --bs "$bs" \
      --mini_bs "$bs" \
      --epsilon 2 \
      --noise 1 \
      --clipping_mode BK-MixOpt \
      --clipping_style layer-wise \
      --cifar_data CIFAR10 \
      --dimension 32 \
      --optimizer SGD \
      --log_path "/content/drive/MyDrive/DP_muP/logs/MLP_SGD_diffwidth_change_bs.txt"
  done
done
