#!/bin/bash
PYTHON=python3.10

PROJECT_ROOT=/content/fast-differential-privacy
export PYTHONPATH="$PROJECT_ROOT"

LRS=(-10.5 -10 -9.5 -9)

for BS in 125 250 500 1000 2000; do
  epoch=$(( 4 * BS / 125 ))
  for lr in "${LRS[@]}"; do
    sig=$(awk "BEGIN {print 2*$BS/125.0}")
    echo "Running BS=$BS, lr=$lr, epoch=$epoch, noise=$sig" 
    $PYTHON -m scripts.MLP_unifed \
      --width 256 \
      --lr "$lr" \
      --epochs "$epoch" \
      --bs "$BS" \
      --mini_bs "$BS" \
      --epsilon 2 \
      --noise "$sig" \
      --clipping_mode BK-MixOpt \
      --clipping_style layer-wise \
      --cifar_data CIFAR10 \
      --dimension 32 \
      --optimizer Adam \
      --log_path "/content/drive/MyDrive/DP_muP/logs/MLP_Adam_diffbs_truenorm.txt"
  done
done
