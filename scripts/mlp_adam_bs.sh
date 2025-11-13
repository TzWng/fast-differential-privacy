#!/bin/bash
PYTHON=python3.10

PROJECT_ROOT=/content/fast-differential-privacy
export PYTHONPATH="$PROJECT_ROOT"

LRS=(-12 -11.5 -11 -10.5)
#  250 500 1000 2000 5000

for BS in 500 1000 2000 5000; do
  epoch=$(( 4 * BS / 125 ))
  for lr in "${LRS[@]}"; do
    sig=$(awk "BEGIN {print 4*$BS/125.0}")
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
      --log_path "/content/drive/MyDrive/DP_muP/logs/MLP_Adam_diffbs_truenorm_ratio.txt"
  done
done

