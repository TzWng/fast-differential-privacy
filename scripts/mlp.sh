#!/bin/bash

PYTHON=python3.10
BS=1024  # 你要的 batch size

# LRS=(-8 -7.5 -7 -6.5 -6 -5.5 -5 -4.5 -4 -3.5 -3)
# LRS=(-5 -4.5 -4 -3.5 -3 -2.5 -2 -1.5 -1 -0.5 0 0.5 1 1.5 2)
LRS=(-2 -1.5 -1 -0.5 0 0.5 1)
# LRS=(-5 -4.5 -4 -3.5 -3)
# LRS=(-4 -3.5 -3 -2.5 -2 -1.5 -1)
# LRS=(1.5 2)

PROJECT_ROOT=/content/fast-differential-privacy
export PYTHONPATH="$PROJECT_ROOT"


for BS in 125 250 500 1000 2000; do
  epoch=$(( 5 * BS / 125 ))
  for lr in "${LRS[@]}"; do
    $PYTHON -m scripts.MLP_unifed \
      --width 4096 \
      --lr "$lr" \
      --epochs "$epoch"\
      --bs "$BS" \
      --mini_bs "$BS" \
      --epsilon 2 \
      --noise 0.5 \
      --clipping_mode BK-MixOpt \
      --clipping_style layer-wise \
      --cifar_data CIFAR10 \
      --dimension 32 \
      --optimizer Adam \
      --log_path "/content/drive/MyDrive/DP_muP/logs/MLP_Adam_diffbs_sig05.txt"
  done
done
