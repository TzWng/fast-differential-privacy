#!/bin/bash

PYTHON=python3.10
BS=1024  # 你要的 batch size

LRS=(-6 -5.5 -5 -4.5 -4 -3.5 -3)
LRS=(-8 -7.5)


PROJECT_ROOT=/content/fast-differential-privacy
export PYTHONPATH="$PROJECT_ROOT"


for BS in 125; do
  epoch=$(( 5 * BS / 125 ))
  # ratio=$(echo "scale=6; x = $BS/125; l(x)/l(2)" | bc -l)
  for lr in "${LRS[@]}"; do
    # scaled_lr=$(echo "$lr + $ratio" | bc -l)
    $PYTHON -m scripts.MLP_unifed \
      --width 4096 \
      --lr "$lr" \
      --epochs "$epoch"\
      --bs "$BS" \
      --mini_bs "$BS" \
      --epsilon 2 \
      --noise 1 \
      --clipping_mode BK-MixOpt \
      --clipping_style layer-wise \
      --cifar_data CIFAR10 \
      --dimension 32 \
      --optimizer SGD \
      --log_path "/content/drive/MyDrive/DP_muP/logs/MLP_SGD_diffbs_sig10.txt"
  done
done

