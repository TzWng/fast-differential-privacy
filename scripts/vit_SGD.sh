#!/bin/bash
PYTHON=python3.10

PROJECT_ROOT=/content/fast-differential-privacy
export PYTHONPATH="$PROJECT_ROOT"


MODELS=(
  vit_tiny_patch16_224
  vit_small_patch16_224
  vit_base_patch16_224
  vit_large_patch16_224
)


LRS=(-3 -2.5 -2 -1.5 -1 -0.5)
# LRS=(-12 -11.5 -11 -10.5 -10)
# LRS=(-5.5 -5 -4.5 -4 -3.5 -3 -2.5)

for s in 2 4 5; do
  for lr in "${LRS[@]}"; do
    $PYTHON -m scripts.vit_unifed \
      --lr "$lr" \
      --epochs 3\
      --bs 200 \
      --mini_bs 200 \
      --epsilon 2 \
      --noise 1 \
      --scale "$s" \
      --clipping_mode BK-MixOpt \
      --clipping_style layer-wise \
      --cifar_data CIFAR10 \
      --dimension 224 \
      --optimizer SGD \
      --log_path "/content/drive/MyDrive/DP_muP/logs/Vit_SGD_diffwidth_DP.txt"
  done
done
