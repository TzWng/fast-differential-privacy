#!/bin/bash
PYTHON=python3.10

PROJECT_ROOT=/content/fast-differential-privacy
export PYTHONPATH="$PROJECT_ROOT"

LRS=(-3 -2.5 -2 -1.5 -1 -0.5 0)

MODELS=(
  vit_tiny_patch16_224
  vit_small_patch16_224
  vit_base_patch16_224
  vit_large_patch16_224
)

for model in "${MODELS[@]}"; do
  for lr in "${LRS[@]}"; do
    $PYTHON -m scripts.vit_unifed \
      --model "$model" \
      --lr "$lr" \
      --epochs 3\
      --bs 200 \
      --mini_bs 200 \
      --epsilon 2 \
      --noise 1 \
      --clipping_mode BK-MixOpt \
      --clipping_style layer-wise \
      --cifar_data CIFAR10 \
      --dimension 224 \
      --optimizer SGD \
      --log_path "/content/drive/MyDrive/DP_muP/logs/Vit_SGD_diffwidth_sig10.txt"
  done
done

