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



LRS=(0 -0.5)

for s in 5; do
  for lr in "${LRS[@]}"; do
    echo "Running scale=$s, lr=$lr"
    $PYTHON -m scripts.vit_unifed \
      --lr "$lr" \
      --epochs 3\
      --bs 500 \
      --mini_bs 500 \
      --epsilon 2 \
      --noise 2 \
      --scale "$s" \
      --clipping_mode BK-MixOpt \
      --clipping_style layer-wise \
      --cifar_data CIFAR10 \
      --dimension 224 \
      --optimizer SGD \
      --log_path "/content/drive/MyDrive/DP_muP/logs/Vit_SGD_compare_2.txt"
  done
done
