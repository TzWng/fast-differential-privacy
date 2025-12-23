#!/bin/bash
PYTHON=python3.10

PROJECT_ROOT=/content/fast-differential-privacy
export PYTHONPATH="$PROJECT_ROOT"

# LRS=(-4)
# for s in 1 2 3 4 5; do
#   for lr in "${LRS[@]}"; do
#     echo "Running scale=$s, lr=$lr"
#     $PYTHON -m scripts.vit_unifed \
#       --lr "$lr" \
#       --epochs 3\
#       --bs 500 \
#       --mini_bs 500 \
#       --epsilon 2 \
#       --noise 2 \
#       --scale "$s" \
#       --clipping_mode BK-MixOpt \
#       --clipping_style layer-wise \
#       --dataset CIFAR10 \
#       --dimension 224 \
#       --optimizer SGD \
#       --log_path "/content/drive/MyDrive/DP_muP/logs/Vit_SGD_compare_1.txt"
#   done
# done


MODELS=(
  vit_small_patch16_224
  vit_base_patch16_224
)


LRS=(-4 -3 -2)
for model in "${MODELS[@]}"; do
  for lr in "${LRS[@]}"; do
    echo "Running model=$model, lr=$lr"
    $PYTHON -m scripts.vit_unifed \
      --model "$model" \
      --lr "$lr" \
      --epochs 3\
      --bs 500 \
      --mini_bs 500 \
      --epsilon 2 \
      --noise 2 \
      --clipping_mode BK-MixOpt \
      --clipping_style layer-wise \
      --dataset CIFAR10 \
      --dimension 224 \
      --optimizer SGD \
      --log_path "/content/drive/MyDrive/DP_muP/logs/Vit_ft_SGD_dpmup.txt"
  done
done

