#!/bin/bash
PYTHON=python3.10

PROJECT_ROOT=/content/fast-differential-privacy
export PYTHONPATH="$PROJECT_ROOT"


LRS=(-3.5 -3 -2.5 -2 -1.5 -1 -0.5 0 0.5)
for s in 1 2 3 4 5; do
  for lr in "${LRS[@]}"; do
    echo "Running scale=$s, lr=$lr"
    $PYTHON -m scripts.vit_sp\
      --lr "$lr" \
      --epochs 3\
      --bs 500 \
      --mini_bs 500 \
      --epsilon 2 \
      --noise 2 \
      --scale "$s" \
      --clipping_mode BK-MixOpt \
      --clipping_style layer-wise \
      --dataset CIFAR10 \
      --dimension 224 \
      --optimizer SGD \
      --log_path "/content/drive/MyDrive/DP_muP/logs/Vit_cifar10_SGD_compare_sp.txt"
  done
done

LRS=(-3.5 -3 -2.5 -2 -1.5 -1 -0.5 0 0.5)
for s in 1 2 3 4 5; do
  for lr in "${LRS[@]}"; do
    echo "Running scale=$s, lr=$lr"
    $PYTHON -m scripts.vit_sp \
      --lr "$lr" \
      --epochs 5\
      --bs 500 \
      --mini_bs 500 \
      --epsilon 2 \
      --noise 2 \
      --scale "$s" \
      --clipping_mode BK-MixOpt \
      --clipping_style layer-wise \
      --dataset CIFAR100 \
      --dimension 224 \
      --optimizer SGD \
      --log_path "/content/drive/MyDrive/DP_muP/logs/Vit_cifar100_SGD_compare_sp.txt"
  done
done

# MODELS=(
#   vit_tiny_patch16_224
#   vit_small_patch16_224
#   vit_base_patch16_224
# )


# LRS=(-4 -3 -2 -1)
# for model in "${MODELS[@]}"; do
#   for lr in "${LRS[@]}"; do
#     echo "Running model=$model, lr=$lr"
#     $PYTHON -m scripts.vit_unifed \
#       --model "$model" \
#       --lr "$lr" \
#       --epochs 3\
#       --bs 500 \
#       --mini_bs 500 \
#       --epsilon 2 \
#       --noise 2 \
#       --clipping_mode BK-MixOpt \
#       --clipping_style layer-wise \
#       --dataset CIFAR100 \
#       --dimension 224 \
#       --optimizer SGD \
#       --log_path "/content/drive/MyDrive/DP_muP/logs/Vit_ft_cifar100_SGD_dpmup.txt"
#   done
# done

