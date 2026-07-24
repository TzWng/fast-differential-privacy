#!/bin/bash
PYTHON=python3.10

PROJECT_ROOT=/content/fast-differential-privacy
export PYTHONPATH="$PROJECT_ROOT"


# LRS=(-2.5 -2 -1.5 -1 -0.5)
# SEEDS=(1 3 4 5)
# for seed in "${SEEDS[@]}"; do
#   for s in 1 2 3 4 5; do
#     for lr in "${LRS[@]}"; do
#       echo "Running seed=$seed, scale=$s, lr=$lr"
#       $PYTHON -m scripts.vit_unifed \
#         --lr "$lr" \
#         --epochs 3 \
#         --bs 500 \
#         --mini_bs 500 \
#         --epsilon 2 \
#         --noise 2 \
#         --seed "$seed" \
#         --scale "$s" \
#         --clipping_mode BK-MixOpt \
#         --clipping_style layer-wise \
#         --dataset CIFAR10 \
#         --dimension 224 \
#         --optimizer SGD \
#         --log_path "/content/drive/MyDrive/DP_muP/logs_rebuttal/Vit_cifar10_SGD_dpmup_seed${seed}.txt"
#     done
#   done
# done



# LRS=(-9 -8.5 -8 -7.5 -7)
# SEEDS=(1 3 4 5)
# for seed in "${SEEDS[@]}"; do
#   for s in 1 2 3 4 5; do
#     for lr in "${LRS[@]}"; do
#       echo "Running seed=$seed, scale=$s, lr=$lr"
#       $PYTHON -m scripts.vit_unifed \
#         --lr "$lr" \
#         --epochs 3 \
#         --bs 500 \
#         --mini_bs 500 \
#         --epsilon 2 \
#         --noise 2 \
#         --seed "$seed" \
#         --scale "$s" \
#         --clipping_mode BK-MixOpt \
#         --clipping_style layer-wise \
#         --dataset CIFAR10 \
#         --dimension 224 \
#         --optimizer Adam \
#         --log_path "/content/drive/MyDrive/DP_muP/logs_rebuttal/Vit_cifar10_Adam_dpmup_seed${seed}.txt"
#     done
#   done
# done


LRS=(-5.75 -5.5 -5.25 -5)
SEEDS=(1 2 3 4 5)
for seed in "${SEEDS[@]}"; do
  for s in 1 2 3 4 5; do
    for lr in "${LRS[@]}"; do
      echo "Running seed=$seed, scale=$s, lr=$lr"
      $PYTHON -m scripts.vit_muon_sgd \
        --lr "$lr" \
        --epochs 3 \
        --bs 500 \
        --mini_bs 500 \
        --epsilon 2 \
        --noise 2 \
        --seed "$seed" \
        --scale "$s" \
        --clipping_mode BK-MixOpt \
        --clipping_style layer-wise \
        --dataset CIFAR10 \
        --dimension 224 \
        --optimizer muon \
        --log_path "/content/drive/MyDrive/DP_muP/logs_rebuttal/Vit_cifar10_muon_seed${seed}.txt"
    done
  done
done
