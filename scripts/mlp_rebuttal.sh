#!/bin/bash
PYTHON=python3.10

PROJECT_ROOT=/content/fast-differential-privacy
export PYTHONPATH="$PROJECT_ROOT"

LRS=(-4 -3.75 -3.5 -3.25 -3 -2.75) # SGD
SEEDS=(5 7)
for seed in "${SEEDS[@]}"; do
  for wid in 256 512 1024 2048 4096 8192; do
    for lr in "${LRS[@]}"; do
      echo "Running width=256, seed=$seed, lr=$lr, dim=32"
      $PYTHON -m scripts.dpmup_sgd \
        --width 256 \
        --lr "$lr" \
        --epochs 10 \
        --bs 500 \
        --mini_bs 500 \
        --noise 7.067494947837502 \
        --seed "$seed" \
        --cifar_data CIFAR10 \
        --clipping_mode BK-MixOpt \
        --clipping_style layer-wise \
        --dimension 32 \
        --optimizer SGD \
        --log_path "/content/drive/MyDrive/DP_muP/logs_rebuttal/MLP_SGD_depth5_s2l_epsilon2_dinfix_dpmup_seed${seed}.txt"
    done
  done
done

# LRS=(-4 -3.75 -3.5 -3.25 -3 -2.75 -2.5) # SGD
# # 256 512 1024 2048 4096 8192
# for wid in 256 512 1024 2048 4096 8192; do
#   for lr in "${LRS[@]}"; do 
#     echo "Running width=$wid, lr=$lr, dim=32"
#     $PYTHON -m scripts.dpmup_sgd \
#       --width "$wid" \
#       --lr "$lr" \
#       --epochs 10 \
#       --bs 500 \
#       --mini_bs 500 \
#       --noise 7.067494947837502 \
#       --seed 3 \
#       --cifar_data CIFAR10 \
#       --clipping_mode BK-MixOpt \
#       --clipping_style layer-wise \
#       --dimension 32 \
#       --optimizer SGD \
#       --log_path "/content/drive/MyDrive/DP_muP/logs_rebuttal/MLP_SGD_depth5_s2l_epsilon2_dinfix_dpmup_seed3.txt"
#   done
# done

# LRS=(-10.25 -10 -9.75 -9.5 -9.25 -9) # SGD
# # 256 512 1024 2048 4096 8192
# for wid in 256 512 1024 2048 4096 8192; do
#   for lr in "${LRS[@]}"; do 
#     echo "Running width=$wid, lr=$lr, dim=32"
#     $PYTHON -m scripts.dpmup_sgd \
#       --width "$wid" \
#       --lr "$lr" \
#       --epochs 10 \
#       --bs 500 \
#       --mini_bs 500 \
#       --noise 7.067494947837502 \
#       --seed 2 \
#       --cifar_data CIFAR10 \
#       --clipping_mode BK-MixOpt \
#       --clipping_style layer-wise \
#       --dimension 32 \
#       --optimizer Adam \
#       --log_path "/content/drive/MyDrive/DP_muP/logs_rebuttal/MLP_Adam_depth5_s2l_epsilon2_dinfix_dpmup_seed2.txt"
#   done
# done
