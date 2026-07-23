#!/bin/bash
PYTHON=python3.10

PROJECT_ROOT=/content/fast-differential-privacy
export PYTHONPATH="$PROJECT_ROOT"

# LRS=(-3.75 -3.5 -3.25 -3 -2.75) # SGD
# SEEDS=(5 7)
# for seed in "${SEEDS[@]}"; do
#   for wid in 256 512 1024 2048 4096 8192; do
#     for lr in "${LRS[@]}"; do
#       echo "Running width=$wid, seed=$seed, lr=$lr, dim=32"
#       $PYTHON -m scripts.dpmup_sgd \
#         --width "$wid" \
#         --lr "$lr" \
#         --epochs 10 \
#         --bs 500 \
#         --mini_bs 500 \
#         --noise 7.067494947837502 \
#         --seed "$seed" \
#         --cifar_data CIFAR10 \
#         --clipping_mode BK-MixOpt \
#         --clipping_style layer-wise \
#         --dimension 32 \
#         --optimizer SGD \
#         --log_path "/content/drive/MyDrive/DP_muP/logs_rebuttal/MLP_SGD_depth5_s2l_epsilon2_dinfix_dpmup_seed${seed}.txt"
#     done
#   done
# done

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
#       --seed 5 \
#       --cifar_data CIFAR10 \
#       --clipping_mode BK-MixOpt \
#       --clipping_style layer-wise \
#       --dimension 32 \
#       --optimizer Adam \
#       --log_path "/content/drive/MyDrive/DP_muP/logs_rebuttal/MLP_Adam_depth5_s2l_epsilon2_dinfix_dpmup_seed5.txt"
#   done
# done

LRS=(-5 -4.5 -4 -3.5 -3 -2.5 -2) # SGD
# 256 512 1024 2048 4096 8192
for wid in 256 512 1024 2048 4096 8192; do
  for lr in "${LRS[@]}"; do 
    echo "Running width=$wid, lr=$lr, dim=32"
    $PYTHON -m scripts.MLP_sp \
      --width "$wid" \
      --lr "$lr" \
      --epochs 10 \
      --bs 500 \
      --mini_bs 500 \
      --noise 1.02294921875 \
      --cifar_data CIFAR10 \
      --clipping_mode BK-MixOpt \
      --clipping_style layer-wise \
      --dimension 32 \
      --optimizer SGD \
      --log_path "/content/drive/MyDrive/DP_muP/logs_rebuttal/MLP_SGD_depth5_s2l_epsilon2_dinfix_dpsp.txt"
  done
done



# LRS=(-15 -14.5 -14 -13.5 -13 -12.5 -12 -11.5 -11) # SGD
# # 256 512 1024 2048 4096 8192
# for wid in 2048 4096 8192; do
#   for lr in "${LRS[@]}"; do 
#     echo "Running width=$wid, lr=$lr, dim=32"
#     $PYTHON -m scripts.MLP_sp \
#       --width "$wid" \
#       --lr "$lr" \
#       --epochs 10 \
#       --bs 500 \
#       --mini_bs 500 \
#       --noise 1.02294921875 \
#       --cifar_data CIFAR10 \
#       --clipping_mode BK-MixOpt \
#       --clipping_style layer-wise \
#       --dimension 32 \
#       --optimizer Adam \
#       --log_path "/content/drive/MyDrive/DP_muP/logs/MLP_Adam_depth5_s2l_epsilon2_dinfix_dpsp.txt"
#   done
# done

