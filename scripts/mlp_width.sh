#!/bin/bash
PYTHON=python3.10

PROJECT_ROOT=/content/fast-differential-privacy
export PYTHONPATH="$PROJECT_ROOT"

# LRS=(-6.5 -6.25 -5.75 -5.5 -5.25) # SGD
# # 288 512 1152 2048 3200 4608
# for lr in "${LRS[@]}"; do
#   for wid in 512 1152 2048 3200 4608; do 
#     sig=$(awk "BEGIN {print 2.0*sqrt(128.0/$wid)}")
#     dim=$(awk "BEGIN {print sqrt($wid/128.0)*8.0}")
#     echo "Running width=$wid, lr=$lr, noise=$sig, dim=$dim"
#     $PYTHON -m scripts.MLP_approx \
#       --width "$wid" \
#       --lr "$lr" \
#       --epochs 10 \
#       --bs 500 \
#       --mini_bs 500 \
#       --epsilon 2 \
#       --noise "$sig" \
#       --clipping_mode BK-MixOpt \
#       --clipping_style layer-wise \
#       --cifar_data CIFAR10 \
#       --dimension "$dim" \
#       --optimizer SGD \
#       --log_path "/content/drive/MyDrive/DP_muP/logs/MLP_SGD_depth5_diffwidth_approx_ratio_1.txt"
#   done
# done


LRS=(-10) # SGD
# 256 512 1024 2048 4096
for wid in 1024; do 
  for lr in "${LRS[@]}"; do
    echo "Running width=$wid, lr=$lr, noise=2, dim=32"
    $PYTHON -m scripts.MLP_muon \
      --width "$wid" \
      --lr "$lr" \
      --epochs 10 \
      --bs 500 \
      --mini_bs 500 \
      --epsilon 2 \
      --noise 2 \
      --clipping_mode BK-MixOpt \
      --clipping_style layer-wise \
      --cifar_data CIFAR10 \
      --dimension 32 \
      --optimizer SGD \
      --log_path "/content/drive/MyDrive/DP_muP/logs/MLP_Muon_depth5_diffwidth_approx_ratio_new.txt"
  done
done

# LRS=(-15 -10) # SGD
# # 288 512 1152 2048 3200 4608
# for wid in 512 1152 2048 3200 4608; do
#   for lr in "${LRS[@]}"; do
#     sig=$(awk "BEGIN {print 4.0*sqrt(128.0/$wid)}")
#     dim=$(awk "BEGIN {print sqrt($wid/128.0)*8.0}")
#     echo "Running width=$wid, lr=$lr, noise=$sig, dim=$dim"
#     $PYTHON -m scripts.MLP_unifed \
#       --width "$wid" \
#       --lr "$lr" \
#       --epochs 10 \
#       --bs 500 \
#       --mini_bs 500 \
#       --epsilon 2 \
#       --noise "$sig" \
#       --clipping_mode BK-MixOpt \
#       --clipping_style layer-wise \
#       --cifar_data CIFAR10 \
#       --dimension "$dim" \
#       --optimizer Adam \
#       --log_path "/content/drive/MyDrive/DP_muP/logs/MLP_Adam_diffwidth_fnorm_ratio.txt"
#   done
# done
