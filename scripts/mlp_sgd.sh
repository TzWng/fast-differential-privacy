#!/bin/bash
PYTHON=python3.10

PROJECT_ROOT=/content/fast-differential-privacy
export PYTHONPATH="$PROJECT_ROOT"

LRS=(-8 -7.5 -7 -6.5 -6) # SGD

for wid in 512 1024 2048 4096; do
  for lr in "${LRS[@]}"; do
    echo "Running width=$wid, lr=$lr"
    $PYTHON -m scripts.MLP_unifed \
      --width "$wid" \
      --lr "$lr" \
      --epochs 20\
      --bs 500 \
      --mini_bs 500 \
      --epsilon 2 \
      --noise 1 \
      --clipping_mode BK-MixOpt \
      --clipping_style layer-wise \
      --cifar_data CIFAR10 \
      --dimension 32 \
      --optimizer SGD \
      --log_path "/content/drive/MyDrive/DP_muP/logs/MLP_SGD_diffwidth_truenorm_lastlayer10_sig10.txt"
  done
done

# LRS=(-6 -5.5) # SGD
# for wid in 1024 2048 4096; do
#   for lr in "${LRS[@]}"; do
#     echo "Running width=$wid, lr=$lr"
#     $PYTHON -m scripts.MLP_unifed \
#       --width "$wid" \
#       --lr "$lr" \
#       --epochs 20\
#       --bs 500 \
#       --mini_bs 500 \
#       --epsilon 2 \
#       --noise 1 \
#       --clipping_mode BK-MixOpt \
#       --clipping_style layer-wise \
#       --cifar_data CIFAR10 \
#       --dimension 32 \
#       --optimizer SGD \
#       --log_path "/content/drive/MyDrive/DP_muP/logs/MLP_SGD_diffwidth_truenorm_sig10_1.txt"
#   done
# done

# LRS=(-6 -5.5 -5 -4.5) # SGD
# for wid in 2048 4096; do
#   for lr in "${LRS[@]}"; do
#     echo "Running width=$wid, lr=$lr"
#     $PYTHON -m scripts.MLP_unifed \
#       --width "$wid" \
#       --lr "$lr" \
#       --epochs 20\
#       --bs 500 \
#       --mini_bs 500 \
#       --epsilon 2 \
#       --noise 1 \
#       --clipping_mode BK-MixOpt \
#       --clipping_style layer-wise \
#       --cifar_data CIFAR10 \
#       --dimension 32 \
#       --optimizer SGD \
#       --log_path "/content/drive/MyDrive/DP_muP/logs/MLP_SGD_diffwidth_truenorm_sig10_1.txt"
#   done
# done
