#!/bin/bash
PYTHON=python3.10

PROJECT_ROOT=/content/fast-differential-privacy
export PYTHONPATH="$PROJECT_ROOT"



# LRS=(-1 -0.5 0) # SGD

# for BS in 125; do
#   epoch=$(( 4 * BS / 125 ))
#   for lr in "${LRS[@]}"; do
#     echo "Running bs=$BS, epoch=$epoch, lr=$lr"
#     $PYTHON -m scripts.MLP_clipping_only \
#       --width 1024 \
#       --lr "$lr" \
#       --epochs "$epoch" \
#       --bs "$BS" \
#       --mini_bs "$BS" \
#       --epsilon 2 \
#       --noise 0 \
#       --clipping_mode BK-MixOpt \
#       --clipping_style layer-wise \
#       --cifar_data CIFAR10 \
#       --dimension 32 \
#       --optimizer SGD \
#       --log_path "/content/drive/MyDrive/DP_muP/logs/MLP_SGD_diffbs_co.txt"
#   done
# done

LRS=(-7 -7.5) # SGD

for BS in 2000; do
  epoch=$(( 4 * BS / 125 ))
  for lr in "${LRS[@]}"; do
    echo "Running bs=$BS, epoch=$epoch, lr=$lr"
    $PYTHON -m scripts.MLP_clipping_only \
      --width 512 \
      --lr "$lr" \
      --epochs "$epoch" \
      --bs "$BS" \
      --mini_bs "$BS" \
      --epsilon 2 \
      --noise 0 \
      --clipping_mode BK-MixOpt \
      --clipping_style layer-wise \
      --cifar_data CIFAR10 \
      --dimension 32 \
      --optimizer SGD \
      --log_path "/content/drive/MyDrive/DP_muP/logs/MLP_SGD_diffbs_truenorm.txt"
  done
done
