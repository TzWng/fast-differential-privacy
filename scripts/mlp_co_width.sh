#!/bin/bash
PYTHON=python3.10

PROJECT_ROOT=/content/fast-differential-privacy
export PYTHONPATH="$PROJECT_ROOT"


LRS=(-2.5 -3 -3.5 -4) # Adam

for wid in 256; do
  for lr in "${LRS[@]}"; do
    echo "Running width=$wid, lr=$lr"
    $PYTHON -m scripts.MLP_clipping_only \
      --width "$wid" \
      --lr "$lr" \
      --epochs 20 \
      --bs 500 \
      --mini_bs 500 \
      --epsilon 2 \
      --noise 0 \
      --clipping_mode BK-MixOpt \
      --clipping_style layer-wise \
      --cifar_data CIFAR10 \
      --dimension 32 \
      --optimizer Adam \
      --log_path "/content/drive/MyDrive/DP_muP/logs/MLP_Adam_diffwid_approx.txt"
  done
done

# LRS=(-3 -2 -1.5 -1) # SGD

# for wid in 256; do
#   for lr in "${LRS[@]}"; do
#     echo "Running width=$wid, lr=$lr"
#     $PYTHON -m scripts.MLP_clipping_only \
#       --width "$wid" \
#       --lr "$lr" \
#       --epochs 20 \
#       --bs 500 \
#       --mini_bs 500 \
#       --epsilon 2 \
#       --noise 0 \
#       --clipping_mode BK-MixOpt \
#       --clipping_style layer-wise \
#       --cifar_data CIFAR10 \
#       --dimension 32 \
#       --optimizer SGD \
#       --log_path "/content/drive/MyDrive/DP_muP/logs/MLP_SGD_diffwid_approx.txt"
#   done
# done



# LRS=(-6 -5.5 -5) # SGD

# for wid in 1024; do
#   for lr in "${LRS[@]}"; do
#     echo "Running width=$wid, lr=$lr"
#     $PYTHON -m scripts.MLP_clipping_only \
#       --width "$wid" \
#       --lr "$lr" \
#       --epochs 20 \
#       --bs 500 \
#       --mini_bs 500 \
#       --epsilon 2 \
#       --noise 0 \
#       --clipping_mode BK-MixOpt \
#       --clipping_style layer-wise \
#       --cifar_data CIFAR10 \
#       --dimension 32 \
#       --optimizer SGD \
#       --log_path "/content/drive/MyDrive/DP_muP/logs/MLP_SGD_diffwid_truenorm.txt"
#   done
# done
