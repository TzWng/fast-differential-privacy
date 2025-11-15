#!/bin/bash

PYTHON=python3.10
BS=1024  # 你要的 batch size

# LRS=(-7 -6 -5 -4 -3)
# for BS in 250 500 1000 2000; do
#   epoch=$(( 4 * BS / 125 ))
#   for lr in "${LRS[@]}"; do
#     sig=$(awk "BEGIN {print 4*$BS/125.0}")
#     echo "Running BS=$BS, lr=$lr, noise=$sig" 
#     # scaled_lr=$(echo "$lr + $ratio" | bc -l)
#     $PYTHON -m scripts.MLP_unifed \
#       --width 256 \
#       --lr "$lr" \
#       --epochs "$epoch"\
#       --bs "$BS" \
#       --mini_bs "$BS" \
#       --epsilon 2 \
#       --noise "$sig" \
#       --clipping_mode BK-MixOpt \
#       --clipping_style layer-wise \
#       --cifar_data CIFAR10 \
#       --dimension 32 \
#       --optimizer SGD \
#       --log_path "/content/drive/MyDrive/DP_muP/logs/MLP_SGD_diffbs_fnorm_ratio.txt"
#   done
# done

LRS=(-7)

for BS in 125; do
  epoch=$(( 4 * BS / 125 ))
  # ratio=$(echo "scale=6; x = $BS/125; l(x)/l(2)" | bc -l)
  for lr in "${LRS[@]}"; do
    echo "Running BS=$BS, lr=$lr" 
    # scaled_lr=$(echo "$lr + $ratio" | bc -l)
    $PYTHON -m scripts.MLP_unifed \
      --width 1024 \
      --lr "$lr" \
      --epochs "$epoch"\
      --bs "$BS" \
      --mini_bs "$BS" \
      --epsilon 2 \
      --noise 4 \
      --clipping_mode BK-MixOpt \
      --clipping_style layer-wise \
      --cifar_data CIFAR10 \
      --dimension 32 \
      --optimizer Adam \
      --log_path "/content/drive/MyDrive/DP_muP/logs/MLP_Adam_diffbs_fnorm_ratio.txt"
  done
done


LRS=(-11 -10 -9 -8 -7)

for BS in 250 500 1000 2000; do
  epoch=$(( 4 * BS / 125 ))
  # ratio=$(echo "scale=6; x = $BS/125; l(x)/l(2)" | bc -l)
  for lr in "${LRS[@]}"; do
    echo "Running BS=$BS, lr=$lr" 
    # scaled_lr=$(echo "$lr + $ratio" | bc -l)
    $PYTHON -m scripts.MLP_unifed \
      --width 1024 \
      --lr "$lr" \
      --epochs "$epoch"\
      --bs "$BS" \
      --mini_bs "$BS" \
      --epsilon 2 \
      --noise 4 \
      --clipping_mode BK-MixOpt \
      --clipping_style layer-wise \
      --cifar_data CIFAR10 \
      --dimension 32 \
      --optimizer Adam \
      --log_path "/content/drive/MyDrive/DP_muP/logs/MLP_Adam_diffbs_fnorm_ratio.txt"
  done
done

