#!/bin/bash
PYTHON=python3.10

PROJECT_ROOT=/content/fast-differential-privacy
export PYTHONPATH="$PROJECT_ROOT"


# LRS=(-16.5 -17) # Adam

# for wid in 512 1024; do
#   for lr in "${LRS[@]}"; do
#     sig=$(awk "BEGIN {print 256.0/$wid}")
#     $PYTHON -m scripts.MLP_unifed \
#       --width "$wid" \
#       --lr "$lr" \
#       --epochs 20\
#       --bs 500 \
#       --mini_bs 500 \
#       --epsilon 2 \
#       --noise "$sig" \
#       --clipping_mode BK-MixOpt \
#       --clipping_style layer-wise \
#       --cifar_data CIFAR10 \
#       --dimension 32 \
#       --optimizer Adam \
#       --log_path "/content/drive/MyDrive/DP_muP/logs/MLP_Adam_diffwidth_truenorm_ratio.txt"
#   done
# done

#288 512 1152 2048 3200 4608
LRS=(-16 -15.5 -15 -14.5)
for wid in 1152 2048 3200 4608; do
  for lr in "${LRS[@]}"; do
    sig=$(awk "BEGIN {print sqrt(128.0/$wid)}*2.0")
    dim=$(awk "BEGIN {print sqrt($wid/128.0)*8.0}")
    echo "Running width=$wid, lr=$lr, noise=$sig, dim=$dim"
    $PYTHON -m scripts.MLP_unifed \
      --width "$wid" \
      --lr "$lr" \
      --epochs 20 \
      --bs 500 \
      --mini_bs 500 \
      --epsilon 2 \
      --noise "$sig" \
      --clipping_mode BK-MixOpt \
      --clipping_style layer-wise \
      --cifar_data CIFAR10 \
      --dimension "$dim" \
      --optimizer Adam \
      --log_path "/content/drive/MyDrive/DP_muP/logs/MLP_Adam_diffwidth_truenorm_ratio.txt"
  done
done
