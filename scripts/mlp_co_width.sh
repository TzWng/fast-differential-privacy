#!/bin/bash
PYTHON=python3.10

PROJECT_ROOT=/content/fast-differential-privacy
export PYTHONPATH="$PROJECT_ROOT"


# LRS=(-1.5 -2 -2.5 -3) # SGD

# for BS in 125 250 500 1000; do
#   epoch=$(( 4 * BS / 125 ))
#   for lr in "${LRS[@]}"; do
#     echo "Running bs=$BS, epoch=$epoch, lr=$lr"
#     $PYTHON -m scripts.MLP_clipping_only \
#       --width 512 \
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
#       --log_path "/content/drive/MyDrive/DP_muP/logs/MLP_SGD_diffbs_approx_10.txt"
#   done
# done



# LRS=(-2.5 -2 -1.5 -1 -0.5) # SGD
LRS=(-5.5 -5) # Adam

for wid in 512 1024 2048; do
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


LRS=(-9 -8.5 -8 -7.5 -7 -6.5 -6 -5.5 -5) # Adam

for wid in 4096; do
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

# LRS=(-1 -1.5 -2 -2.5 -3) # SGD

# for wid in 4096; do
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

# LRS=(-6 -5.5) # SGD

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
#       --log_path "/content/drive/MyDrive/DP_muP/logs/MLP_SGD_diffwid_truenorm.txt"
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
