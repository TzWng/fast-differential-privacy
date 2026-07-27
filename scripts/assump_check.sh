#!/bin/bash
PYTHON=python3.10

PROJECT_ROOT=/content/fast-differential-privacy
export PYTHONPATH="$PROJECT_ROOT"


# LRS=(0)
# for s in 5; do
#   for lr in "${LRS[@]}"; do
#     echo "Running scale=$s, lr=$lr"
#     $PYTHON -m scripts.vit_sp \
#       --lr "$lr" \
#       --epochs 3 \
#       --bs 250 \
#       --mini_bs 250 \
#       --epsilon 2 \
#       --noise 2 \
#       --scale "$s" \
#       --clipping_mode BK-MixOpt \
#       --clipping_style layer-wise \
#       --dataset CIFAR100 \
#       --dimension 224 \
#       --optimizer Adam \
#       --log_path "/content/drive/MyDrive/DP_muP/logs_rebuttal/temp.txt"
#   done
# done


LRS=(-6)
for s in 3; do
  for lr in "${LRS[@]}"; do
    echo "Running scale=$s, lr=$lr"
    $PYTHON -m scripts.vit_muon_sgd \
      --lr "$lr" \
      --epochs 2 \
      --bs 250 \
      --mini_bs 250 \
      --epsilon 2 \
      --noise 2 \
      --scale "$s" \
      --clipping_mode BK-MixOpt \
      --clipping_style layer-wise \
      --dataset CIFAR10 \
      --dimension 224 \
      --optimizer muon \
      --log_path "/content/drive/MyDrive/DP_muP/logs_rebuttal/temp.txt"
  done
done

