#!/bin/bash
PYTHON=python3.10

PROJECT_ROOT=/content/fast-differential-privacy
export PYTHONPATH="$PROJECT_ROOT"


# LRS=(-10.5 -10 -9.5 -9 -8.5 -8 -7.5)
# LRS=(-7.5 -8 -8.5 -9 -9.5 -10 -10.5)
# for s in 5; do
#   for lr in "${LRS[@]}"; do
#     echo "Running scale=$s, lr=$lr" 
#     $PYTHON -m scripts.vit_sp\
#       --lr "$lr" \
#       --epochs 5\
#       --bs 500 \
#       --mini_bs 500 \
#       --epsilon 2 \
#       --noise 2 \
#       --scale "$s" \
#       --clipping_mode BK-MixOpt \
#       --clipping_style layer-wise \
#       --dataset CIFAR100 \
#       --dimension 224 \
#       --optimizer Adam \
#       --log_path "/content/drive/MyDrive/DP_muP/logs/Vit_cifar100_Adam_compare_sp.txt"
#   done
# done


LRS=(-10.5 -10 -9.5 -9 -8.5 -8 -7.5)
for s in 3 4 5; do
  for lr in "${LRS[@]}"; do
    echo "Running scale=$s, lr=$lr" 
    $PYTHON -m scripts.vit_sp\
      --lr "$lr" \
      --epochs 5\
      --bs 500 \
      --mini_bs 500 \
      --epsilon 2 \
      --noise 2 \
      --scale "$s" \
      --clipping_mode BK-MixOpt \
      --clipping_style layer-wise \
      --dataset CIFAR100 \
      --dimension 224 \
      --optimizer Adam \
      --log_path "/content/drive/MyDrive/DP_muP/logs/Vit_cifar100_Adam_compare_sp.txt"
  done
done



# LRS=(-12 -11.5 -11 -10.5 -10 -9.5 -9 -8.5)
# for s in 3 4 5; do
#   for lr in "${LRS[@]}"; do
#     echo "Running scale=$s, lr=$lr"
#     $PYTHON -m scripts.vit_sp\
#       --lr "$lr" \
#       --epochs 3\
#       --bs 500 \
#       --mini_bs 500 \
#       --epsilon 2 \
#       --noise 2 \
#       --scale "$s" \
#       --clipping_mode BK-MixOpt \
#       --clipping_style layer-wise \
#       --dataset CIFAR10 \
#       --dimension 224 \
#       --optimizer Adam \
#       --log_path "/content/drive/MyDrive/DP_muP/logs/Vit_cifar10_Adam_compare_sp.txt"
#   done
# done


# LRS=(-7.5)
# for s in 1; do
#   for lr in "${LRS[@]}"; do
#     echo "Running scale=$s, lr=$lr"
#     $PYTHON -m scripts.vit_unifed\
#       --lr "$lr" \
#       --epochs 3\
#       --bs 500 \
#       --mini_bs 500 \
#       --epsilon 2 \
#       --noise 2 \
#       --scale "$s" \
#       --clipping_mode BK-MixOpt \
#       --clipping_style layer-wise \
#       --dataset CIFAR10 \
#       --dimension 224 \
#       --optimizer Adam \
#       --log_path "/content/drive/MyDrive/DP_muP/logs/Vit_cifar10_adam.txt"
#   done
# done

# LRS=(-7.5 -7)
# for s in 1 2 3 4 5; do
#   for lr in "${LRS[@]}"; do
#     echo "Running scale=$s, lr=$lr"
#     $PYTHON -m scripts.vit_unifed\
#       --lr "$lr" \
#       --epochs 3\
#       --bs 500 \
#       --mini_bs 500 \
#       --epsilon 2 \
#       --noise 2 \
#       --scale "$s" \
#       --clipping_mode BK-MixOpt \
#       --clipping_style layer-wise \
#       --dataset CIFAR10 \
#       --dimension 224 \
#       --optimizer Adam \
#       --log_path "/content/drive/MyDrive/DP_muP/logs/Vit_cifar10_adam.txt"
#   done
# done

# LRS=(-11 -10.5 -10 -9.5 -9 -8.5 -8 -7.5)
# for s in 1 2 3 4 5; do
#   for lr in "${LRS[@]}"; do
#     echo "Running scale=$s, lr=$lr"
#     $PYTHON -m scripts.vit_unifed\
#       --lr "$lr" \
#       --epochs 5\
#       --bs 500 \
#       --mini_bs 500 \
#       --epsilon 2 \
#       --noise 2 \
#       --scale "$s" \
#       --clipping_mode BK-MixOpt \
#       --clipping_style layer-wise \
#       --dataset CIFAR100 \
#       --dimension 224 \
#       --optimizer Adam \
#       --log_path "/content/drive/MyDrive/DP_muP/logs/Vit_cifar100_adam.txt"
#   done
# done

# LRS=(-5.25 -7.5)
# for s in 5; do
#   for lr in "${LRS[@]}"; do
#     echo "Running scale=$s, lr=$lr"
#     $PYTHON -m scripts.vit_muon_sgd\
#       --lr "$lr" \
#       --epochs 3\
#       --bs 500 \
#       --mini_bs 500 \
#       --epsilon 2 \
#       --noise 2 \
#       --scale "$s" \
#       --clipping_mode BK-MixOpt \
#       --clipping_style layer-wise \
#       --dataset CIFAR10 \
#       --dimension 224 \
#       --optimizer muon \
#       --log_path "/content/drive/MyDrive/DP_muP/logs/Vit_cifar10_muon.txt"
#   done
# done




# LRS=(-6 -5 -7 -8)
# for s in 2 3 4 5; do
#   for lr in "${LRS[@]}"; do
#     echo "Running scale=$s, lr=$lr"
#     $PYTHON -m scripts.vit_muon_sgd\
#       --lr "$lr" \
#       --epochs 5\
#       --bs 500 \
#       --mini_bs 500 \
#       --epsilon 2 \
#       --noise 2 \
#       --scale "$s" \
#       --clipping_mode BK-MixOpt \
#       --clipping_style layer-wise \
#       --dataset CIFAR100 \
#       --dimension 224 \
#       --optimizer muon \
#       --log_path "/content/drive/MyDrive/DP_muP/logs/Vit_cifar100_muon.txt"
#   done
# done


# MODELS=(
#   vit_tiny_patch16_224
#   vit_small_patch16_224
#   vit_base_patch16_224
# )



# LRS=(-6 -5 -4 -3)
# for s in 1; do
#   for lr in "${LRS[@]}"; do
#     echo "Running scale=$s, lr=$lr"
#     $PYTHON -m scripts.vit_unifed \
#       --lr "$lr" \
#       --epochs 3\
#       --bs 500 \
#       --mini_bs 500 \
#       --epsilon 2 \
#       --noise 0.9036090970039368 \
#       --scale "$s" \
#       --clipping_mode BK-MixOpt \
#       --clipping_style layer-wise \
#       --dataset CIFAR10 \
#       --dimension 224 \
#       --optimizer SGD \
#       --log_path "/content/drive/MyDrive/DP_muP/logs/Vit_cifar10_SGD_dpmup_new.txt"
#   done
# done



# LRS=(1 0.5 0 -2 -2.5)
# for s in 3; do
#   for lr in "${LRS[@]}"; do
#     echo "Running scale=$s, lr=$lr"
#     $PYTHON -m scripts.vit_unifed \
#       --lr "$lr" \
#       --epochs 5\
#       --bs 500 \
#       --mini_bs 500 \
#       --epsilon 2 \
#       --noise 0.9036090970039368 \
#       --scale "$s" \
#       --clipping_mode BK-MixOpt \
#       --clipping_style layer-wise \
#       --dataset CIFAR100 \
#       --dimension 224 \
#       --optimizer SGD \
#       --log_path "/content/drive/MyDrive/DP_muP/logs/Vit_cifar100_SGD_dpmup_new_1.txt"
#   done
# done




