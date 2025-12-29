#!/bin/bash

PYTHON=python3.10
BS=1024  # 你要的 batch size

# # batch size 500 epsilon=2
# LRS=(-3.75) # SGD
# # 256 512 1024 2048 4096 8192
# for BS in 125 250 500 1000 2000; do
#   for lr in "${LRS[@]}"; do 
#     epoch=$(( 4 * BS / 125 ))
#     sig=$(awk "BEGIN {print 3.586439615946674*$BS/500.0}")
#     echo "Running batch size=$BS, lr=$lr, noise=$sig, epoch=$epoch"
#     $PYTHON -m scripts.dpmup_sgd \
#       --width 2048 \
#       --lr "$lr" \
#       --epochs "$epoch" \
#       --bs "$BS" \
#       --mini_bs "$BS" \
#       --noise "$sig" \
#       --seed 3 \
#       --cifar_data CIFAR10 \
#       --clipping_mode BK-MixOpt \
#       --clipping_style layer-wise \
#       --dimension 32 \
#       --optimizer SGD \
#       --log_path "/content/drive/MyDrive/DP_muP/logs/MLP_SGD_depth5_diffbs_dinfix_dpmup.txt"
#   done
# done


# LRS=(-7.25) # SGD
# # 256 512 1024 2048 4096 8192
# for BS in 125 250 500 1000 2000; do
#   for lr in "${LRS[@]}"; do 
#     epoch=$(( 4 * BS / 125 ))
#     sig=$(awk "BEGIN {print 3.586439615946674*$BS/500.0}")
#     echo "Running batch size=$BS, lr=$lr, noise=$sig, epoch=$epoch"
#     $PYTHON -m scripts.dpmup_sgd \
#       --width 2048 \
#       --lr "$lr" \
#       --epochs "$epoch" \
#       --bs "$BS" \
#       --mini_bs "$BS" \
#       --noise "$sig" \
#       --seed 3 \
#       --cifar_data CIFAR10 \
#       --clipping_mode BK-MixOpt \
#       --clipping_style layer-wise \
#       --dimension 32 \
#       --optimizer muon \
#       --log_path "/content/drive/MyDrive/DP_muP/logs/MLP_muon_depth5_diffbs_dinfix_dpmup.txt"
#   done
# done


# LRS=(-3.75) # SGD
# # 256 512 1024 2048 4096 8192
# for BS in 200 400; do
#   for lr in "${LRS[@]}"; do 
#     epoch=$(( 2 * BS / 50 ))
#     sig=$(awk "BEGIN {print 1.4983855926238738*$BS/400.0}")
#     echo "Running batch size=$BS, lr=$lr, noise=$sig, epoch=$epoch"
#     $PYTHON -m scripts.vit_unifed \
#       --lr "$lr" \
#       --epochs "$epoch" \
#       --bs "$BS" \
#       --mini_bs "$BS" \
#       --noise "$sig" \
#       --scale 1 \
#       --dataset CIFAR10 \
#       --clipping_mode BK-MixOpt \
#       --clipping_style layer-wise \
#       --dimension 224 \
#       --optimizer SGD \
#       --log_path "/content/drive/MyDrive/DP_muP/logs/Vit_sgd_diffbs_dinfix_dpmup.txt"
#   done
# done



LRS=(-6.75) # SGD
# 256 512 1024 2048 4096 8192
for BS in 50 100 200 400; do
  for lr in "${LRS[@]}"; do 
    epoch=$(( 2 * BS / 50 ))
    sig=$(awk "BEGIN {print 1.4983855926238738*$BS/400.0}")
    echo "Running batch size=$BS, lr=$lr, noise=$sig, epoch=$epoch"
    $PYTHON -m scripts.vit_muon_sgd \
      --lr "$lr" \
      --epochs "$epoch" \
      --bs "$BS" \
      --mini_bs "$BS" \
      --noise "$sig" \
      --scale 1 \
      --dataset CIFAR10 \
      --clipping_mode BK-MixOpt \
      --clipping_style layer-wise \
      --dimension 224 \
      --optimizer muon \
      --log_path "/content/drive/MyDrive/DP_muP/logs/Vit_muon_diffbs_dinfix_dpmup.txt"
  done
done

# LRS=(-13)
# for BS in 125; do
#   epoch=$(( 4 * BS / 125 ))
#   for lr in "${LRS[@]}"; do
#     sig=$(awk "BEGIN {print 4*$BS/125.0}")
#     echo "Running BS=$BS, lr=$lr, noise=$sig" 
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
#       --optimizer Adam \
#       --log_path "/content/drive/MyDrive/DP_muP/logs/MLP_Adam_diffbs_fnorm_ratio.txt"
#   done
# done


# LRS=(-9 -8 -7 -6 -5)
# # 125 250 500 1000 2000
# for lr in "${LRS[@]}"; do
#   for BS in 2000; do
#     epoch=$(( 4 * BS / 125 ))
#     sig=$(awk "BEGIN {print 1*$BS/125.0}")
#     echo "Running BS=$BS, lr=$lr, noise=$sig" 
#     # scaled_lr=$(echo "$lr + $ratio" | bc -l)
#     $PYTHON -m scripts.MLP_approx \
#       --width 1024 \
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
#       --log_path "/content/drive/MyDrive/DP_muP/logs/MLP_SGD_depth5_diffbs_approx_ratio_1024.txt"
#   done
# done


# LRS=(-4.5 -5.5 -6.5 -7.5)
# # 125 250 500 1000 2000
# for BS in 125 250 500 1000 2000; do
#   for lr in "${LRS[@]}"; do
#     epoch=$(( 4 * BS / 125 ))
#     sig=$(awk "BEGIN {print 1*$BS/125.0}")
#     echo "Running BS=$BS, lr=$lr, noise=$sig" 
#     # scaled_lr=$(echo "$lr + $ratio" | bc -l)
#     $PYTHON -m scripts.MLP_muon \
#       --width 1024 \
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
#       --optimizer Adam \
#       --log_path "/content/drive/MyDrive/DP_muP/logs/MLP_muon_depth5_diffbs_approx_ratio_1.txt"
#   done
# done

# LRS=(-7.25 -8.5 -9)
# # 125 250 500 1000 2000
# for lr in "${LRS[@]}"; do
#   for BS in 125; do
#     epoch=$(( 4 * BS / 125 ))
#     sig=$(awk "BEGIN {print 1*$BS/125.0}")
#     echo "Running BS=$BS, lr=$lr, noise=$sig" 
#     # scaled_lr=$(echo "$lr + $ratio" | bc -l)
#     $PYTHON -m scripts.MLP_muon_sgd \
#       --width 1024 \
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
#       --optimizer Adam \
#       --log_path "/content/drive/MyDrive/DP_muP/logs/MLP_muon_depth5_diffbs_approx_ratio_2.txt"
#   done
# done


# LRS=(-14 -13 -12 -11 -10)
# # 125 250 500 1000 2000
# for lr in "${LRS[@]}"; do
#   for BS in 125; do
#     epoch=$(( 4 * BS / 125 ))
#     sig=$(awk "BEGIN {print 1*$BS/125.0}")
#     echo "Running BS=$BS, lr=$lr, noise=$sig" 
#     # scaled_lr=$(echo "$lr + $ratio" | bc -l)
#     $PYTHON -m scripts.MLP_approx \
#       --width 1024 \
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
#       --optimizer Adam \
#       --log_path "/content/drive/MyDrive/DP_muP/logs/MLP_Adam_depth5_diffbs_approx_ratio_1024.txt"
#   done
# done



# LRS=(-12 -13)
# for BS in 250 500 1000 2000; do
#   epoch=$(( 4 * BS / 125 ))
#   for lr in "${LRS[@]}"; do
#     sig=$(awk "BEGIN {print 4*$BS/125.0}")
#     echo "Running BS=$BS, lr=$lr, noise=$sig" 
#     # scaled_lr=$(echo "$lr + $ratio" | bc -l)
#     $PYTHON -m scripts.MLP_unifed \
#       --width 1024 \
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
#       --optimizer Adam \
#       --log_path "/content/drive/MyDrive/DP_muP/logs/MLP_Adam_diffbs_fnorm_ratio_1024.txt"
#   done
# done



