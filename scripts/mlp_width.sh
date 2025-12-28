#!/bin/bash
PYTHON=python3.10

PROJECT_ROOT=/content/fast-differential-privacy
export PYTHONPATH="$PROJECT_ROOT"


# LRS=(-5.25 -5.75) # SGD
# # 256 512 1024 2048 4096 8192
# for lr in "${LRS[@]}"; do 
#   for wid in 512 1024 2048 4096 8192; do
#     echo "Running width=$wid, lr=$lr, dim=32"
#     $PYTHON -m scripts.dpmup_sgd \
#       --width "$wid" \
#       --lr "$lr" \
#       --epochs 10 \
#       --bs 500 \
#       --mini_bs 500 \
#       --noise 0.28948843479156494 \
#       --seed 3 \
#       --cifar_data CIFAR10 \
#       --clipping_mode BK-MixOpt \
#       --clipping_style layer-wise \
#       --dimension 32 \
#       --optimizer SGD \
#       --log_path "/content/drive/MyDrive/DP_muP/logs/MLP_SGD_depth5_l2s_approx_ratio_dinfix_dpmup.txt"
#   done
# done




LRS=(-6 -5.5 -5 -4.5 -4 -3.5) # SGD
# 256 512 1024 2048 4096 8192
for lr in "${LRS[@]}"; do 
  for wid in 8192; do
    echo "Running width=$wid, lr=$lr, dim=32"
    $PYTHON -m scripts.dpmup_sgd \
      --width "$wid" \
      --lr "$lr" \
      --epochs 10 \
      --bs 500 \
      --mini_bs 500 \
      --noise 7.067494947837502 \
      --cifar_data CIFAR10 \
      --clipping_mode BK-MixOpt \
      --clipping_style layer-wise \
      --dimension 32 \
      --optimizer SGD \
      --log_path "/content/drive/MyDrive/DP_muP/logs/temp_sp.txt"
  done
done


# # LRS=(-5.5 -6) # SGD
# LRS=(-1.5 -1 -0.5 0 0.5) # SGD
# # 288 512 1152 2048 4608 8192
# for lr in "${LRS[@]}"; do
#   for wid in 256 512 1024 2048 4096; do 
#     echo "Running width=$wid, lr=$lr, dim=32"
#     $PYTHON -m scripts.MLP_approx \
#       --width "$wid" \
#       --lr "$lr" \
#       --epochs 10 \
#       --bs 500 \
#       --mini_bs 500 \
#       --epsilon 2 \
#       --noise 2 \
#       --clipping_mode BK-MixOpt \
#       --clipping_style layer-wise \
#       --cifar_data CIFAR10 \
#       --dimension 32 \
#       --optimizer SGD \
#       --log_path "/content/drive/MyDrive/DP_muP/logs/MLP_SGD_depth5_diffwidth_approx_ratio_dinfix_sp_new.txt"
#   done
# done


# LRS=(-6.25) # SGD
# # 288 512 1152 2048 4608 8192
# for wid in 8192 4096 2048 1024 512 256; do 
#   for lr in "${LRS[@]}"; do
#     echo "Running width=$wid, lr=$lr, dim=32"
#     $PYTHON -m scripts.MLP_approx \
#       --width "$wid" \
#       --lr "$lr" \
#       --epochs 10 \
#       --bs 500 \
#       --mini_bs 500 \
#       --epsilon 2 \
#       --noise 2 \
#       --clipping_mode BK-MixOpt \
#       --clipping_style layer-wise \
#       --cifar_data CIFAR10 \
#       --dimension 32 \
#       --optimizer SGD \
#       --log_path "/content/drive/MyDrive/DP_muP/logs/MLP_SGD_depth5_diffwidth_approx_ratio_dinfix_new_4.txt"
#   done
# done


# LRS=(-1.75) # SGD
# # 288 512 1152 2048 4608 8192
# for lr in "${LRS[@]}"; do
#   for wid in 288; do 
#     dim=$(awk "BEGIN {print sqrt($wid/128.0)*8.0}")
#     echo "Running width=$wid, lr=$lr, noise=$sig, dim=$dim"
#     $PYTHON -m scripts.MLP_nonDP_muP \
#       --width "$wid" \
#       --lr "$lr" \
#       --epochs 10 \
#       --bs 500 \
#       --mini_bs 500 \
#       --cifar_data CIFAR10 \
#       --dimension "$dim" \
#       --optimizer SGD \
#       --log_path "/content/drive/MyDrive/DP_muP/logs/MLP_SGD_depth5_diffwidth_nonDP_sp_4.txt"
#   done
# done

# LRS=(-2.5 -2) # SGD
# # 256 512 1024 2048 4096 8192
# for wid in 256; do 
#   for lr in "${LRS[@]}"; do
#     echo "Running width=$wid, lr=$lr, noise=$sig, dim=32"
#     $PYTHON -m scripts.MLP_nonDP_muP \
#       --width "$wid" \
#       --lr "$lr" \
#       --epochs 10 \
#       --bs 500 \
#       --mini_bs 500 \
#       --cifar_data CIFAR10 \
#       --dimension 32 \
#       --optimizer SGD \
#       --log_path "/content/drive/MyDrive/DP_muP/logs/MLP_SGD_depth5_l2s_nonDP_sp_seed2.txt"
#   done
# done


# LRS=(-4.75 -7.5 -8 -8.5 -9) # SGD
# # 256 512 1024 2048 4096 8192
# for wid in 256 512 1024 2048 4096 8192; do 
#   for lr in "${LRS[@]}"; do
#     echo "Running width=$wid, lr=$lr, noise=$sig, dim=32"
#     $PYTHON -m scripts.MLP_nonDP_muP \
#       --width "$wid" \
#       --lr "$lr" \
#       --epochs 10 \
#       --bs 500 \
#       --mini_bs 500 \
#       --cifar_data CIFAR10 \
#       --dimension 32 \
#       --optimizer SGD \
#       --log_path "/content/drive/MyDrive/DP_muP/logs/MLP_SGD_depth5_l2s_nonDP_mup_seed2.txt"
#   done
# done

# LRS=(-6.5) # SGD
# # 256 512 1024 2048 4096 8192
# for wid in 8192; do 
#   for lr in "${LRS[@]}"; do
#     dim=32
#     echo "Running width=$wid, lr=$lr, noise=$sig, dim=$dim"
#     $PYTHON -m scripts.MLP_nonDP_muP \
#       --width "$wid" \
#       --lr "$lr" \
#       --epochs 5 \
#       --bs 500 \
#       --mini_bs 500 \
#       --cifar_data CIFAR10 \
#       --dimension 32 \
#       --optimizer muon \
#       --log_path "/content/drive/MyDrive/DP_muP/logs/MLP_muon_depth5_diffwidth_nonDP_doublesgd.txt"
#   done
# done

# LRS=(-9.5) # SGD
# # 256 512 1024 2048 4096 8192
# for wid in 8192; do 
#   for lr in "${LRS[@]}"; do
#     dim=32
#     echo "Running width=$wid, lr=$lr, noise=$sig, dim=$dim"
#     $PYTHON -m scripts.MLP_nonDP_muP \
#       --width "$wid" \
#       --lr "$lr" \
#       --epochs 5 \
#       --bs 500 \
#       --mini_bs 500 \
#       --cifar_data CIFAR10 \
#       --dimension 32 \
#       --optimizer muon \
#       --log_path "/content/drive/MyDrive/DP_muP/logs/MLP_muon_depth5_diffwidth_nonDP_all.txt"
#   done
# done

# LRS=(-5.75) # SGD
# # 288 512 1152 2048 4608 8192
# for wid in 4608 8192; do 
#   for lr in "${LRS[@]}"; do
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
#       --log_path "/content/drive/MyDrive/DP_muP/logs/MLP_SGD_depth5_diffwidth_approx_ratio_mup_3.txt"
#   done
# done


# LRS=(-1.75) # SGD
# # 288 512 1152 2048 3200 4608
# for wid in 288; do
#   for lr in "${LRS[@]}"; do 
#     sig=$(awk "BEGIN {print 2.0*sqrt(128.0/$wid)}")
#     dim=$(awk "BEGIN {print sqrt($wid/128.0)*8.0}")
#     echo "Running width=$wid, lr=$lr, noise=$sig, dim=$dim"
#     $PYTHON -m scripts.MLP_sp \
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
#       --log_path "/content/drive/MyDrive/DP_muP/logs/MLP_SGD_depth5_diffwidth_approx_ratio_sp_4.txt"
#   done
# done


# LRS=(0) # SGD
# # 288 512 1152 2048 3200 4608
# for wid in 288; do
#   for lr in "${LRS[@]}"; do 
#     sig=$(awk "BEGIN {print 2.0*sqrt(128.0/2048)}")
#     echo "Running width=$wid, lr=$lr, noise=$sig, dim=$dim"
#     $PYTHON -m scripts.MLP_sp \
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
#       --dimension 32 \
#       --optimizer SGD \
#       --log_path "/content/drive/MyDrive/DP_muP/logs/MLP_SGD_depth5_diffwidth_approx_ratio_sp_nosnr.txt"
#   done
# done



# LRS=(-2 2.5) # SGD
# # 256 512 1024 2048 4096 8192
# for wid in 256 512 1024 2048 4096 8192; do 
#   for lr in "${LRS[@]}"; do
#     echo "Running width=$wid, lr=$lr, noise=$sig, dim=32"
#     $PYTHON -m scripts.MLP_clipping_only \
#       --width "$wid" \
#       --lr "$lr" \
#       --epochs 10 \
#       --bs 500 \
#       --mini_bs 500 \
#       --epsilon 2 \
#       --noise 0 \
#       --clipping_mode BK-MixOpt \
#       --clipping_style layer-wise \
#       --cifar_data CIFAR10 \
#       --dimension 32 \
#       --optimizer Adam \
#       --log_path "/content/drive/MyDrive/DP_muP/logs/MLP_Adam_depth5_diffwidth_co.txt"
#   done
# done

# LRS=(-2.5 -1.5 -0.5) # SGD
# # 256 512 1024 2048 4096 8192
# for wid in 256 512 1024 2048 4096 8192; do 
#   for lr in "${LRS[@]}"; do
#     echo "Running width=$wid, lr=$lr, noise=$sig, dim=32"
#     $PYTHON -m scripts.MLP_clipping_only \
#       --width "$wid" \
#       --lr "$lr" \
#       --epochs 10 \
#       --bs 500 \
#       --mini_bs 500 \
#       --epsilon 2 \
#       --noise 0 \
#       --clipping_mode BK-MixOpt \
#       --clipping_style layer-wise \
#       --cifar_data CIFAR10 \
#       --dimension 32 \
#       --optimizer SGD \
#       --log_path "/content/drive/MyDrive/DP_muP/logs/MLP_SGD_depth5_diffwidth_co.txt"
#   done
# done


# LRS=(-7.75) # SGD
# # 256 512 1024 2048 4096
# for wid in 256 512 1024 2048 4096 8192; do 
#   for lr in "${LRS[@]}"; do
#     sig=$(awk "BEGIN {print 1.0*sqrt(512.0/$wid)}")
#     echo "Running width=$wid, lr=$lr, noise=$sig, dim=32"
#     $PYTHON -m scripts.MLP_muon_sgd \
#       --width "$wid" \
#       --lr "$lr" \
#       --epochs 10 \
#       --bs 500 \
#       --mini_bs 500 \
#       --epsilon 2 \
#       --noise 2 \
#       --clipping_mode BK-MixOpt \
#       --clipping_style layer-wise \
#       --cifar_data CIFAR10 \
#       --dimension 32 \
#       --optimizer SGD \
#       --log_path "/content/drive/MyDrive/DP_muP/logs/MLP_Muon_depth5_diffwidth_approx_ratio_double_sgd_new_1.txt"
#   done
# done





# LRS=(-12.5 -13 -13.5 -14) # SGD
# # 256 512 1024 2048 4096
# for wid in 8192; do 
#   for lr in "${LRS[@]}"; do
#     echo "Running width=$wid, lr=$lr, noise=$sig, dim=32"
#     $PYTHON -m scripts.MLP_muon \
#       --width "$wid" \
#       --lr "$lr" \
#       --epochs 10 \
#       --bs 500 \
#       --mini_bs 500 \
#       --epsilon 2 \
#       --noise 2 \
#       --clipping_mode BK-MixOpt \
#       --clipping_style layer-wise \
#       --cifar_data CIFAR10 \
#       --dimension 32 \
#       --optimizer SGD \
#       --log_path "/content/drive/MyDrive/DP_muP/logs/MLP_Muon_depth5_diffwidth_approx_new_1.txt"
#   done
# done




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
