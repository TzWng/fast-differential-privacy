#288 512 1152 2048 3200 4608
LRS=(-8 -7 -6 -5)

for depth in 3 6 12 24; do
  for lr in "${LRS[@]}"; do
    sig=$(awk "BEGIN {print sqrt($depth/3)*16.0}")
    echo "Running width=$wid, depth=$depth, lr=$lr, noise=$sig"
    $PYTHON -m scripts.MLP_unifed \
      --width 256 \
      --layer "$depth"\
      --lr "$lr" \
      --epochs 16 \
      --bs 500 \
      --mini_bs 500 \
      --epsilon 2 \
      --noise "$sig" \
      --clipping_mode BK-MixOpt \
      --clipping_style layer-wise \
      --cifar_data CIFAR10 \
      --dimension 32 \
      --optimizer SGD \
      --log_path "/content/drive/MyDrive/DP_muP/logs/MLP_SGD_diffdepth_truenorm_ratio.txt"
  done
