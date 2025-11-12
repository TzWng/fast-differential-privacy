LRS=(-14 -13.5 -13 -12.5)

for BS in 125 250 500 1000 2000; do
  epoch=$(( 4 * BS / 125 ))
  for lr in "${LRS[@]}"; do
    echo "Running BS=$BS, lr=$lr, epoch=$epoch" 
    $PYTHON -m scripts.MLP_unifed \
      --width 256 \
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
      --log_path "/content/drive/MyDrive/DP_muP/logs/MLP_Adam_diffbs_truenorm.txt"
  done
done
