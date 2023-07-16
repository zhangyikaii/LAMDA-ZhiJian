model=$1
config=zhijian/models/configs/$2
dataset=$3
dataset_dir=$4
initial_checkpoint=$5

echo -e "[$(date)] ($0) Input Parameters:\n    --model \"$model\"\n    --config \"$config\"\n    --dataset \"$dataset\"\n    --dataset-dir \"$dataset_dir\"\n    --initial-checkpoint \"$initial_checkpoint\"\n"

confirm=""
while [[ "$confirm" != "Y" && "$confirm" != "y" && "$confirm" != "n" && "$confirm" != "N" ]]; do
  read -p "Is the information correct? [Y/n] " confirm
done

if [[ "$confirm" == "n" ]] || [[ "$confirm" == "N" ]]; then
  exit 1
fi

python main.py \
  --model "$model" \
  --config "$config" \
  --dataset "$dataset" \
  --dataset-dir "$dataset_dir" \
  --initial-checkpoint "$initial_checkpoint" \
  --training-mode nearest_class_mean \
  --reuse-keys \
  --verbose \
  --gpu 0

# bash run/train_nearest_class_mean.sh timm.vit_base_patch16_224_in21k lora.yaml VTAB-1k.CIFAR-100 /data/zhangyk/data/petl /data/zhangyk/data/petl/model/ViT-B_16.npz