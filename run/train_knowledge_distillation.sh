model=$1
config=zhijian/models/configs/$2
dataset=$3
dataset_dir=$4
initial_checkpoint=$5
t_config=$6
kd_mode=$7

echo -e "[$(date)] ($0) Input Parameters:\n    --model \"$model\"\n    --config \"$config\"\n    --dataset \"$dataset\"\n    --dataset-dir \"$dataset_dir\"\n    --initial-checkpoint \"$initial_checkpoint\"\n    --t-config \"$t_config\"\n    --kd-mode \"$7\"\n"

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
  --training-mode knowledge_distillation \
  --t-config "$t_config" \
  --kd-mode "$kd_mode" \
  --verbose \
  --gpu 0

# bash run/train_knowledge_distillation.sh timm.vit_base_patch16_224_in21k lora.yaml VTAB-1k.CIFAR-100 /data/zhangyk/data/petl /data/zhangyk/data/petl/model/ViT-B_16.npz zhijian/models/configs/twin_ptm.yaml st
