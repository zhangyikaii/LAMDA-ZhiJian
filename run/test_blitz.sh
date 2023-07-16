model=$1
config_blitz=$2
dataset=$3
dataset_dir=$4
initial_checkpoint=$5
pretrained_url=$6

echo -e "[$(date)] ($0) Input Parameters:\n    --model \"$model\"\n    --config-blitz \"$config_blitz\"\n    --dataset \"$dataset\"\n    --dataset-dir \"$dataset_dir\"\n    --initial-checkpoint \"$initial_checkpoint\"\n    --pretrained-url \"$pretrained_url\"\n"

confirm=""
while [[ "$confirm" != "Y" && "$confirm" != "y" && "$confirm" != "n" && "$confirm" != "N" ]]; do
    read -p "Is the information correct? [Y/n] " confirm
done

if [[ "$confirm" == "n" ]] || [[ "$confirm" == "N" ]]; then
  exit 1
fi

python main.py \
    --model "$model" \
    --config-blitz "$config_blitz" \
    --dataset "$dataset" \
    --dataset-dir "$dataset_dir" \
    --initial-checkpoint "$initial_checkpoint" \
    --pretrained-url "$pretrained_url" \
    --reuse-keys \
    --only-do-test \
    --verbose \
    --gpu 0

# bash run/test_blitz.sh timm.vit_base_patch16_224_in21k "(LoRA.adapt): ...->(blocks[0:12].attn.qkv){inout1}->..." VTAB-1k.CIFAR-100 /data/zhangyk/data/petl /data/zhangyk/data/petl/model/ViT-B_16.npz /data/zhangyk/models/kel_logs/lora_cifar/best.pt
