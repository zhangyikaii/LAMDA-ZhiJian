source activate zykcym
python main.py \
  --model "timm.vit_base_patch16_224_in21k" \
  --config "finetune.yaml" \
  --dataset "VTAB-1k.CIFAR-100" \
  --dataset-dir "/data/zhangyk/data/petl" \
  --initial-checkpoint "/data/zhangyk/data/petl/model/ViT-B_16.npz" \
  --verbose \
  --only-do-test \
  --training-mode nearest_class_mean \
  --gpu 0
