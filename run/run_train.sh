source activate zykcym
python main.py \
    --model "timm.vit_base_patch16_224_in21k" \
    --config "zhijian/models/configs/partial_1.yaml" \
    --dataset "VMT" \
    --dataset-dir "/data/zhangyk/data/petl" \
    --initial-checkpoint "/data/zhangyk/data/petl/model/ViT-B_16.npz" \
    --max-epoch 2 \
    --test-interval 2 \
    --alchemy \
    --gpu 0
