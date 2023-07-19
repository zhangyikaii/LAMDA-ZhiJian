# import zhijian
# print(zhijian.__version__)

def select_from_input(prompt_for_select, valid_selections):
    selections2print = '\n\t'.join([f'[{idx + 1}] {i}' for idx, i in enumerate(valid_selections)])
    while True:
        selected = input(f"Please input a {prompt_for_select}, type 'help' to show the options: ")

        if selected == 'help':
            print(f"Available {prompt_for_select}(s):\n\t{selections2print}")
        elif selected.isdigit() and int(selected) >= 1 and int(selected) <= len(valid_selections):
            selected = valid_selections[int(selected) - 1]
            break
        elif selected in valid_selections:
            break
        else:
            print("Sorry, input not support.")
            print(f"Available {prompt_for_select}(s):\n\t{selections2print}")

    print(f"Your selection: {selected}")
    return selected


available_datasets = [
    'VTAB-1k.CIFAR-100', 'VTAB-1k.CLEVR-Count', 'VTAB-1k.CLEVR-Distance', 'VTAB-1k.Caltech101', 'VTAB-1k.DTD',
    'VTAB-1k.Diabetic-Retinopathy', 'VTAB-1k.Dmlab', 'VTAB-1k.EuroSAT', 'VTAB-1k.KITTI', 'VTAB-1k.Oxford-Flowers-102',
    'VTAB-1k.Oxford-IIIT-Pet', 'VTAB-1k.PatchCamelyon', 'VTAB-1k.RESISC45', 'VTAB-1k.SUN397', 'VTAB-1k.SVHN',
    'VTAB-1k.dSprites-Location', 'VTAB-1k.dSprites-Orientation', 'VTAB-1k.smallNORB-Azimuth', 'VTAB-1k.smallNORB-Elevation'
] # dataset options.
# dataset     = select_from_input('dataset', available_datasets)  # user input about dataset
dataset     = 'VTAB-1k.CIFAR-100'

# dataset_dir = input(f"Please input your dataset directory: ")   # user input about dataset directory
dataset_dir = '/data/zhangyk/data/zhijian'
print(f"Your dataset directory: {dataset_dir}")


available_example_models = {
    'timm.vit_base_patch16_224_in21k': {
        'LoRA': '(LoRA.adapt): ...->(blocks[0:12].attn.qkv){inout1}->...',
        'Adapter': '(Adapter.adapt): ...->(blocks[0:12].drop_path1){inout1}->...',
        'Convpass': ('(Convpass.adapt): ...->(blocks[0:12].norm1){in1}->(blocks[0:11].drop_path1){in2}->...,' # follow the next line
                     '(Convpass.adapt): ...->{in1}(blocks[0:11].norm2)->(blocks[0:12].drop_path2){in2}->...'),
        'None': None
    }
} # model options, Dict(model name: Dict(add-in structure name: add-in blitz configuration)).

model = 'timm.vit_base_patch16_224_in21k'
# model = select_from_input('model', list(available_example_models.keys())) # user input about model


availables   = available_example_models[model]
# config_blitz = availables[select_from_input('add-in structure', list(availables.keys()))]   # user input about add-in structure
config_blitz = '(LoRA.adapt): ...->(blocks[0:12].attn.qkv){inout1}->...'


available_example_reuse_modules = {
    'timm.vit_base_patch16_224_in21k': {
        'linear layer only': 'addin,head,fc_norm',
        'the last block and the linear layer (Partial-1)': 'addin,blocks[11],head,fc_norm',
        'the last two blocks and the linear layer (Partial-2)': 'addin,blocks[10:12],head,fc_norm',
        'the last four blocks and the linear layer (Partial-4)': 'addin,blocks[8:12],head,fc_norm',
        'all parameters': ''
    }
}

availables          = available_example_reuse_modules[model]
reuse_keys_blitz = 'addin,head,fc_norm'
# reuse_keys_blitz = availables[select_from_input('reuse modules', list(availables.keys()))] # user input about reuse modules


from zhijian.trainers.base import prepare_args
from zhijian.models.utils import pprint, dict2args
training_mode = 'finetune'
args = dict2args({
    'log_url': 'your/log/directory',        # log directory
    'dataset': dataset,                     # dataset
    'dataset_dir': dataset_dir,             # dataset directory
    'model': model,                         # backbone network
    'config_blitz': config_blitz,           # addin blitz configuration
    'training_mode': training_mode,         # training mode
    'reuse_keys_blitz': reuse_keys_blitz,   # reuse keys blitz configuration
    'optimizer': 'adam',                    # optimizer
    'batch_size': 64,                       # batch size
    'num_workers': 8,                       # num workers
    'max_epoch': 5,                         # max epoch
    'eta_min': 0,                           # eta_min of CosineAnnealingLR
    'lr': 1e-3,                             # learning rate
    'wd': 5e-5,                             # weight decay
    'gpu': '0',                             # gpu id
    'seed': 0,                              # random seed
    'verbose': True,                        # control the verbosity of the output
    'only_do_test': False                   # test flag
})
# Namespace(dataset='VTAB-1k.CIFAR-100', dataset_dir='/data/zhangyk/data/zhijian', model='timm.vit_base_patch16_224_in21k', config='', config_blitz='(LoRA.adapt): ...->(blocks[0:12].attn.qkv){inout1}->...', pretrained_url=[], training_mode='finetune', reuse_keys=None, reuse_keys_blitz='addin,head,fc_norm', batch_size=8, max_epoch=30, optimizer='adam', lr=0.01, wd=1e-05, mom=0.9, lr_scheduler='cosine', eta_min=0, criterion='cross-entropy', seed=1, gpu='0', num_workers=8, log_url='your/log/directory', time_str='', verbose=True, only_do_test=False)

args = prepare_args(args, update_default=True)
pprint(vars(args))


import torch
import os
assert torch.cuda.is_available()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
torch.cuda.set_device(int(args.gpu))


from zhijian.trainers.finetune import get_model
model, model_args, device = get_model(args)


from zhijian.data.base import prepare_vision_dataloader
train_loader, val_loader, num_classes = prepare_vision_dataloader(args, model_args)

import torch.nn as nn
import torch.optim as optim
optimizer = optim.Adam(
    model.parameters(),
    lr=args.lr,
    weight_decay=args.wd
)
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    args.max_epoch,
    eta_min=args.eta_min
)
criterion = nn.CrossEntropyLoss()


from zhijian.trainers.base import prepare_trainer
trainer = prepare_trainer(
    args,
    model=model, model_args=model_args, device=device,
    train_loader=train_loader, val_loader=val_loader, num_classes=num_classes,
    optimizer=optimizer, lr_scheduler=lr_scheduler, criterion=criterion
)


trainer.fit()
trainer.test()
