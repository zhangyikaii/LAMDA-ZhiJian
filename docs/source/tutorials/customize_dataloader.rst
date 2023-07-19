.. role:: lamdablue
    :class: lamdablue

.. role:: lamdaorange
    :class: lamdaorange

.. raw:: html

    <style>

    .lamdablue {
        color: #47479e;
        font-weight: bold;
    }
    .lamdaorange {
        color: #fd4d01;
        font-weight: bold;
    }

    </style>


Customize Dataloader
====================

.. raw:: html

   <span style="font-size: 25px;">📂</span>
   <p></p>

This example shows **how to customize your own dataloader** for a new target dataset in :lamdaorange:`Z`:lamdablue:`h`:lamdablue:`i`:lamdaorange:`J`:lamdablue:`i`:lamdablue:`a`:lamdablue:`n`.

Feel free to deploy model reusability technology on *any* dataset, with loading in the conventional `PyTorch` style.


+ Configure basic parameters, without dataset configuration:

  ::

   training_mode = 'finetune'
   args = get_args(
       model='timm.vit_base_patch16_224_in21k',                              # backbone network
       config_blitz='(LoRA.adapt): ...->(blocks[0:12].attn.qkv){in1}->...',  # addin blitz configuration
       training_mode='finetune',                                             # training mode
       optimizer='adam',                                                     # optimizer
       lr=1e-2,                                                              # learning rate
       wd=1e-5,                                                              # weight decay
       verbose=True                                                          # control the verbosity of the output
   )
   pprint(vars(args))

  .. code-block:: bash

    $ {'aa': None,
      'addins': [{'hook': [['adapt', 'post']],
                  'location': [['blocks', 0, 'attn', 'qkv']],
                  'name': 'LoRA'},
                  {'hook': [['adapt', 'post']],
                  'location': [['blocks', 1, 'attn', 'qkv']],
                  'name': 'LoRA'},
                  {'hook': [['adapt', 'post']],
                  'location': [['blocks', 2, 'attn', 'qkv']],
                  'name': 'LoRA'},
                  {'hook': [['adapt', 'post']],
                  'location': [['blocks', 3, 'attn', 'qkv']],
                  'name': 'LoRA'},
                  {'hook': [['adapt', 'post']],
                  'location': [['blocks', 4, 'attn', 'qkv']],
                  'name': 'LoRA'},
                  {'hook': [['adapt', 'post']],
                  'location': [['blocks', 5, 'attn', 'qkv']],
                  'name': 'LoRA'},
                  {'hook': [['adapt', 'post']],
                  'location': [['blocks', 6, 'attn', 'qkv']],
                  'name': 'LoRA'},
                  {'hook': [['adapt', 'post']],
                  'location': [['blocks', 7, 'attn', 'qkv']],
                  'name': 'LoRA'},
                  {'hook': [['adapt', 'post']],
                  'location': [['blocks', 8, 'attn', 'qkv']],
                  'name': 'LoRA'},
                  {'hook': [['adapt', 'post']],
                  'location': [['blocks', 9, 'attn', 'qkv']],
                  'name': 'LoRA'},
                  {'hook': [['adapt', 'post']],
                  'location': [['blocks', 10, 'attn', 'qkv']],
                  'name': 'LoRA'},
                  {'hook': [['adapt', 'post']],
                  'location': [['blocks', 11, 'attn', 'qkv']],
                  'name': 'LoRA'}],
      'amp': False,
      'amp_dtype': 'float16',
      'amp_impl': 'native',
      'aot_autograd': False,
      'aug_repeats': 0,
      'aug_splits': 0,
      'batch_size': 64,
      'bce_loss': False,
      ...
      'warmup_prefix': False,
      'wd': 5e-05,
      'weight_decay': 2e-05,
      'worker_seeding': 'all'}


+ Set up the GPU and prepare the model:

  ::

   assert torch.cuda.is_available()
   os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
   torch.cuda.set_device(int(args.gpu))

   model, model_args, device = get_model(args)

+ Organize the custom dataset in the following structure: within the :code:`your/dataset/dir` directory, create a separate folder for each category, and store all the data corresponding to each category within its respective folder:

  .. code-block:: bash

    /your/dataset/directory
    ├── train
    │   ├── class_1
    │   │   ├── train_class_1_img_1.jpg
    │   │   ├── train_class_1_img_2.jpg
    │   │   ├── train_class_1_img_3.jpg
    │   │   └── ...
    │   ├── class_2
    │   │   ├── train_class_2_img_1.jpg
    │   │   └── ...
    │   ├── class_3
    │   │   └── ...
    │   ├── class_4
    │   │   └── ...
    │   ├── class_5
    │   │   └── ...
    └── test
        ├── class_1
        │   ├── test_class_1_img_1.jpg
        │   ├── test_class_1_img_2.jpg
        │   ├── test_class_1_img_3.jpg
        │   └── ...
        ├── class_2
        │   ├── test_class_2_img_1.jpg
        │   └── ...
        ├── class_3
        │   └── ...
        ├── class_4
        │   └── ...
        └── class_5
            └── ...


+ Set up the custom dataset:

  ::

   train_transform = transforms.Compose([
       transforms.RandomResizedCrop(224),
       transforms.RandomHorizontalFlip(),
       transforms.ToTensor(),
       transforms.Normalize(
           mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
       )
   ])
   val_transform = transforms.Compose([
       transforms.Resize(256),
       transforms.CenterCrop(224),
       transforms.ToTensor(),
       transforms.Normalize(
           mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
       )
   ])

   train_dataset = ImageFolder(root='/your/dataset/directory/train', transform=train_transform)
   val_dataset = ImageFolder(root='/your/dataset/directory/test', transform=val_transform)


+ Implement the corresponding loader:

  ::

   train_loader = torch.utils.data.DataLoader(
           train_dataset,
           batch_size=args.batch_size,
           num_workers=args.num_workers,
           pin_memory=True,
           shuffle=True
       )
   val_loader = torch.utils.data.DataLoader(
           val_dataset,
           batch_size=args.batch_size,
           num_workers=args.num_workers,
           pin_memory=True,
           shuffle=True
       )
   num_classes = len(train_dataset.classes)

+ Set up the optimizer and the loss function:

  ::

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

+ Set up the trainer and start training and testing:

  ::

   trainer = prepare_trainer(
       args,
       model=model, model_args=model_args, device=device,
       train_loader=train_loader,
       val_loader=val_loader,
       num_classes=num_classes,
       optimizer=optimizer,
       lr_scheduler=lr_scheduler,
       criterion=criterion
   )

   trainer.fit()
   trainer.test()

  .. code-block:: bash

    $ Log level set to: INFO
      Log files are recorded in: your/log/directory/0718-20-10-57-792
      Trainable/total parameters of the model: 0.30M / 86.10M (0.34700%)
  
          Epoch   GPU Mem.       Time       Loss         LR
              1/5      5.48G      1.686       1.73      0.001: 100%|██████████| 1.00/1.00 [00:01<00:00, 1.22s/batch]
  
          Epoch   GPU Mem.       Time      Acc@1      Acc@5
              1/5      5.48G     0.3243         16        100: 100%|██████████| 1.00/1.00 [00:00<00:00, 2.39batch/s]
      ***   Best results: [Acc@1: 16.0], [Acc@5: 100.0]
  
          Epoch   GPU Mem.       Time       Loss         LR
              2/5       5.6G      1.093      1.448 0.00090451: 100%|██████████| 1.00/1.00 [00:00<00:00, 1.52batch/s]
  
          Epoch   GPU Mem.       Time      Acc@1      Acc@5
              2/5       5.6G     0.2647         12        100: 100%|██████████| 1.00/1.00 [00:00<00:00, 2.58batch/s]
      ***   Best results: [Acc@1: 12.0], [Acc@5: 100.0]
  
          Epoch   GPU Mem.       Time       Loss         LR
              3/5       5.6G      1.088      1.369 0.00065451: 100%|██████████| 1.00/1.00 [00:00<00:00, 1.54batch/s]
  
          Epoch   GPU Mem.       Time      Acc@1      Acc@5
              3/5       5.6G     0.2899         12        100: 100%|██████████| 1.00/1.00 [00:00<00:00, 2.54batch/s]
      ***   Best results: [Acc@1: 12.0], [Acc@5: 100.0]
  
          Epoch   GPU Mem.       Time       Loss         LR
              4/5       5.6G      1.067      1.403 0.00034549: 100%|██████████| 1.00/1.00 [00:00<00:00, 1.53batch/s]
  
          Epoch   GPU Mem.       Time      Acc@1      Acc@5
              4/5       5.6G     0.2879         16        100: 100%|██████████| 1.00/1.00 [00:00<00:00, 2.42batch/s]
      ***   Best results: [Acc@1: 16.0], [Acc@5: 100.0]
  
          Epoch   GPU Mem.       Time       Loss         LR
              5/5       5.6G      1.077      1.342 9.5492e-05: 100%|██████████| 1.00/1.00 [00:00<00:00, 1.55batch/s]
  
          Epoch   GPU Mem.       Time      Acc@1      Acc@5
              5/5       5.6G      0.246         16        100: 100%|██████████| 1.00/1.00 [00:00<00:00, 2.79batch/s]
      ***   Best results: [Acc@1: 16.0], [Acc@5: 100.0]
  
          Epoch   GPU Mem.       Time      Acc@1      Acc@5
              1/5       5.6G     0.2901         16        100: 100%|██████████| 1.00/1.00 [00:00<00:00, 2.52batch/s]
      ***   Best results: [Acc@1: 16.0], [Acc@5: 100.0]
      (16.0, 100.0)
