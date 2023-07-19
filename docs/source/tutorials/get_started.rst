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


Get Started
=================================

.. raw:: html

   <span style="font-size: 25px;">👋🏼</span>
   <p></p>

This example shows how to **reuse pre-trained models** with :lamdaorange:`Z`:lamdablue:`h`:lamdablue:`i`:lamdaorange:`J`:lamdablue:`i`:lamdablue:`a`:lamdablue:`n`.

- **Why** :lamdaorange:`Z`:lamdablue:`h`:lamdablue:`i`:lamdaorange:`J`:lamdablue:`i`:lamdablue:`a`:lamdablue:`n`?

    - **Effortlessly** and **swiftly** customize and reuse *any* pre-trained model on *any* dataset you want.

- **All in just 10 minutes**

    - 1 min to install `zhijian`
    - 2 mins to select the dataset
    - 3 mins to customize the model structure
    - 4 mins to deploy training and test process

🚀 **Let's get started!**

Install :lamdaorange:`Z`:lamdablue:`h`:lamdablue:`i`:lamdaorange:`J`:lamdablue:`i`:lamdablue:`a`:lamdablue:`n`
-------------------------

  .. code-block:: bash

    $ pip install zhijian

After installation, open your python console and type
  ::

    import zhijian
    print(zhijian.__version__)

If no error occurs, you have successfully installed.

Select Dataset
-------------------------

:lamdaorange:`Z`:lamdablue:`h`:lamdablue:`i`:lamdaorange:`J`:lamdablue:`i`:lamdablue:`a`:lamdablue:`n` provides the loading interface for *19 datasets* spanning several domains, including general objects, animals and plants, food and daily necessities, medicine, and remote sensing. These datasets cover the `VTAB` benchmark.

Customize your own dataset, please see TODO.

+ For better prompting, we first create a tool function that guides the input:
  ::

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

        return selected

+ Now, run the following code block, input and select the target dataset and corresponding directory to be deployed:
  ::

    available_datasets = [
        'VTAB-1k.CIFAR-100', 'VTAB-1k.CLEVR-Count', 'VTAB-1k.CLEVR-Distance', 'VTAB-1k.Caltech101', 'VTAB-1k.DTD',
        'VTAB-1k.Diabetic-Retinopathy', 'VTAB-1k.Dmlab', 'VTAB-1k.EuroSAT', 'VTAB-1k.KITTI', 'VTAB-1k.Oxford-Flowers-102',
        'VTAB-1k.Oxford-IIIT-Pet', 'VTAB-1k.PatchCamelyon', 'VTAB-1k.RESISC45', 'VTAB-1k.SUN397', 'VTAB-1k.SVHN',
        'VTAB-1k.dSprites-Location', 'VTAB-1k.dSprites-Orientation', 'VTAB-1k.smallNORB-Azimuth', 'VTAB-1k.smallNORB-Elevation'
    ] # dataset options.
    dataset     = select_from_input('dataset', available_datasets)  # user input about dataset
    dataset_dir = input(f"Please input your dataset directory: ")   # user input about dataset directory

  .. code-block:: bash

    $ Available dataset(s):
              [1] VTAB-1k.CIFAR-100
              [2] VTAB-1k.CLEVR-Count
              [3] VTAB-1k.CLEVR-Distance
              [4] VTAB-1k.Caltech101
              [5] VTAB-1k.DTD
              [6] VTAB-1k.Diabetic-Retinopathy
              [7] VTAB-1k.Dmlab
              [8] VTAB-1k.EuroSAT
              [9] VTAB-1k.KITTI
              [10] VTAB-1k.Oxford-Flowers-102
              [11] VTAB-1k.Oxford-IIIT-Pet
              [12] VTAB-1k.PatchCamelyon
              [13] VTAB-1k.RESISC45
              [14] VTAB-1k.SUN397
              [15] VTAB-1k.SVHN
              [16] VTAB-1k.dSprites-Location
              [17] VTAB-1k.dSprites-Orientation
              [18] VTAB-1k.smallNORB-Azimuth
              [19] VTAB-1k.smallNORB-Elevation

Customize Pre-trained Model
-------------------------

Seamlessly modify the structure is possible. :lamdaorange:`Z`:lamdablue:`h`:lamdablue:`i`:lamdaorange:`J`:lamdablue:`i`:lamdablue:`a`:lamdablue:`n` welcomes any base model and any additional modifications. The base part supports:

    + 🤗 **Hugging Face** series — `PyTorch Image Models (timm) <https://github.com/huggingface/pytorch-image-models>`_, `Transformers <https://github.com/huggingface/transformers>`_, **PyTorch** series — `Torchvision <https://pytorch.org/vision/stable/models.html>`_, and **OpenAI** series — `CLIP <https://github.com/openai/CLIP>`_.
    + Other popular projects, *e.g.*, `vit-pytorch <https://github.com/lucidrains/vit-pytorch>`_ (stars `14k <https://github.com/lucidrains/vit-pytorch/stargazers>`_) and **any custom** architecture.
    + **Large Language Model**, including `baichuan <https://huggingface.co/baichuan-inc/baichuan-7B>`_ (*7B*), `LLaMA <https://github.com/facebookresearch/llama>`_ (*7B/13B*), and `BLOOM <https://huggingface.co/bigscience/bloom>`_ (*560M/1.1B/1.7B/3B/7.1B*).

:lamdaorange:`Z`:lamdablue:`h`:lamdablue:`i`:lamdaorange:`J`:lamdablue:`i`:lamdablue:`a`:lamdablue:`n` also includes assembling additional tuning structures, similar to building *LEGO* bricks. For more detailed customization of each part, please see `here <TODO>`_.

Adapt the `Vision Transformer` structure just requires **1~3** lines of code.

+ Now, run the following code block, input and select the model architecture:
  ::

    available_example_models = {
        'timm.vit_base_patch16_224_in21k': {
            'LoRA': '(LoRA.adapt): ...->(blocks[0:12].attn.qkv){inout1}->...',
            'Adapter': '(Adapter.adapt): ...->(blocks[0:12].drop_path1){inout1}->...',
            'Convpass': ('(Convpass.adapt): ...->(blocks[0:12].norm1){in1}->(blocks[0:11].drop_path1){in2}->...,' # follow the next line
                        '(Convpass.adapt): ...->{in1}(blocks[0:11].norm2)->(blocks[0:12].drop_path2){in2}->...'),
            'None': None
        }
    } # model options, Dict(model name: Dict(add-in structure name: add-in blitz configuration)).

    model = select_from_input('model', list(available_example_models.keys())) # user input about model

  .. code-block:: bash

    $ Available model(s):
	            [1] timm.vit_base_patch16_224_in21k

+ Next, run the following code block, input and select the additional add-in structure for parameter-efficient transfer:
  ::

    availables   = available_example_models[model]
    config_blitz = availables[select_from_input('add-in structure', availables.keys())]   # user input about add-in structure

  .. code-block:: bash

    $ Available add-in structure(s):
                        [1] LoRA
                        [2] Adapter
                        [3] Convpass
                        [4] None
      Your selection: LoRA

Deploy Training and Test Process
-------------------------

:lamdaorange:`Z`:lamdablue:`h`:lamdablue:`i`:lamdaorange:`J`:lamdablue:`i`:lamdablue:`a`:lamdablue:`n` enables customization of the updated portion using `args.reuse_key`, *such as* assigning `blocks[6:8]` to only tune `model.blocks[6]` to `model.blocks[8]` and their sub-modules.

:lamdaorange:`Z`:lamdablue:`h`:lamdablue:`i`:lamdaorange:`J`:lamdablue:`i`:lamdablue:`a`:lamdablue:`n` also supports diverse training and testing methodologies, including **knowledge distillation** with teacher model supervision, **regularization** under model initialization constraints, **model merging** for comprehensive adaptation settings, and so on.

+ Now, run the following code block, input and select the parameters to fine-tune (the rest are frozen)
  ::

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
    reuse_modules_blitz = availables[select_from_input('reuse modules', availables.keys())] # user input about reuse modules

  .. code-block:: bash

    $ Available reuse modules(s):
                      [1] add-ins and linear layer
                      [2] add-ins and the last block and the linear layer (Partial-1)
                      [3] add-ins and the last two blocks and the linear layer (Partial-2)
                      [4] add-ins and the last four blocks and the linear layer (Partial-4)
      Your selection: add-ins and linear layer

+ Taking *finetune* mode as an example, next, we configure the parameters

  For the rest of the training configuration with more customization options, such as knowledge distillation, regular constraints and model merging, please see `here <TODO>`_
  ::

    training_mode = 'finetune'
    args = get_args(
        dataset=dataset,                # dataset
        dataset_dir=dataset_dir,        # dataset directory
        model=model,                    # backbone network
        config_blitz=config_blitz,      # addin blitz configuration
        training_mode=training_mode,    # training mode
        optimizer='adam',               # optimizer
        lr=1e-2,                        # learning rate
        wd=1e-5,                        # weight decay
        gpu='0',                        # gpu id
        verbose=True                    # control the verbosity of the output
    )
    pprint(vars(args))

  .. code-block:: bash

    $ 2023-07-18 15:17:49.411113: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F AVX512_VNNI FMA
      To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
      2023-07-18 15:17:49.595826: I tensorflow/core/util/port.cc:104] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
      2023-07-18 15:17:50.315077: W tensorflow/compiler/xla/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libnvinfer.so.7'; dlerror: libnvinfer.so.7: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /home/zhangyk/miniconda3/lib
      2023-07-18 15:17:50.315154: W tensorflow/compiler/xla/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libnvinfer_plugin.so.7'; dlerror: libnvinfer_plugin.so.7: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /home/zhangyk/miniconda3/lib
      2023-07-18 15:17:50.315162: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Cannot dlopen some TensorRT libraries. If you would like to use Nvidia GPU with TensorRT, please make sure the missing libraries mentioned above are installed properly.
      {'aa': None,
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
      'warmup_epochs': 5,
      'warmup_lr': 1e-05,
      'warmup_prefix': False,
      'wd': 5e-05,
      'weight_decay': 2e-05,
      'worker_seeding': 'all'}

+ Next, run the following code block to configure the GPU:
  ::

    assert torch.cuda.is_available()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.cuda.set_device(int(args.gpu))

+ Run the following to get the pre-trained model, which includes the additional add-in modules that have been accessed:
  ::

    model, model_args, device = get_model(args)

+ Run the following to get the `dataloader`:
  ::
    
    train_loader, val_loader, num_classes = prepare_vision_dataloader(args, model_args)

  .. code-block:: bash

    $ Log level set to: INFO
      Log files are recorded in: your/log/directory/0718-15-17-52-580
      Trainable/total parameters of the model: 0.37M / 86.17M (0.43148%)

+ Run the following to prepare the optimizer, learning rate scheduler and loss function

  For more customization options, please see TODO
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

+ Run the following to initialize the `trainer`, ready to **start training**:
  ::

    trainer = prepare_trainer(
        args,
        model=model, model_args=model_args, device=device,
        train_loader=train_loader, val_loader=val_loader, num_classes=num_classes,
        optimizer=optimizer, lr_scheduler=lr_scheduler, criterion=criterion
    )

+ Run the following to train and test with :lamdaorange:`Z`:lamdablue:`h`:lamdablue:`i`:lamdaorange:`J`:lamdablue:`i`:lamdablue:`a`:lamdablue:`n`:
  ::

    trainer.fit()
    trainer.test()

  .. code-block:: bash

     
    $       Epoch   GPU Mem.       Time       Loss         LR
              1/5      7.16G     0.3105      4.629      0.001: 100%|██████████| 16.0/16.0 [00:04<00:00, 3.66batch/s]

            Epoch   GPU Mem.       Time      Acc@1      Acc@5
              1/5      7.16G     0.1188      3.334      14.02: 100%|██████████| 157/157 [00:18<00:00, 8.35batch/s] 
      ***   Best results: [Acc@1: 3.3339968152866244], [Acc@5: 14.022691082802547]

            Epoch   GPU Mem.       Time       Loss         LR
              2/5      7.16G     0.2883      4.255 0.00090451: 100%|██████████| 16.0/16.0 [00:04<00:00, 3.96batch/s]

            Epoch   GPU Mem.       Time      Acc@1      Acc@5
              2/5      7.16G     0.1182       4.22      16.28: 100%|██████████| 157/157 [00:18<00:00, 8.37batch/s] 
      ***   Best results: [Acc@1: 4.219745222929936], [Acc@5: 16.28184713375796]

            Epoch   GPU Mem.       Time       Loss         LR
              3/5      7.16G      0.296      4.026 0.00065451: 100%|██████████| 16.0/16.0 [00:04<00:00, 3.96batch/s]

            Epoch   GPU Mem.       Time      Acc@1      Acc@5
              3/5      7.16G     0.1197      5.255      17.71: 100%|██████████| 157/157 [00:18<00:00, 8.28batch/s] 
      ***   Best results: [Acc@1: 5.254777070063694], [Acc@5: 17.70501592356688]

            Epoch   GPU Mem.       Time       Loss         LR
              4/5      7.16G     0.2983       3.88 0.00034549: 100%|██████████| 16.0/16.0 [00:04<00:00, 3.87batch/s]

            Epoch   GPU Mem.       Time      Acc@1      Acc@5
              4/5      7.16G     0.1189      5.862      19.06: 100%|██████████| 157/157 [00:18<00:00, 8.33batch/s] 
      ***   Best results: [Acc@1: 5.8618630573248405], [Acc@5: 19.058519108280255]

            Epoch   GPU Mem.       Time       Loss         LR
              5/5      7.16G     0.2993      3.811 9.5492e-05: 100%|██████████| 16.0/16.0 [00:04<00:00, 3.90batch/s]

            Epoch   GPU Mem.       Time      Acc@1      Acc@5
              5/5      7.16G      0.119      5.723      19.39: 100%|██████████| 157/157 [00:18<00:00, 8.33batch/s] 
      ***   Best results: [Acc@1: 5.722531847133758], [Acc@5: 19.386942675159236]

            Epoch   GPU Mem.       Time      Acc@1      Acc@5
              1/5      7.16G     0.1192      5.723      19.39: 100%|██████████| 157/157 [00:18<00:00, 8.30batch/s] 
      ***   Best results: [Acc@1: 5.722531847133758], [Acc@5: 19.386942675159236]
      (5.722531847133758, 19.386942675159236)

