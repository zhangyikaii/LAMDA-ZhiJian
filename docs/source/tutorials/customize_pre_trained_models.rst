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

Customize Pre-trained Models
============================

.. raw:: html

   <span style="font-size: 25px;">🛠️</span>
   <p></p>

This example shows how to **customize your own pre-trained model** for new ideas. Tailor and integrate *any* **add-in** extra module within the vast pre-trained model **with lightning speed**.


.. figure:: ../_static/images/tutorials_overview.png
   :align: center

Introduce the Custom Model
-------------------------

Let's begin with a three-layer Multilayer Perceptron (MLP).

Although a multi-layer perceptron is not a good image learner, we can quickly get started with it. For other custom networks, we can also make similar designs and modifications by analogy. 

+ Run the code block below to customize the model:

.. code-block:: python

   import torch.nn as nn
   
   class MLP(nn.Module):
       """
       MLP Class
       ==============

       Multilayer Perceptron (MLP) model for image (224x224) classification tasks.

       Args:
           args (object): Custom arguments or configurations.
           num_classes (int): Number of output classes.
       """
       def __init__(self, args, num_classes):
           super(MLP, self).__init__()
           self.args = args
           self.image_size = 224
           self.fc1 = nn.Linear(self.image_size * self.image_size * 3, 256)
           self.fc2 = nn.Linear(256, 256)
           self.fc3 = nn.Linear(256, num_classes)

       def forward(self, x):
           """
           Forward pass of the model.

           Args:
               x (torch.Tensor): Input tensor.

           Returns:
               torch.Tensor: Output logits.
           """
           x = x.view(x.size(0), -1)
           x = self.fc1(x)
           x = nn.ReLU()(x)
           x = self.fc2(x)
           x = nn.ReLU()(x)
           x = self.fc3(x)
           return x


.. figure:: ../_static/images/tutorials_mlp.png
   :align: center

   Custom Multilayer Perceptron (MLP) Architecture

Now, expand models from **fleeting moments of inspiration**.

We will customize and modify the network structure through a few lines of code from :lamdaorange:`Z`:lamdablue:`h`:lamdablue:`i`:lamdaorange:`J`:lamdablue:`i`:lamdablue:`a`:lamdablue:`n`.

The additional auxiliary structures are also implemented based on the PyTorch framework. The auxiliary structures inherit the base class `AddinBase`, which integrates some basic methods for data access.

Design Additional Add-in Modules
-------------------------
+ Run the code block below to customize add-in modules and entry points for the model.

.. code-block:: python

   class MLPAddin(AddinBase):
       """
       MLPAddin Class
       ==============

       Multilayer Perceptron (MLP) add-in.

       Args:
           config (object): Custom configuration or arguments.
           model_config (object): Configuration specific to the model.
       """
       def __init__(self, config, model_config):
           super(MLPAddin, self).__init__()

           self.config = config
           self.embed_dim = model_config.hidden_size

           self.reduction_dim = 16

           self.fc1 = nn.Linear(self.embed_dim, self.reduction_dim)
           if config.mlp_addin_output_size is not None:
               self.fc2 = nn.Linear(self.reduction_dim, config.mlp_addin_output_size)
           else:
               self.fc2 = nn.Linear(self.reduction_dim, self.embed_dim)

       def forward(self, x):
           """
           Forward pass of the MLP add-in.

           Args:
               x (tensor): Input tensor.

           Returns:
               tensor: Output tensor after passing through the MLP add-in.
           """
           identity = x 
           out = self.fc1(identity)
           out = nn.ReLU()(out)
           out = self.fc2(out)

           return out

       def adapt_input(self, module, inputs):
           """
           Hook function to adapt the input data before it enters the module.

           Args:
               module (nn.Module): The module being hooked.
               inputs (tuple): (Inputs before the module,).

           Returns:
               tensor: Adapted input tensor after passing through the MLP add-in.
           """
           x = inputs[0]
           return self.forward(x)

       def adapt_output(self, module, inputs, outputs):
           """
           Hook function to adapt the output data after it leaves the module.

           Args:
               module (nn.Module): The module being hooked.
               inputs (tuple): (Inputs before the module,).
               outputs (tensor): Outputs after the module.

           Returns:
               tensor: Adapted output tensor after passing through the MLP add-in.
           """
           return self.forward(outputs)

       def adapt_across_input(self, module, inputs):
           """
           Hook function to adapt the data across the modules.

           Args:
               module (nn.Module): The module being hooked.
               inputs (tuple): (Inputs before the module,).

           Returns:
               tensor: Adapted input tensor after adding the MLP add-in output to the subsequent module.
           """
           x = inputs[0]
           x = x + self.forward(self.inputs_cache)
           return x

       def adapt_across_output(self, module, inputs, outputs):
           """
           Hook function to adapt the data across the modules.

           Args:
               module (nn.Module): The module being hooked.
               inputs (tuple): (Inputs before the module,).
               outputs (tensor): Outputs after the module.

           Returns:
               tensor: Adapted input tensor after adding the MLP add-in output to the previous module.
           """
           outputs = outputs + self.forward(self.inputs_cache)
           return outputs

In the extended auxiliary structure `MLPAddin` mentioned above, we add a low-rank bottleneck (consisting of two linear layers, with a reduced dimension in the middle) inspired by efficient parameter methods like *Adapter* or *LoRA*. We define and implement this in the `__init__` and `forward` functions.


.. figure:: ../_static/images/tutorials_addin_structure.png
   :align: center

   Additional Auxiliary Structure Example


As shown above, the `hook` methods starting with `adapt_` are our entry functions. They serve as hooks to attach the extended modules to the base model. We will further explain their roles in the following text.


Deploy the Inter-layer Insertion & Cross-layer Concatenation Points
-------------------------

We aim to customize our model by **inter-layer insertion** and **cross-layer concatenation** of the auxiliary structures at different positions within the base model (such as the custom MLP mentioned earlier). When configuring the insertion or concatenation positions, :lamdaorange:`Z`:lamdablue:`h`:lamdablue:`i`:lamdaorange:`J`:lamdablue:`i`:lamdablue:`a`:lamdablue:`n` provides **a minimalistic one-line configuration syntax**.

The syntax for configuring add-in module into the base model is as follows. We will start with one or two examples and gradually understand the meaning of each configuration part.

+ *Inter-layer Insertion*:

  ::
    
    >>> (MLPAddin.adapt_input): ...->{inout1}(fc2)->...

  .. figure:: ../_static/images/tutorials_mlp_addin_1.png
    :align: center
    :name: tutorials-mlp-addin-1
    
    Additional Add-in Structure - Inter-layer Insertion 1


  ::
    
    >>> (MLPAddin.adapt_input): ...->(fc2){inout1}->...


  .. figure:: ../_static/images/tutorials_mlp_addin_2.png
    :align: center
    :name: tutorials-mlp-addin-2

    Additional Add-in Structure - Inter-layer Insertion 2

+ *Cross-layer Concatenation*:

  ::
    
    >>> (MLPAddin.adapt_across_input): ...->(fc1){in1}->...->{out1}(fc3)->...


  .. figure:: ../_static/images/tutorials_mlp_addin_3.png
    :align: center
    :name: tutorials-mlp-addin-3

    Additional Add-in Structure - Cross-layer Concatenation


Base Module: :code:`->(fc1)`
^^^^^^^^^^^^^^^^^^^^^^^^^

Consider a base model implemented based on the PyTorch framework, where the representation of each layer and module in the model is straightforward：

+ As shown in the figure, the print command can output the defined names of the model structure:

  ::
    
    print(model)

+ The structure of some classic backbone can be represented as follows

  + MLP:
    ::
        
        >>> input->(fc1)->(fc2)->(fc3)->output
  + ViT :code:`block[i]``:
    ::
        
        >>> input->...->(block[i].norm1)->
              (block[i].attn.qkv)->(block[i].attn.attn_drop)->(block[i].attn.proj)->(block[i].attn.proj_drop)->
                (block[i].ls1)->(block[i].drop_path1)->
                  (block[i].norm2)->
                    (block[i].mlp.fc1)->(block[i].mlp.act)->(block[i].mlp.drop1)->(block[i].mlp.fc2)->(block[i].mlp.drop2)->
                      (block[i].ls2)->(block[i].drop_path2)->...->output

Default Module: :code:`...`
^^^^^^^^^^^^^^^^^^^^^^^^^

In the configuration syntax of :lamdaorange:`Z`:lamdablue:`h`:lamdablue:`i`:lamdaorange:`J`:lamdablue:`i`:lamdablue:`a`:lamdablue:`n`, the :code:`...` can be used to represent the default layer or module.

+ For example, when we only focus on the :code:`(fc2)` module in MLP and the :code:`(block[i].mlp.fc2)` module in ViT:

  + MLP:
    ::
        
        >>> ...->(fc2)->...
  + ViT:
    ::
        
        >>> ...->(block[i].mlp.fc2)->...


Insertion & Concatenation Function: :code:`():`
^^^^^^^^^^^^^^^^^^^^^^^^^

Considering the custom auxiliary structure :code:`MLPAddin` mentioned above, the functions starting with :code:`adapt_` will serve as the processing center that **insert** and **concatenate** into the base model.

+ There are primarily two types of parameter passing methods:

  ::

    def adapt_input(self, module, inputs):
        """
        Args:
            module (nn.Module): The module being hooked.
            inputs (tuple): (Inputs before the module,).
        """
        ...

    def adapt_output(self, module, inputs, outputs):
        """
        Args:
            module (nn.Module): The module being hooked.
            inputs (tuple): (Inputs before the module,).
            outputs (tensor): Outputs after the module.
        """
        ...

  where

  + :code:`adapt_input(self, module, inputs)` is generally set before the module and is called before the data enters the module to process inputs and truncate the :code:`input`.

  + :code:`adapt_output(self, module, inputs, outputs)` is generally set before the module and is called before the data enters the module to process outputs and truncate the :code:`output`.

These functions will be "hooked" into the base model in the main method of configuring the module, serving as key connectors between the base model and the auxiliary structure.

Insertion & Concatenation Point: :code:`{}`
^^^^^^^^^^^^^^^^^^^^^^^^^

Consider an independent extended auxiliary structure (such as the :code:`MLPAddin` mentioned above), its **insertion or concatenation points** with the base network must consist of *"Data Input"* and *"Data Output"* where:

+ **"Data Input"** refers to the network features input into the extended auxiliary structure.
+ **"Data Output"** refers to the adapted features output from the auxiliary structure back to the base network.


Next, let's use some configuration examples of MLP to illustrate the syntax and functionality of :lamdaorange:`Z`:lamdablue:`h`:lamdablue:`i`:lamdaorange:`J`:lamdablue:`i`:lamdablue:`a`:lamdablue:`n` for **module integration**:

Inter-layer Insertion: :code:`inout`
^^^^^^^^^^^^^^^^^^^^^^^^^

+ As shown in the above :numref:`tutorials-mlp-addin-1`, the configuration expression is:

  ::

    >>> (MLPAddin.adapt_input): ...->{inout1}(fc2)->...


  where

  + :code:`{inout1}` refers to the position which gets the base model features (or output, at any layer or module).
  
    It denotes the *"Data Input"* and *"Data Output"*. The configuration can be :code:`{inoutx}`, where :code:`x` represents the x\ :sup:`th` integration point. For example, :code:`{inout1}` represents the first integration point.

  + In the example above, this inter-layer insertion configuration *truncates* the features of the input :code:`fc2` module, *passes* them through, and then return to the :code:`fc2` module. At this point, the original :code:`fc2` features no longer enter.

Cross-layer Concatenation :code:`in`, :code:`out`
^^^^^^^^^^^^^^^^^^^^^^^^^

+ As shown in the above :numref:`tutorials-mlp-addin-3`, the configuration expression is:

  ::

    >>> (MLPAddin.adapt_across_input): ...->(fc1){in1}->...->{out1}(fc3)->...`

  where

  + :code:`{in1}`: represents the integration point where the base network features (or output, at any layer or module) *enter* the additional add-in structure.
  
    It denotes the *"Data Input"*. The configuration can be :code:`{inx}`, where :code:`x` represents the x\ :sup:`th` integration point. For example, :code:`{in1}` represents the first integration point.

  + :code:`{out1}`: represent the integration points where the features processed by the additional add-in structure are *returned* to the base network.

    It denotes the *"Data Output"*. The configuration can be :code:`{outx}`, where :code:`x` represents the x\ :sup:`th` integration point. For example, :code:`{out1}` represents the first integration point.
    
  + This cross-layer concatenation configuration *extracts* the features of the :code:`fc1` module's output, *passes them into* the auxiliary structure, and then *returns* them to the base network before the :code:`fc3` module in the form of residual addition.

+ For a better prompt, let's create a tool function that guides the input first:

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

    available_example_config_blitzs = {
        'Insert between `fc1` and `fc2` layer (performed before `fc2`)': "(MLPAddin.adapt_input): ...->{inout1}(fc2)->...",
        'Insert between `fc1` and `fc2` layer (performed after `fc1`)': "(MLPAddin.adapt_output): ...->(fc1){inout1}->...",
        'Splice across `fc2` layer (performed before `fc2` and `fc3`)': "(MLPAddin.adapt_across_input): ...->{in1}(fc2)->{out1}(fc3)->...",
        'Splice across `fc2` layer (performed after `fc1` and before `fc3`)': "(MLPAddin.adapt_across_input): ...->(fc1){in1}->...->{in2}(fc3)->...",
        'Splice across `fc2` layer (performed before and after `fc2`)': "(MLPAddin.adapt_across_output): ...->{in1}(fc2){in2}->...",
        'Splice across `fc2` layer (performed after `fc1` and `fc2`)': "(MLPAddin.adapt_across_output): ...->(fc1){in1}->(fc2){in2}->...",
    }

    config_blitz = select_from_input('add-in structure', available_example_config_blitzs.keys()) # user input about model

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
      Your selection: VTAB-1k.CIFAR-100
      Your dataset directory: /data/zhangyk/data/zhijian
+ Next, we will configure the parameters and proceed with model training and testing:

  ::

   args = get_args(
       model='timm.vit_base_patch16_224_in21k',    # backbone network
       config_blitz=config_blitz,                  # addin blitz configuration
       dataset='VTAB.cifar',                       # dataset
       dataset_dir='your/dataset/directory',       # dataset directory
       training_mode='finetune',                   # training mode
       optimizer='adam',                           # optimizer
       lr=1e-2,                                    # learning rate
       wd=1e-5,                                    # weight decay
       verbose=True                                # control the verbosity of the output
   )
   pprint(vars(args))

  .. code-block:: bash

    $ {'aa': None,
       'addins': [{'hook': [['get_pre', 'pre'], ['adapt_across_output', 'post']],
                   'location': [['fc2'], ['fc2']],
                   'name': 'MLPAddin'}],
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


+ Run the code block below to configure the GPU and the model (excluding additional auxiliary structures):

  ::

   assert torch.cuda.is_available()
   os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
   torch.cuda.set_device(int(args.gpu))

   model = MLP(args, DATASET2NUM_CLASSES[args.dataset.replace('VTAB.','')])
   model = ModelWrapper(model)
   model_args = dict2args({'hidden_size': 512})

+ Run the code block below to configure additional auxiliary structures:

  ::

   args.mlp_addin_output_size = 256
   addins, fixed_params = prepare_addins(args, model_args, addin_classes=[MLPAddin])

   prepare_hook(args.addins, addins, model, 'addin')
   prepare_gradient(args.reuse_keys, model)
   device = prepare_cuda(model)

+ Run the code block below to configure the dataset, optimizer, loss function, and other settings:

  ::

   train_loader, val_loader, num_classes = prepare_vision_dataloader(args, model_args)

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


+ Run the code block below to prepare the :code:`trainer` object and start training and testing:

  ::

   trainer = prepare_trainer(
       args,
       model=model,
       model_args=model_args,
       device=device,
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
      Log files are recorded in: your/log/directory/0718-19-52-36-748
      Trainable/total parameters of the model: 0.03M / 38.64M (0.08843%)

            Epoch   GPU Mem.       Time       Loss         LR
              1/5     0.589G     0.1355      4.602      0.001: 100%|██████████| 16.0/16.0 [00:01<00:00, 12.9batch/s]

            Epoch   GPU Mem.       Time      Acc@1      Acc@5
              1/5     0.629G    0.03114      1.871      7.932: 100%|██████████| 157/157 [00:05<00:00, 30.9batch/s] 
      ***   Best results: [Acc@1: 1.8710191082802548], [Acc@5: 7.931926751592357]

            Epoch   GPU Mem.       Time       Loss         LR
              2/5     0.784G     0.1016      4.538 0.00090451: 100%|██████████| 16.0/16.0 [00:00<00:00, 19.4batch/s]

            Epoch   GPU Mem.       Time      Acc@1      Acc@5
              2/5     0.784G    0.02669      2.498      9.504: 100%|██████████| 157/157 [00:04<00:00, 35.9batch/s] 
      ***   Best results: [Acc@1: 2.4980095541401273], [Acc@5: 9.504378980891719]

            Epoch   GPU Mem.       Time       Loss         LR
              3/5     0.784G    0.09631      4.488 0.00065451: 100%|██████████| 16.0/16.0 [00:00<00:00, 20.6batch/s]

            Epoch   GPU Mem.       Time      Acc@1      Acc@5
              3/5     0.784G    0.02688      2.379      10.16: 100%|██████████| 157/157 [00:04<00:00, 36.0batch/s] 
      ***   Best results: [Acc@1: 2.3785828025477707], [Acc@5: 10.161226114649681]

            Epoch   GPU Mem.       Time       Loss         LR
              4/5     0.784G    0.09126       4.45 0.00034549: 100%|██████████| 16.0/16.0 [00:00<00:00, 20.2batch/s]

            Epoch   GPU Mem.       Time      Acc@1      Acc@5
              4/5     0.784G    0.02644      2.468      10.29: 100%|██████████| 157/157 [00:04<00:00, 36.2batch/s] 
      ***   Best results: [Acc@1: 2.468152866242038], [Acc@5: 10.290605095541402]

            Epoch   GPU Mem.       Time       Loss         LR
              5/5     0.784G     0.0936      4.431 9.5492e-05: 100%|██████████| 16.0/16.0 [00:00<00:00, 20.5batch/s]

            Epoch   GPU Mem.       Time      Acc@1      Acc@5
              5/5     0.784G    0.02706      2.558      10.43: 100%|██████████| 157/157 [00:04<00:00, 35.8batch/s] 
      ***   Best results: [Acc@1: 2.557722929936306], [Acc@5: 10.429936305732484]

            Epoch   GPU Mem.       Time      Acc@1      Acc@5
              1/5     0.784G    0.02667      2.558      10.43: 100%|██████████| 157/157 [00:04<00:00, 36.0batch/s] 
      ***   Best results: [Acc@1: 2.557722929936306], [Acc@5: 10.429936305732484]
