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
    .customcolor1 {
        color: #f48702;
        font-weight: bold;
    }
    .customcolor2 {
        color: #f64600;
        font-weight: bold;
    }
    .customcolor3 {
        color: #de1500;
        font-weight: bold;
    }
    .customcolor4 {
        color: #b70501;
        font-weight: bold;
    }
    .customcolor5 {
        color: #d6005c;
        font-weight: bold;
    }
    .lamdablue {
        color: #47479e;
        font-weight: bold;
    }
    .lamdaorange {
        color: #fd4d01;
        font-weight: bold;
    }

    </style>

Customize Pre-trained Model
============================

.. raw:: html

   <span style="font-size: 25px;">🛠️</span>
   <p></p>


:lamdaorange:`Z`:lamdablue:`h`:lamdablue:`i`:lamdaorange:`J`:lamdablue:`i`:lamdablue:`a`:lamdablue:`n` is an **unifying** and **rapidly deployable** toolbox for **pre-trained model reuse**.


Overview
-------------------------

:customcolor5:`In` :customcolor4:`the` :customcolor2:`following` :customcolor1:`example`, we show how to customize **your own** pre-trained model with **a new target structure** in :lamdaorange:`Z`:lamdablue:`h`:lamdablue:`i`:lamdaorange:`J`:lamdablue:`i`:lamdablue:`a`:lamdablue:`n`.

**Feel free** to deploy model reusability technology on *any* pre-trained model, with loading in the conventional `PyTorch` style.


Construct Custom Model
-------------------------

Let's begin with a three-layer Multilayer Perceptron (MLP).


.. figure:: ../_static/images/tutorials_mlp.png
   :align: center

   Custom Multilayer Perceptron (MLP) Architecture


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


+ Next, run the code block below to configure the GPU and the model:

  ::

   model = MLP(args, DATASET2NUM_CLASSES[args.dataset.replace('VTAB.','')])
   model = ModelWrapper(model)
   model_args = dict2args({'hidden_size': 512})


+ Now, run the code block below to prepare the :code:`trainer` with passing in the parameter :code:`model`:

  ::

   trainer = prepare_trainer(
       args,
       model=model,
       model_args=model_args,
       device=device,
       ...
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
