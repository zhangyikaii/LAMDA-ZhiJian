.. role:: customcolor1
    :class: customcolor1

.. role:: customcolor2
    :class: customcolor2

.. role:: customcolor3
    :class: customcolor3

.. role:: customcolor4
    :class: customcolor4

.. role:: customcolor5
    :class: customcolor5

.. role:: lamdablue
    :class: lamdablue

.. role:: lamdaorange
    :class: lamdaorange

.. raw:: html

    <style>

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


Welcome to :lamdaorange:`Z`:lamdablue:`h`:lamdablue:`i`:lamdaorange:`J`:lamdablue:`i`:lamdablue:`a`:lamdablue:`n`!
====================
.. toctree::
   :maxdepth: 2
   :caption: Contents:


:lamdaorange:`Z`:lamdablue:`h`:lamdablue:`i`:lamdaorange:`J`:lamdablue:`i`:lamdablue:`a`:lamdablue:`n` (`执简驭繁 <https://baike.baidu.com/item/%E6%89%A7%E7%AE%80%E9%A9%AD%E7%B9%81>`_) is a PyTorch-based lightweight framework for **reusing pre-trained models and transferring them to new datasets**. It offers a unified and flexible solution for popular methods such as *regularization* based, *parameter-efficient transfer learning* based, *feature reuse* based, *knowledge distillation* based, and *model merging* based.
:lamdaorange:`Z`:lamdablue:`h`:lamdablue:`i`:lamdaorange:`J`:lamdablue:`i`:lamdablue:`a`:lamdablue:`n` achieves state-of-the-art model capabilities with a :customcolor5:`R`:customcolor4:`E`:customcolor3:`U`:customcolor2:`S`:customcolor1:`E` workflow. **The provided interface methods include**:

* :customcolor5:`R` **egularization** via Pre-Trained Parameters
    * **L**\ :sup:`2` **penalty / L**\ :sup:`2` **SP**, *Explicit Inductive Bias for Transfer Learning with Convolutional Networks.* In:ICML'18. `[Paper] <https://arxiv.org/pdf/1802.01483.pdf>`_ `[Code] <https://github.com>`_
    * **Spectral Norm**, *Spectral Normalization for Generative Adversarial Networks.* In: ICLR'18. `[Paper] <https://arxiv.org/pdf/1802.05957.pdf>`_ `[Code] <https://github.com>`_
    * **Optimal Transport Based**, *Trading beams for bandwidth: Imaging with randomized beamforming.* In: ICML'19. `[Paper] <https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Learning_Debiased_Representations_via_Conditional_Attribute_Interpolation_CVPR_2023_paper.pdf>`_ `[Code] <https://github.com>`_
    * **BSS**, *Catastrophic Forgetting Meets Negative Transfer:Batch Spectral Shrinkage for Safe Transfer Learning.* In: NeurIPS'19.. `[Paper] <https://proceedings.neurips.cc/paper_files/paper/2019/file/c6bff625bdb0393992c9d4db0c6bbe45-Paper.pdf>`_ `[Code] <https://github.com>`_
    * **DELTA**, *DELTA: DEep Learning Transfer using Feature Map with Attention for Convolutional Networks.* In: ICLR'19. `[Paper] <https://arxiv.org/pdf/1901.09229.pdf>`_ `[Code] <https://github.com>`_
    * **Elastic Weight Consolidation**, *Overcoming catastrophic forgetting in neural networks.* In: PNAS'17. `[Paper] <https://arxiv.org/pdf/1612.00796.pdf>`_ `[Code] <https://github.com>`_
* :customcolor4:`E` **fficient Tuning** with Transferred Addin-like **Parameters**
    * **LoRA**, *LoRA: Low-Rank Adaptation of Large Language Models.* In: ICLR'22. `[Paper] <https://arxiv.org/pdf/2106.09685.pdf>`_ `[Code] <https://github.com>`_
    * **Adapter**, *Parameter-Efficient Transfer Learning for NLP.* In: ICML'19. `[Paper] <https://arxiv.org/pdf/1902.00751.pdf>`_ `[Code] <https://github.com>`_
    * **Visual Prompt Tuning / Prefix**, *Visual Prompt Tuning.* In: ECCV'22. `[Paper] <https://arxiv.org/pdf/2203.12119.pdf>`_ `[Code] <https://github.com>`_
    * **Convpass**, *Convolutional Bypasses Are Better Vision Transformer Adapters.* In: Tech Report 07-2022. `[Paper] <https://arxiv.org/pdf/2207.07039.pdf>`_ `[Code] <https://github.com>`_
    * **Scaling & Shifting**, *Scaling & Shifting Your Features: A New Baseline for Efficient Model Tuning.* In: NeurIPS'22. `[Paper] <https://arxiv.org/pdf/2210.08823.pdf>`_ `[Code] <https://github.com>`_
    * **AdaptFormer**, *AdaptFormer: Adapting Vision Transformers for Scalable Visual Recognition.* In: NeurIPS'22. `[Paper] <https://arxiv.org/pdf/2205.13535.pdf>`_ `[Code] <https://github.com>`_
    * **Fact-Tuning**, *FacT: Factor-Tuning for Lightweight Adaptation on Vision Transformer.* In: AAAI'23. `[Paper] <https://arxiv.org/pdf/2212.03145.pdf>`_ `[Code] <https://github.com>`_
* :customcolor3:`U` **tilization of Pre-Trained Features** and Partial Backbones
    * **Linear Probing**, *Parameter-Efficient Transfer Learning for NLP.* In:  ICML'19. `[Paper] <https://arxiv.org/pdf/1902.00751.pdf>`_ `[Code] <https://github.com>`_
    * **Partial-k**, *How transferable are features in deep neural networks?* In: NeurIPS'14. `[Paper] <https://arxiv.org/pdf/1411.1792.pdf>`_ `[Code] <https://github.com>`_
    * **Nearest Class Mean**, *Parameter-Efficient Transfer Learning for NLP.* In: ICML'19. `[Paper] <https://arxiv.org/pdf/1902.00751.pdf>`_ `[Code] <https://github.com>`_
    * **SimpleShot**, *SimpleShot: Revisiting Nearest-Neighbor Classification for Few-Shot Learning.* In: CVPR'19. `[Paper] <https://arxiv.org/pdf/1911.04623.pdf>`_ `[Code] <https://github.com>`_
    * **BitFit**, *BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models.* In: ACL'22. `[Paper] <https://arxiv.org/pdf/2106.10199.pdf>`_ `[Code] <https://github.com>`_
    * **Diff Pruning**, *Parameter-Efficient Transfer Learning with Diff Pruning.* In:  ACL'21. `[Paper] <https://arxiv.org/pdf/2012.07463.pdf>`_ `[Code] <https://github.com>`_
* :customcolor2:`S` **upervisions** from Pre-Trained Predictions as **Knowledge Distillation**
    * **Vanilla Knowledge Distillation / LwF**, *Learning without Memorizing.* In: CVPR'19. `[Paper] <https://arxiv.org/pdf/1811.08051.pdf>`_ `[Code] <https://github.com>`_
    * **FitNet**, *FitNets: Hints for Thin Deep Nets.* In:  ICLR'15. `[Paper] <https://arxiv.org/pdf/1412.6550.pdf>`_ `[Code] <https://github.com>`_
    * **FSP**, *A Gift from Knowledge Distillation: Fast Optimization, Network Minimization and Transfer Learning.* In: CVPR'17. `[Paper] <https://openaccess.thecvf.com/content_cvpr_2017/papers/Yim_A_Gift_From_CVPR_2017_paper.pdf>`_ `[Code] <https://github.com>`_
    * **NST**, *Like What You Like: Knowledge Distill via Neuron Selectivity Transfer.* In: CVPR'17. `[Paper] <https://arxiv.org/pdf/1707.01219.pdf>`_ `[Code] <https://github.com>`_
    * **RKD**, *Relational Knowledge Distillation.* In: CVPR'19. `[Paper] <https://arxiv.org/pdf/1412.6550.pdf>`_ `[Code] <https://github.com>`_
    * **SPKD**, *Similarity-Preserving Knowledge Distillation.* In: CVPR'19. `[Paper] <https://arxiv.org/pdf/1907.09682.pdf>`_ `[Code] <https://github.com>`_
    * **CRD**, *Contrastive Representation Distillation.* In: ICLR'20. `[Paper] <https://arxiv.org/pdf/1910.10699.pdf>`_ `[Code] <https://github.com>`_
    * **REFILLED**, *Distilling Cross-Task Knowledge via Relationship Matching.* In: CVPR'20. `[Paper] <http://www.lamda.nju.edu.cn/lus/files/CVPR20_ReFilled.pdf>`_ `[Code] <https://github.com>`_
* :customcolor1:`E` **nsembling and Merging** Multiple Models
    * **Model Soup:** *averaging weights of multiple fine-tuned models improves accuracy without increasing inference time.* In: ICML'22. `[Paper] <https://arxiv.org/pdf/2203.05482.pdf>`_ `[Code] <https://github.com>`_
    * **WiSE-FT**, *Robust fine-tuning of zero-shot models.* In: CVPR'22. `[Paper] <https://arxiv.org/pdf/2109.01903.pdf>`_ `[Code] <https://github.com>`_
    * **Head2Toe**, *Head2Toe: Utilizing Intermediate Representations for Better Transfer Learning.* In:ICML'22. `[Paper] <https://arxiv.org/pdf/2201.03529.pdf>`_ `[Code] <https://github.com>`_

💡 :lamdaorange:`Z`:lamdablue:`h`:lamdablue:`i`:lamdaorange:`J`:lamdablue:`i`:lamdablue:`a`:lamdablue:`n` also has the following **highlights**:

+ Support access to any of the **pre-trained model zoo**, including:
    + 🤗 **Hugging Face** series — `PyTorch Image Models (timm) <https://github.com/huggingface/pytorch-image-models>`_, `Transformers <https://github.com/huggingface/transformers>`_, **PyTorch** series — `Torchvision <https://pytorch.org/vision/stable/models.html>`_, and **OpenAI** series — `CLIP <https://github.com/openai/CLIP>`_.
    + Other popular projects, *e.g.*, `vit-pytorch <https://github.com/lucidrains/vit-pytorch>`_ (stars `14k <https://github.com/lucidrains/vit-pytorch/stargazers>`_) and **any custom** architecture.
    + **Large Language Model**, including `baichuan <https://huggingface.co/baichuan-inc/baichuan-7B>`_ (*7B*), `LLaMA <https://github.com/facebookresearch/llama>`_ (*7B/13B*), and `BLOOM <https://huggingface.co/bigscience/bloom>`_ (*560M/1.1B/1.7B/3B/7.1B*).
+ Extremely easy to **get started** and **customize**
    + Get started with a 10 minute blitz. `[Tutorial] <https://colab.research.google.com/github/qubvel/segmentation_models.pytorch/blob/master/examples/binary_segmentation_intro.ipynb>`_ & `[Open In Colab] <https://colab.research.google.com/github/qubvel/segmentation_models.pytorch/blob/master/examples/binary_segmentation_intro.ipynb>`_
    + Customize datasets and pre-trained models with step-by-step instructions `[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qubvel/segmentation_models.pytorch/blob/master/examples/binary_segmentation_intro.ipynb)`
    + Feel free to create a novel approach for reusing pre-trained model `[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qubvel/segmentation_models.pytorch/blob/master/examples/binary_segmentation_intro.ipynb)`
+ **Concise** things do **big**
    + **State-of-the-art** on `VTAB <https://arxiv.org/pdf/1910.04867.pdf>`_ benchmark. `[Full Results] <TODO>`_
    + Only ~TODO lines of the code
    + Support friendly guideline and comprehensive documentation `here <TODO>` (文档tutorial链接)
    + Support incorporating method like building *LEGO* blocks `here <TODO>` (文档tutorial链接)
    + Support any dataset and pre-trained model `here <TODO>` (文档tutorial链接)
    + Support multi-GPU training `here <TODO>` (文档tutorial链接)
    + Support both `TensorBoard <https://www.tensorflow.org/tensorboard>`_ and `W&B <https://wandb.ai/>`_ log tools `here <TODO>` (文档tutorial链接)

🔥 **The Naming of** :lamdaorange:`Z`:lamdablue:`h`:lamdablue:`i`:lamdaorange:`J`:lamdablue:`i`:lamdablue:`a`:lamdablue:`n`: In Chinese "ZhiJian-YuFan" means handling complexity with concise and efficient methods. Given the variations in pre-trained models and the deployment overhead of full parameter fine-tuning, :lamdaorange:`Z`:lamdablue:`h`:lamdablue:`i`:lamdaorange:`J`:lamdablue:`i`:lamdablue:`a`:lamdablue:`n` represents a solution that is easily reusable, maintains high accuracy, and maximizes the potential of pre-trained models. “执简驭繁”的意思是用简洁高效的方法驾驭纷繁复杂的事物。“繁”表示现有预训练模型和复用方法种类多、差异大、部署难，所以取名"执简"的意思是通过该工具包，能轻松地驾驭模型复用方法，易上手、快复用、稳精度，最大限度地唤醒预训练模型的知识。

🕹️ Quick Start
------------

1. An environment with Python 3.7+ from
   `conda <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_,
   `venv <https://docs.python.org/3/library/venv.html>`_, or `virtualenv <https://virtualenv.pypa.io/en/latest/>`_.

2. Install :lamdaorange:`Z`:lamdablue:`h`:lamdablue:`i`:lamdaorange:`J`:lamdablue:`i`:lamdablue:`a`:lamdablue:`n` using pip:

   .. code-block:: bash

      $ pip install zhijian

   For more details please click `installation instructions <TODO/INSTALL.md>`_.

   + [Option] Install with the newest version through GitHub:

     .. code-block:: bash

        $ pip install git+https://github.com/zhangyikaii/lamda-zhijian.git@main --upgrade

3. Open your python console and type:

   .. code-block:: python

      import zhijian
      print(zhijian.__version__)

   If no error occurs, you have successfully installed :lamdaorange:`Z`:lamdablue:`h`:lamdablue:`i`:lamdaorange:`J`:lamdablue:`i`:lamdablue:`a`:lamdablue:`n`.


📚 Documentation
------------

The tutorials and API documentation are hosted on `zhijian.readthedocs.io <https://zhijian.readthedocs.io/>`_

中文文档位于 `zhijian.readthedocs.io/zh <https://zhijian.readthedocs.io/zh/master/>`_

Why :lamdaorange:`Z`:lamdablue:`h`:lamdablue:`i`:lamdaorange:`J`:lamdablue:`i`:lamdablue:`a`:lamdablue:`n`?
------------

🎯 Concise things do big
~~~~~~~~~~~~~


.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/get_started
   tutorials/customize_pre_trained_models
   tutorials/customize_dataloader

.. toctree::
   :maxdepth: 2
   :caption: API Docs

   api/zhijian.data


.. toctree::
   :maxdepth: 2
   :caption: Community

   contributing
   contributors


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
