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


.. figure:: https://github.com/zhangyikaii/LAMDA-ZhiJian/blob/main/docs/source/_static/images/overview.png?raw=true
   :align: center

:lamdaorange:`Z`:lamdablue:`h`:lamdablue:`i`:lamdaorange:`J`:lamdablue:`i`:lamdablue:`a`:lamdablue:`n` (`执简驭繁 <https://baike.baidu.com/item/%E6%89%A7%E7%AE%80%E9%A9%AD%E7%B9%81>`_) is a *comprehensive* and *user-friendly* :code:`PyTorch`-based **toolbox** for leveraging **foundation pre-trained models** and their **fine-tuned counterparts** to *extract* knowledge and *expedite* learning in real-world tasks, *i.e.*, **serving the** :customcolor1:`M`:customcolor1:`o`:customcolor2:`d`:customcolor2:`e`:customcolor3:`l` :customcolor3:`R`:customcolor4:`e`:customcolor4:`u`:customcolor5:`s`:customcolor5:`e` **tasks**.

**The rapid progress** in deep learning has led to the emergence of **numerous open-source Pre-Trained Models (PTMs)** on platforms like PyTorch, TensorFlow, and HuggingFace Transformers. Leveraging these PTMs for specific tasks empowers them to handle objectives effectively, creating valuable resources for the machine-learning community. **Reusing PTMs is vital in enhancing target models' capabilities and efficiency**, achieved through adapting the architecture, customizing learning on target data, or devising optimized inference strategies to leverage PTM knowledge. **To facilitate a holistic consideration of various model reuse strategies**, :lamdaorange:`Z`:lamdablue:`h`:lamdablue:`i`:lamdaorange:`J`:lamdablue:`i`:lamdablue:`a`:lamdablue:`n` categorizes model reuse methods into *three* sequential modules: :customcolor2:`Architect`, :customcolor3:`Tuner`, and :customcolor5:`Merger`, aligning with the stages of **model preparation**, **model learning**, and **model inference** on the target task, respectively. **The provided interface methods include**:

* :customcolor2:`A` **rchitect Module**

    The Architect module involves **modifying the pre-trained model to fit the target task**, and reusing certain parts of the pre-trained model while introducing new learnable parameters with specialized structures.

    * **Linear Probing** & **Partial-k**, *How transferable are features in deep neural networks?* In: NeurIPS'14. `[Paper] <https://arxiv.org/pdf/1411.1792.pdf>`_
    * **Adapter**, *Parameter-Efficient Transfer Learning for NLP.* In: ICML'19. `[Paper] <https://arxiv.org/pdf/1902.00751.pdf>`_
    * **Diff Pruning**, *Parameter-Efficient Transfer Learning with Diff Pruning.* In: ACL'21. `[Paper] <https://arxiv.org/pdf/2012.07463.pdf>`_
    * **LoRA**, *LoRA: Low-Rank Adaptation of Large Language Models.* In: ICLR'22. `[Paper] <https://arxiv.org/pdf/2106.09685.pdf>`_
    * **Visual Prompt Tuning / Prefix**, *Visual Prompt Tuning.* In: ECCV'22. `[Paper] <https://arxiv.org/pdf/2203.12119.pdf>`_
    * **Head2Toe**, *Head2Toe: Utilizing Intermediate Representations for Better Transfer Learning.* In:ICML'22. `[Paper] <https://arxiv.org/pdf/2201.03529.pdf>`_
    * **Scaling & Shifting**, *Scaling & Shifting Your Features: A New Baseline for Efficient Model Tuning.* In: NeurIPS'22. `[Paper] <https://arxiv.org/pdf/2210.08823.pdf>`_
    * **AdaptFormer**, *AdaptFormer: Adapting Vision Transformers for Scalable Visual Recognition.* In: NeurIPS'22. `[Paper] <https://arxiv.org/pdf/2205.13535.pdf>`_
    * **BitFit**, *BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models.* In: ACL'22. `[Paper] <https://arxiv.org/pdf/2106.10199.pdf>`_
    * **Convpass**, *Convolutional Bypasses Are Better Vision Transformer Adapters.* In: Tech Report 07-2022. `[Paper] <https://arxiv.org/pdf/2207.07039.pdf>`_
    * **Fact-Tuning**, *FacT: Factor-Tuning for Lightweight Adaptation on Vision Transformer.* In: AAAI'23. `[Paper] <https://arxiv.org/pdf/2212.03145.pdf>`_
    * **VQT**, *Visual Query Tuning: Towards Effective Usage of Intermediate Representations for Parameter and Memory Efficient Transfer Learning.* In: CVPR'23. `[Paper] <https://arxiv.org/pdf/2212.03220.pdf>`_

* :customcolor3:`T` **uner Module**

    The Tuner module focuses on **training the target model with guidance from pre-trained model knowledge** to expedite the optimization process, *e.g.*, via adjusting objectives, optimizers, or regularizers.

    * **Knowledge Transfer and Matching**, *NeC4.5: neural ensemble based C4.5.* In: IEEE Trans. Knowl. Data Eng. 2004. `[Paper] <https://ieeexplore.ieee.org/document/1294896>`_
    * **FitNet**, *FitNets: Hints for Thin Deep Nets.* In: ICLR'15. `[Paper] <https://arxiv.org/pdf/1412.6550.pdf>`_
    * **LwF**, *Learning without Forgetting.* In: ECCV'16. `[Paper] <https://arxiv.org/pdf/1606.09282.pdf>`_
    * **FSP**, *A Gift from Knowledge Distillation: Fast Optimization, Network Minimization and Transfer Learning.* In: CVPR'17. `[Paper] <https://openaccess.thecvf.com/content_cvpr_2017/papers/Yim_A_Gift_From_CVPR_2017_paper.pdf>`_
    * **NST**, *Like What You Like: Knowledge Distill via Neuron Selectivity Transfer.* In: CVPR'17. `[Paper] <https://arxiv.org/pdf/1707.01219.pdf>`_
    * **RKD**, *Relational Knowledge Distillation.* In: CVPR'19. `[Paper] <https://arxiv.org/pdf/1412.6550.pdf>`_
    * **SPKD**, *Similarity-Preserving Knowledge Distillation.* In: CVPR'19. `[Paper] <https://arxiv.org/pdf/1907.09682.pdf>`_
    * **CRD**, *Contrastive Representation Distillation.* In: ICLR'20. `[Paper] <https://arxiv.org/pdf/1910.10699.pdf>`_
    * **REFILLED**, *Distilling Cross-Task Knowledge via Relationship Matching.* In: CVPR'20. `[Paper] <http://www.lamda.nju.edu.cn/lus/files/CVPR20_ReFilled.pdf>`_
    * **WiSE-FT**, *Robust fine-tuning of zero-shot models.* In: CVPR'22. `[Paper] <https://arxiv.org/pdf/2109.01903.pdf>`_
    * **L**\ :sup:`2` **penalty / L**\ :sup:`2` **SP**, *Explicit Inductive Bias for Transfer Learning with Convolutional Networks.* In:ICML'18. `[Paper] <https://arxiv.org/pdf/1802.01483.pdf>`_
    * **Spectral Norm**, *Spectral Normalization for Generative Adversarial Networks.* In: ICLR'18. `[Paper] <https://arxiv.org/pdf/1802.05957.pdf>`_
    * **BSS**, *Catastrophic Forgetting Meets Negative Transfer:Batch Spectral Shrinkage for Safe Transfer Learning.* In: NeurIPS'19.. `[Paper] <https://proceedings.neurips.cc/paper_files/paper/2019/file/c6bff625bdb0393992c9d4db0c6bbe45-Paper.pdf>`_
    * **DELTA**, *DELTA: DEep Learning Transfer using Feature Map with Attention for Convolutional Networks.* In: ICLR'19. `[Paper] <https://arxiv.org/pdf/1901.09229.pdf>`_
    * **DeiT**, *Training data-efficient image transformers & distillation through attention.* In ICML'21. `[Paper] <https://arxiv.org/pdf/2012.12877.pdf>`_
    * **DIST**, *Knowledge Distillation from A Stronger Teacher.* In: NeurIPS'22. `[Paper] <https://arxiv.org/pdf/2205.10536.pdf>`_

* :customcolor3:`M` **erger Module**

    The Merger module **influences the inference phase** by either reusing pre-trained features or incorporating adapted logits from the pre-trained model.

    * **Logits Ensemble**, *Ensemble Methods: Foundations and Algorithms.* 2012. `[Book] <https://www.routledge.com/Ensemble-Methods-Foundations-and-Algorithms/Zhou/p/book/9781439830031>`_
    * **Nearest Class Mean**, *Distance-Based Image Classification: Generalizing to New Classes at Near-Zero Cost.* In: IEEE Trans. Pattern Anal. Mach. Intell. 2013. `[Paper] <https://ieeexplore.ieee.org/document/6517188>`_
    * **SimpleShot**, *SimpleShot: Revisiting Nearest-Neighbor Classification for Few-Shot Learning.* In: CVPR'19. `[Paper] <https://arxiv.org/pdf/1911.04623.pdf>`_
    * **via Optimal Transport**, *Model Fusion via Optimal Transport.* In: NeurIPS'20. `[Paper] <https://arxiv.org/pdf/1910.05653.pdf>`_
    * **Model Soup**, *Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time.* In: ICML'22. `[Paper] <https://arxiv.org/pdf/2203.05482.pdf>`_
    * **Fisher Merging**, *Merging Models with Fisher-Weighted Averaging.* In: NeurIPS'22. `[Paper] <https://arxiv.org/pdf/2111.09832.pdf>`_
    * **Deep Model Reassembly**, *Deep Model Reassembly.* In: NeurIPS'22. `[Paper] <https://arxiv.org/pdf/2210.17409.pdf>`_
    * **REPAIR**, *REPAIR: REnormalizing Permuted Activations for Interpolation Repair.* In: ICLR'23. `[Paper] <https://arxiv.org/pdf/2211.08403.pdf>`_
    * **Git Re-Basin**, *Git Re-Basin: Merging Models modulo Permutation Symmetries.* In: ICLR'23. `[Paper] <https://arxiv.org/pdf/2209.04836.pdf>`_
    * **ZipIt**, *ZipIt! Merging Models from Different Tasks without Training.* `[Paper] <https://arxiv.org/pdf/2305.03053.pdf>`_

💡 :lamdaorange:`Z`:lamdablue:`h`:lamdablue:`i`:lamdaorange:`J`:lamdablue:`i`:lamdablue:`a`:lamdablue:`n` **also has the following** :customcolor1:`hi`:customcolor2:`gh`:customcolor3:`li`:customcolor4:`gh`:customcolor5:`ts`:

+ **Support** reuse of various **pre-trained model zoo**, including:

  +  PyTorch `Torchvision <https://pytorch.org/vision/stable/models.html>`_; OpenAI `CLIP <https://github.com/openai/CLIP>`_; 🤗Hugging Face `PyTorch Image Models (timm) <https://github.com/huggingface/pytorch-image-models>`_, `Transformers <https://github.com/huggingface/transformers>`_
  + Other popular projects, *e.g.*, `vit-pytorch <https://github.com/lucidrains/vit-pytorch>`_ (stars `14k <https://github.com/lucidrains/vit-pytorch/stargazers>`_)
  + Large Language Model, including `baichuan <https://huggingface.co/baichuan-inc/baichuan-7B>`_, `LLaMA <https://github.com/facebookresearch/llama>`_, and `BLOOM <https://huggingface.co/bigscience/bloom>`_

+ **Extremely easy** to get started and **customize**

  + Get started with a 10 minute blitz `[Open In Colab] <TODO>`_
  + Customize datasets and pre-trained models with step-by-step instructions `[Open In Colab] <TODO>`_
  + Feel free to create a novel approach for reusing pre-trained model `[Open In Colab] <TODO>`_

+ **Concise** things do **big**

  + Only ~5000 lines of the base code, with incorporating method like building *LEGO* blocks
  + **State-of-the-art** results on `VTAB benchmark <https://google-research.github.io/task_adaptation/>`_ with approximately **10k** experiments `[here] <https://github.com/zhangyikaii/LAMDA-ZhiJian/tree/main/results>`_
  + Support friendly guideline and comprehensive documentation to custom dataset and pre-trained model `[here] <TODO>`_

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

+-----------------------------------------------------------------------------------------------------------------------------------------------------------------+-------+-----------+-------------+--------------+--------------------+----------+-------+ 
| Related Library                                                                                                                                                 | Stars | # of Alg. | # of Model  | # of Dataset | # of Fields        | LLM Supp.| Docs. | 
+=================================================================================================================================================================+=======+===========+=============+==============+====================+==========+=======+ 
| `PEFT <https://github.com/huggingface/peft>`_                                                                                                                   |  8k+  | 6         | ~15         | --:sup:`(3)` | 1 :sup:`(a)`       |     ✔️   |   ✔️  | 
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------+-------+-----------+-------------+--------------+--------------------+----------+-------+ 
| `adapter-transformers <https://github.com/adapter-hub/adapter-transformers>`_                                                                                   |  1k+  | 10        | ~15         | --:sup:`(3)` | 1 :sup:`(a)`       |     ❌   |   ✔️  | 
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------+-------+-----------+-------------+--------------+--------------------+----------+-------+ 
| `LLaMA-Efficient-Tuning <https://github.com/hiyouga/LLaMA-Efficient-Tuning>`_                                                                                   |  2k+  | 4         | 5           | ~20          | 1 :sup:`(a)`       |     ✔️   |   ❌  | 
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------+-------+-----------+-------------+--------------+--------------------+----------+-------+ 
| `Knowledge-Distillation-Zoo <https://github.com/AberHu/Knowledge-Distillation-Zoo>`_                                                                            |  1k+  | 20        | 2           | 2            | 1 :sup:`(b)`       |     ❌   |   ❌  | 
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------+-------+-----------+-------------+--------------+--------------------+----------+-------+ 
| `Easy Few-Shot Learning <https://github.com/sicara/easy-few-shot-learning>`_                                                                                    |  608  | 10        | 3           | 2            | 1 :sup:`(c)`       |     ❌   |   ❌  | 
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------+-------+-----------+-------------+--------------+--------------------+----------+-------+ 
| `Model Soups <https://github.com/mlfoundations/model-soups>`_                                                                                                   |  255  | 3         | 3           | 5            | 1 :sup:`(d)`       |     ❌   |   ❌  | 
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------+-------+-----------+-------------+--------------+--------------------+----------+-------+ 
| `Git Re-Basin <https://github.com/samuela/git-re-basin>`_                                                                                                       |  410  | 3         | 5           | 4            | 1 :sup:`(d)`       |     ❌   |   ❌  | 
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------+-------+-----------+-------------+--------------+--------------------+----------+-------+
| :lamdaorange:`Z`:lamdablue:`h`:lamdablue:`i`:lamdaorange:`J`:lamdablue:`i`:lamdablue:`a`:lamdablue:`n` `(Ours) <https://github.com/zhangyikaii/LAMDA-ZhiJian>`_ |  ing  | 30+       | ~50         | 19           | 1 :sup:`(a,b,c,d)` |     ✔️   |   ✔️  | 
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------+-------+-----------+-------------+--------------+--------------------+----------+-------+ 


.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/get_started
   tutorials/config_with_one_line_blitz
   tutorials/customize_pre_trained_model
   tutorials/customize_dataloader
   tutorials/finetune_a_pre_trained_vit_from_timm
   tutorials/finetune_a_custom_pre_trained_model
   tutorials/advanced_extended_structure
   tutorials/advanced_customize_knowledge_transfer
   tutorials/advanced_customize_model_merging

.. toctree::
   :maxdepth: 2
   :caption: API Docs

   api/zhijian.args
   api/zhijian.models
   api/zhijian.data
   api/zhijian.trainer
   api/architect
   api/tuner
   api/merger


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
