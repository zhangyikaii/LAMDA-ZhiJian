<div align="center">
  <a href="http://zhijian.readthedocs.io"><img width="450px" height="auto" src="https://github.com/zhangyikaii/LAMDA-ZhiJian/raw/main/assests/logo.png?raw=true"></a>
</div>

&nbsp;

<div align="center">
    <img src="https://img.shields.io/badge/License-MIT-<COLOR>.svg?style=for-the-badge" alt="Generic badge", height="21">
    <img src="https://img.shields.io/github/actions/workflow/status/zhangyikaii/LAMDA-ZhiJian/tests.yml?branch=main&style=for-the-badge" alt="GitHub Workflow Status (branch)", height="21">
    <img src="https://img.shields.io/readthedocs/smp?style=for-the-badge&logo=readthedocs&logoColor=white" alt="Read the Docs", height="21">
    <br>
    <img src="https://img.shields.io/pypi/v/ZhiJian?color=blue&style=for-the-badge&logo=pypi&logoColor=white" alt="PyPI", height="21">
    <img src="https://img.shields.io/pypi/dm/ZhiJian?style=for-the-badge&color=blue" alt="PyPI - Downloads", height="21">
    <br>
    <img src="https://img.shields.io/badge/PYTORCH-1.4+-red?style=for-the-badge&logo=pytorch" alt="PyTorch - Version", height="21">
    <img src="https://img.shields.io/badge/PYTHON-3.7+-red?style=for-the-badge&logo=python&logoColor=white" alt="Python - Version", height="21">
</div>
<h4 align="center">
    <p>
        执简：为预训练模型复用提供快速部署的统一方案
    <p>
    <p>
        <a href="https://arxiv.org/abs/2308.09158">[论文]</a> [<b>代码</b>] <a href="https://zhijian.readthedocs.io/en/latest/#">[文档]</a>
    <p>
    <p>
        <a href="https://github.com/zhangyikaii/LAMDA-ZhiJian/blob/main/README.md">English</a> |
        <b>中文</b>
    <p>
</h4>


**ZhiJian** ([**执简**驭繁](https://baike.baidu.com/item/%E6%89%A7%E7%AE%80%E9%A9%AD%E7%B9%81)) 是一个基于`PyTorch`框架的*模型复用*工具包，为再次利用许多**基座预训练模型**以及已有任务上的**已训练模型**，充分提取它们蕴含的知识**并激发目标任务上的学习**，提供了全面且统一的复用方案。

近年来人工智能的蓬勃发展产出了许多开源预训练模型（Pre-Trained Models），例如PyTorch、TensorFlow和HuggingFace Transformers等平台上存储了大量模型资源。模型复用通过**适配网络结构、定制学习方式以及优化推理策略**来利用这些预训练模型，来**进一步加速和强化**目标任务上的学习，这将为机器学习社区源源不断地贡献价值。

![overview](https://github.com/zhangyikaii/LAMDA-ZhiJian/raw/main/assests/overview.png?raw=true)

为了全面而简洁地考虑各种模型复用策略，**ZhiJian** 将复用方法归类为三个主要模块：**构建者**，**微调者**，和**融合者**，它们分别与目标任务部署时模型准备阶段、学习阶段和推理阶段相对应。执简工具包提供的接口和方法包括：

<details>
<summary style="margin-left: 2px;"><b>构建者</b> 模块 [<em>点击以展开</em>]<p style="margin-left: 12px;">构建者模块包含<b>修改预训练模型以适应目标任务</b>，引入具有任务特定结构的全新可学习参数，同时确定重用预训练模型的某些部分。</p></summary>
  <details>
  <summary style="margin-left: 12px;"><strong>&nbsp;&nbsp;&nbsp;&nbsp;Linear Probing</strong> & <strong>Partial-k</strong>, <em>How transferable are features in deep neural networks?</em> In: NeurIPS'14. <a href="https://arxiv.org/pdf/1411.1792.pdf">[Paper]</a> <a href="https://github.com">[Code]</a></summary>
  <div style="text-align: center;">
    <img src="https://github.com/zhangyikaii/LAMDA-ZhiJian/blob/main/assests/linear_probing.png?raw=true" alt="WSFG" width="auto" height="300px" />
  </div>
  </details>

  <details>
  <summary style="margin-left: 12px;"><strong>&nbsp;&nbsp;&nbsp;&nbsp;Adapter</strong>, <em>Parameter-Efficient Transfer Learning for NLP.</em> In: ICML'19. <a href="https://arxiv.org/pdf/1902.00751.pdf">[Paper]</a> <a href="https://github.com">[Code]</a></summary>
  <div style="text-align: center;">
    <img src="https://github.com/zhangyikaii/LAMDA-ZhiJian/blob/main/assests/adapter.png?raw=true" alt="WSFG" width="auto" height="300px" />
  </div>
  </details>

  <details>
  <summary style="margin-left: 12px;"><strong>&nbsp;&nbsp;&nbsp;&nbsp;Diff Pruning</strong>, <em>Parameter-Efficient Transfer Learning with Diff Pruning.</em> In: ACL'21. <a href="https://arxiv.org/pdf/2012.07463.pdf">[Paper]</a> <a href="https://github.com">[Code]</a></summary>
  <div style="text-align: center;">
    <img src="https://github.com/zhangyikaii/LAMDA-ZhiJian/blob/main/assests/diff_pruning.png?raw=true" alt="WSFG" width="auto" height="300px" />
  </div>
  </details>

  <details>
  <summary style="margin-left: 12px;"><strong>&nbsp;&nbsp;&nbsp;&nbsp;LoRA</strong>, <em>LoRA: Low-Rank Adaptation of Large Language Models.</em> In: ICLR'22. <a href="https://arxiv.org/pdf/2106.09685.pdf">[Paper]</a> <a href="https://github.com">[Code]</a></summary>
  <div style="text-align: center;">
    <img src="https://github.com/zhangyikaii/LAMDA-ZhiJian/blob/main/assests/lora.png?raw=true" alt="WSFG" width="auto" height="300px" />
  </div>
  </details>

  <details>
  <summary style="margin-left: 12px;"><strong>&nbsp;&nbsp;&nbsp;&nbsp;Visual Prompt Tuning / Prefix</strong>, <em>Visual Prompt Tuning.</em> In: ECCV'22. <a href="https://arxiv.org/pdf/2203.12119.pdf">[Paper]</a> <a href="https://github.com">[Code]</a></summary>
  <div style="text-align: center;">
    <img src="https://github.com/zhangyikaii/LAMDA-ZhiJian/blob/main/assests/visual_prompt_tuning.png?raw=true" alt="WSFG" width="auto" height="300px" />
  </div>
  </details>

  <details>
  <summary style="margin-left: 12px;"><strong>&nbsp;&nbsp;&nbsp;&nbsp;Scaling &amp; Shifting</strong>, <em>Scaling &amp; Shifting Your Features: A New Baseline for Efficient Model Tuning.</em> In: NeurIPS'22. <a href="https://arxiv.org/pdf/2210.08823.pdf">[Paper]</a> <a href="https://github.com">[Code]</a></summary>
  <div style="text-align: center;">
    <img src="https://github.com/zhangyikaii/LAMDA-ZhiJian/blob/main/assests/scaling_and_shifting.png?raw=true" alt="WSFG" width="auto" height="300px" />
  </div>
  </details>

  <details>
  <summary style="margin-left: 12px;"><strong>&nbsp;&nbsp;&nbsp;&nbsp;AdaptFormer</strong>, <em>AdaptFormer: Adapting Vision Transformers for Scalable Visual Recognition.</em> In: NeurIPS'22. <a href="https://arxiv.org/pdf/2205.13535.pdf">[Paper]</a> <a href="https://github.com">[Code]</a></summary>
  <div style="text-align: center;">
    <img src="https://github.com/zhangyikaii/LAMDA-ZhiJian/blob/main/assests/adapterformer.png?raw=true" alt="WSFG" width="auto" height="300px" />
  </div>
  </details>

  <details>
  <summary style="margin-left: 12px;"><strong>&nbsp;&nbsp;&nbsp;&nbsp;BitFit</strong>, <em>BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models.</em> In: ACL'22. <a href="https://arxiv.org/pdf/2106.10199.pdf">[Paper]</a> <a href="https://github.com">[Code]</a></summary>
  <div style="text-align: center;">
    <img src="https://github.com/zhangyikaii/LAMDA-ZhiJian/blob/main/assests/bitfit.png?raw=true" alt="WSFG" width="auto" height="300px" />
  </div>
  </details>

  <details>
  <summary style="margin-left: 12px;"><strong>&nbsp;&nbsp;&nbsp;&nbsp;Convpass</strong>, <em>Convolutional Bypasses Are Better Vision Transformer Adapters.</em> In: Tech Report 07-2022. <a href="https://arxiv.org/pdf/2207.07039.pdf">[Paper]</a> <a href="https://github.com">[Code]</a></summary>
  <div style="text-align: center;">
    <img src="https://github.com/zhangyikaii/LAMDA-ZhiJian/blob/main/assests/convpass.png?raw=true" alt="WSFG" width="auto" height="300px" />
  </div>
  </details>

  <details>
  <summary style="margin-left: 12px;"><strong>&nbsp;&nbsp;&nbsp;&nbsp;Fact-Tuning</strong>, <em>FacT: Factor-Tuning for Lightweight Adaptation on Vision Transformer.</em> In: AAAI'23. <a href="https://arxiv.org/pdf/2212.03145.pdf">[Paper]</a> <a href="https://github.com">[Code]</a></summary>
  <div style="text-align: center;">
    <img src="https://github.com/zhangyikaii/LAMDA-ZhiJian/blob/main/assests/fact_tuning.png?raw=true" alt="WSFG" width="auto" height="300px" />
  </div>
  </details>
</details>

<details>
<summary style="margin-left: 2px;"><b>微调者</b> 模块 [<em>点击以展开</em>]<p style="margin-left: 12px;">微调者模块专注于<b>在预训练模型知识的引导下训练目标模型</b>，以加快优化过程，例如通过调整训练目标、优化器或正则化器等方式。</p></summary>
  <details>
  <summary style="margin-left: 12px;"><strong>&nbsp;&nbsp;&nbsp;&nbsp;Knowledge Transfer</strong>, <em>NeC4.5: neural ensemble based C4.5.</em> In: IEEE Trans. Knowl. Data Eng. 2004. <a href="https://ieeexplore.ieee.org/document/1294896">[Paper]</a> <a href="https://github.com">[Code]</a></summary>
  <div style="text-align: center;">
    <img src="https://github.com/zhangyikaii/LAMDA-ZhiJian/blob/main/assests/knowledge_transfer.png?raw=true" alt="WSFG" width="auto" height="300px" />
  </div>
  </details>

  <details>
  <summary style="margin-left: 12px;"><strong>&nbsp;&nbsp;&nbsp;&nbsp;FitNet</strong>, <em>FitNets: Hints for Thin Deep Nets.</em> In: ICLR'15. <a href="https://arxiv.org/pdf/1412.6550.pdf">[Paper]</a> <a href="https://github.com">[Code]</a></summary>
  <div style="text-align: center;">
    <img src="https://github.com/zhangyikaii/LAMDA-ZhiJian/blob/main/assests/fitnet.png?raw=true" alt="WSFG" width="auto" height="300px" />
  </div>
  </details>

  <details>
  <summary style="margin-left: 12px;"><strong>&nbsp;&nbsp;&nbsp;&nbsp;LwF</strong>, <em>Learning without Forgetting.</em> In: CVPR'19. <a href="https://arxiv.org/pdf/1606.09282.pdf">[Paper]</a> <a href="https://github.com">[Code]</a></summary>
  <div style="text-align: center;">
    <img src="https://github.com/zhangyikaii/LAMDA-ZhiJian/blob/main/assests/learning_without_forgetting.png?raw=true" alt="WSFG" width="auto" height="300px" />
  </div>
  </details>

  <details>
  <summary style="margin-left: 12px;"><strong>&nbsp;&nbsp;&nbsp;&nbsp;FSP</strong>, <em>A Gift from Knowledge Distillation: Fast Optimization, Network Minimization and Transfer Learning.</em> In: CVPR'17. <a href="https://openaccess.thecvf.com/content_cvpr_2017/papers/Yim_A_Gift_From_CVPR_2017_paper.pdf">[Paper]</a> <a href="https://github.com">[Code]</a></summary>
  <div style="text-align: center;">
    <img src="https://github.com/zhangyikaii/LAMDA-ZhiJian/blob/main/assests/fsp.png?raw=true" alt="WSFG" width="auto" height="300px" />
  </div>
  </details>

  <details>
  <summary style="margin-left: 12px;"><strong>&nbsp;&nbsp;&nbsp;&nbsp;NST</strong>, <em>Like What You Like: Knowledge Distill via Neuron Selectivity Transfer.</em> In: CVPR'17. <a href="https://arxiv.org/pdf/1707.01219.pdf">[Paper]</a> <a href="https://github.com">[Code]</a></summary>
  <div style="text-align: center;">
    <img src="https://github.com/zhangyikaii/LAMDA-ZhiJian/blob/main/assests/nst.png?raw=true" alt="WSFG" width="auto" height="300px" />
  </div>
  </details>

  <details>
  <summary style="margin-left: 12px;"><strong>&nbsp;&nbsp;&nbsp;&nbsp;RKD</strong>, <em>Relational Knowledge Distillation.</em> In: CVPR'19. <a href="https://arxiv.org/pdf/1904.05068.pdf">[Paper]</a> <a href="https://github.com">[Code]</a></summary>
  <div style="text-align: center;">
    <img src="https://github.com/zhangyikaii/LAMDA-ZhiJian/blob/main/assests/rkd.png?raw=true" alt="WSFG" width="auto" height="300px" />
  </div>
  </details>

  <details>
  <summary style="margin-left: 12px;"><strong>&nbsp;&nbsp;&nbsp;&nbsp;SPKD</strong>, <em>Similarity-Preserving Knowledge Distillation.</em> In: CVPR'19. <a href="https://arxiv.org/pdf/1907.09682.pdf">[Paper]</a> <a href="https://github.com">[Code]</a></summary>
  <div style="text-align: center;">
    <img src="https://github.com/zhangyikaii/LAMDA-ZhiJian/blob/main/assests/spkd.png?raw=true" alt="WSFG" width="auto" height="300px" />
  </div>
  </details>

  <details>
  <summary style="margin-left: 12px;"><strong>&nbsp;&nbsp;&nbsp;&nbsp;CRD</strong>, <em>Contrastive Representation Distillation.</em> In: ICLR'20. <a href="https://arxiv.org/pdf/1910.10699.pdf">[Paper]</a> <a href="https://github.com">[Code]</a></summary>
  <div style="text-align: center;">
    <img src="https://github.com/zhangyikaii/LAMDA-ZhiJian/blob/main/assests/crd.png?raw=true" alt="WSFG" width="auto" height="300px" />
  </div>
  </details>

  <details>
  <summary style="margin-left: 12px;"><strong>&nbsp;&nbsp;&nbsp;&nbsp;REFILLED</strong>, <em>Distilling Cross-Task Knowledge via Relationship Matching.</em> In: CVPR'20. <a href="http://www.lamda.nju.edu.cn/lus/files/CVPR20_ReFilled.pdf">[Paper]</a> <a href="https://github.com">[Code]</a></summary>
  <div style="text-align: center;">
    <img src="https://github.com/zhangyikaii/LAMDA-ZhiJian/blob/main/assests/refilled.png?raw=true" alt="WSFG" width="auto" height="300px" />
  </div>
  </details>

  <details>
  <summary style="margin-left: 12px;"><strong>&nbsp;&nbsp;&nbsp;&nbsp;WiSE-FT</strong>, <em>Robust fine-tuning of zero-shot models.</em> In: CVPR'22. <a href="https://arxiv.org/pdf/2109.01903.pdf">[Paper]</a> <a href="https://github.com">[Code]</a></summary>
  <div style="text-align: center;">
    <img src="https://github.com/zhangyikaii/LAMDA-ZhiJian/blob/main/assests/wise_tune.png?raw=true" alt="WSFG" width="auto" height="300px" />
  </div>
  </details>

  <details>
  <summary style="margin-left: 12px;"><strong>&nbsp;&nbsp;&nbsp;&nbsp;L<sup>2</sup> penalty / L<sup>2</sup>-SP</strong>, <em>Explicit Inductive Bias for Transfer Learning with Convolutional Networks.</em> In: ICML'18. <a href="https://arxiv.org/pdf/1802.01483.pdf">[Paper]</a> <a href="https://github.com">[Code]</a></summary>
  <div style="text-align: center;">
    <img src="https://github.com/zhangyikaii/LAMDA-ZhiJian/blob/main/assests/l_2_penalty.png?raw=true" alt="WSFG" width="auto" height="300px" />
  </div>
  </details>

  <details>
  <summary style="margin-left: 12px;"><strong>&nbsp;&nbsp;&nbsp;&nbsp;Spectral Norm</strong>, <em>Spectral Normalization for Generative Adversarial Networks.</em> In: ICLR'18. <a href="https://arxiv.org/pdf/1802.05957.pdf">[Paper]</a> <a href="https://github.com">[Code]</a></summary>
  <div style="text-align: center;">
    <img src="https://github.com/zhangyikaii/LAMDA-ZhiJian/blob/main/assests/spectral_norm.png?raw=true" alt="WSFG" width="auto" height="300px" />
  </div>
  </details>

  <details>
  <summary style="margin-left: 12px;"><strong>&nbsp;&nbsp;&nbsp;&nbsp;BSS</strong>, <em>Catastrophic Forgetting Meets Negative Transfer: Batch Spectral Shrinkage for Safe Transfer Learning.</em> In: NeurIPS'19. <a href="https://proceedings.neurips.cc/paper_files/paper/2019/file/c6bff625bdb0393992c9d4db0c6bbe45-Paper.pdf">[Paper]</a> <a href="https://github.com">[Code]</a></summary>
  <div style="text-align: center;">
    <img src="https://github.com/zhangyikaii/LAMDA-ZhiJian/blob/main/assests/bss.png?raw=true" alt="WSFG" width="auto" height="300px" />
  </div>
  </details>

  <details>
  <summary style="margin-left: 12px;"><strong>&nbsp;&nbsp;&nbsp;&nbsp;DELTA</strong>, <em>DELTA: DEep Learning Transfer using Feature Map with Attention for Convolutional Networks.</em> In: ICLR'19. <a href="https://arxiv.org/pdf/1901.09229.pdf">[Paper]</a> <a href="https://github.com">[Code]</a></summary>
  <div style="text-align: center;">
    <img src="https://github.com/zhangyikaii/LAMDA-ZhiJian/blob/main/assests/delta.png?raw=true" alt="WSFG" width="auto" height="300px" />
  </div>
  </details>

  <details>
  <summary style="margin-left: 12px;"><strong>&nbsp;&nbsp;&nbsp;&nbsp;DeiT</strong>, <em>Training data-efficient image transformers & distillation through attention.</em> In: ICML'21. <a href="https://arxiv.org/pdf/2012.12877.pdf">[Paper]</a> <a href="https://github.com">[Code]</a></summary>
  <div style="text-align: center;">
    <img src="https://github.com/zhangyikaii/LAMDA-ZhiJian/blob/main/assests/deit.png?raw=true" alt="WSFG" width="auto" height="300px" />
  </div>
  </details>

  <details>
  <summary style="margin-left: 12px;"><strong>&nbsp;&nbsp;&nbsp;&nbsp;DIST</strong>, <em>Knowledge Distillation from A Stronger Teacher.</em> In: NeurIPS'22. <a href="https://arxiv.org/pdf/2205.10536.pdf">[Paper]</a> <a href="https://github.com">[Code]</a></summary>
  <div style="text-align: center;">
    <img src="https://github.com/zhangyikaii/LAMDA-ZhiJian/blob/main/assests/dist.png?raw=true" alt="WSFG" width="auto" height="300px" />
  </div>
  </details>
</details>

<details>
<summary style="margin-left: 2px;"><b>融合者</b> 模块 [<em>点击以展开</em>]<p style="margin-left: 12px;">融合者模块<b>在推理阶段</b>通过复用预训练特征，或融合来自适配后的预训练输出来获得更强的泛化能力。</p></summary>
  <details>
  <summary style="margin-left: 12px;"><strong>&nbsp;&nbsp;&nbsp;&nbsp;Nearest Class Mean</strong>, <em>Generalizing to new classes at near-zero cost.</em> In: TPAMI'13. <a href="https://ieeexplore.ieee.org/document/6517188">[Paper]</a> <a href="https://github.com">[Code]</a></summary>
  <div style="text-align: center;">
    <img src="https://github.com/zhangyikaii/LAMDA-ZhiJian/blob/main/assests/ncm.png?raw=true" alt="WSFG" width="auto" height="300px" />
  </div>
  </details>

  <details>
  <summary style="margin-left: 12px;"><strong>&nbsp;&nbsp;&nbsp;&nbsp;SimpleShot</strong>, <em>SimpleShot: Revisiting Nearest-Neighbor Classification for Few-Shot Learning.</em> In: CVPR'19. <a href="https://arxiv.org/pdf/1911.04623.pdf">[Paper]</a> <a href="https://github.com">[Code]</a></summary>
  <div style="text-align: center;">
    <img src="https://github.com/zhangyikaii/LAMDA-ZhiJian/blob/main/assests/simpleshot.png?raw=true" alt="WSFG" width="auto" height="300px" />
  </div>
  </details>

  <details>
  <summary style="margin-left: 12px;"><strong>&nbsp;&nbsp;&nbsp;&nbsp;Head2Toe</strong>, <em>Head2Toe: Utilizing Intermediate Representations for Better Transfer Learning.</em> In: ICML'22. <a href="https://arxiv.org/pdf/2201.03529.pdf">[Paper]</a> <a href="https://github.com">[Code]</a></summary>
  <div style="text-align: center;">
    <img src="https://github.com/zhangyikaii/LAMDA-ZhiJian/blob/main/assests/head2toe.png?raw=true" alt="WSFG" width="auto" height="300px" />
  </div>
  </details>
  
  <details>
    <summary style="margin-left: 12px;"><strong>&nbsp;&nbsp;&nbsp;&nbsp;VQT</strong>, <em>Visual Query Tuning: Towards Effective Usage of Intermediate Representations for Parameter and Memory Efficient Transfer Learning.</em> In: CVPR'23. <a href="https://arxiv.org/pdf/2212.03220.pdf">[Paper]</a> <a href="https://github.com">[Code]</a></summary>
    <div style="text-align: center;">
      <img src="https://github.com/zhangyikaii/LAMDA-ZhiJian/blob/main/assests/vqt.png?raw=true" alt="WSFG" width="auto" height="300px" />
    </div>
  </details>

  <details>
  <summary style="margin-left: 12px;"><strong>&nbsp;&nbsp;&nbsp;&nbsp;via Optimal Transport</strong>, <em>Model Fusion via Optimal Transport.</em> In: NeurIPS'20. <a href="https://arxiv.org/pdf/1910.05653.pdf">[Paper]</a> <a href="https://github.com">[Code]</a></summary>
  <div style="text-align: center;">
    <img src="https://github.com/zhangyikaii/LAMDA-ZhiJian/blob/main/assests/otfusion.png?raw=true" alt="WSFG" width="auto" height="300px" />
  </div>
  </details>

  <details>
  <summary style="margin-left: 12px;"><strong>&nbsp;&nbsp;&nbsp;&nbsp;Model Soup</strong> <em>Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time.</em> In: ICML'22. <a href="https://arxiv.org/pdf/2203.05482.pdf">[Paper]</a> <a href="https://github.com">[Code]</a></summary>
  <div style="text-align: center;">
    <img src="https://github.com/zhangyikaii/LAMDA-ZhiJian/blob/main/assests/modelsoup.png?raw=true" alt="WSFG" width="auto" height="300px" />
  </div>
  </details>

  <details>
  <summary style="margin-left: 12px;"><strong>&nbsp;&nbsp;&nbsp;&nbsp;Fisher Merging</strong> <em>Merging Models with Fisher-Weighted Averaging.</em> In: NeurIPS'22. <a href="https://arxiv.org/pdf/2111.09832.pdf">[Paper]</a> <a href="https://github.com">[Code]</a></summary>
  <div style="text-align: center;">
    <img src="https://github.com/zhangyikaii/LAMDA-ZhiJian/blob/main/assests/fishermerging.png?raw=true" alt="WSFG" width="auto" height="300px" />
  </div>
  </details>

  <details>
  <summary style="margin-left: 12px;"><strong>&nbsp;&nbsp;&nbsp;&nbsp;Deep Model Reassembly</strong> <em>Deep Model Reassembly.</em> In: NeurIPS'22. <a href="https://arxiv.org/pdf/2210.17409.pdf">[Paper]</a> <a href="https://github.com">[Code]</a></summary>
  <div style="text-align: center;">
    <img src="https://github.com/zhangyikaii/LAMDA-ZhiJian/blob/main/assests/dmr.png?raw=true" alt="WSFG" width="auto" height="300px" />
  </div>
  </details>

  <details>
  <summary style="margin-left: 12px;"><strong>&nbsp;&nbsp;&nbsp;&nbsp;REPAIR</strong> <em>REPAIR: REnormalizing Permuted Activations for Interpolation Repair.</em> In: ICLR'23. <a href="https://arxiv.org/pdf/2211.08403.pdf">[Paper]</a> <a href="https://github.com">[Code]</a></summary>
  <div style="text-align: center;">
    <img src="https://github.com/zhangyikaii/LAMDA-ZhiJian/blob/main/assests/repair.png?raw=true" alt="WSFG" width="auto" height="300px" />
  </div>
  </details>

  <details>
  <summary style="margin-left: 12px;"><strong>&nbsp;&nbsp;&nbsp;&nbsp;Git Re-Basin</strong> <em>Git Re-Basin: Merging Models modulo Permutation Symmetries.</em> In: ICLR'23. <a href="https://arxiv.org/pdf/2209.04836.pdf">[Paper]</a> <a href="https://github.com">[Code]</a></summary>
  <div style="text-align: center;">
    <img src="https://github.com/zhangyikaii/LAMDA-ZhiJian/blob/main/assests/gitrebasin.png?raw=true" alt="WSFG" width="auto" height="300px" />
  </div>
  </details>

  <details>
  <summary style="margin-left: 12px;"><strong>&nbsp;&nbsp;&nbsp;&nbsp;ZipIt</strong> <em>ZipIt! Merging Models from Different Tasks without Training.</em> In: ICLR'23. <a href="https://arxiv.org/pdf/2305.03053.pdf">[Paper]</a> <a href="https://github.com">[Code]</a></summary>
  <div style="text-align: center;">
    <img src="https://github.com/zhangyikaii/LAMDA-ZhiJian/blob/main/assests/zipit.png?raw=true" alt="WSFG" width="auto" height="300px" />
  </div>
  </details>
</details>

<!-- &nbsp; -->

💡 **ZhiJian** 还包含如下特色:

+ 支持复用许多开源预训练模型库, 包含：
  +  PyTorch [Torchvision](https://pytorch.org/vision/stable/models.html); OpenAI [CLIP](https://github.com/openai/CLIP); 🤗Hugging Face [PyTorch Image Models (timm)](https://github.com/huggingface/pytorch-image-models), [Transformers](https://github.com/huggingface/transformers)
  + 其他流行的复用框架，*例如*，[vit-pytorch](https://github.com/lucidrains/vit-pytorch) (stars [14k](https://github.com/lucidrains/vit-pytorch/stargazers)).
  + 大语言模型，包含 [baichuan](https://huggingface.co/baichuan-inc/baichuan-7B), [LLaMA](https://github.com/facebookresearch/llama), and [BLOOM](https://huggingface.co/bigscience/bloom).
+ 极简上手与个性化定制：
  + 在10分钟内极速开始 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Ho1R6h5FEg6zXBJVauXcBnSpBrfi6JmN?usp=sharing)
  + 一步步地定制数据集和预训练模型 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1PKy1U7DyAy5AJYIBv5VEoHWEDJ6NCwTZ?usp=sharing)
  + 随心所欲地创造模型复用的新方法 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1vHQjlaAGhoeiTVAwOrSQCraAlDvWOlh9?usp=sharing)
+ 简洁的结构做得大事儿
  + 基础代码仅约5k行，引入方法就像搭积木一样
  + 在 VTAB-M 基线标准上的最优性能，超过 **10k** 次实验 [[here]](https://github.com/zhangyikaii/LAMDA-ZhiJian/tree/main/results)
  + 支持用户友好的指引和全面的文档来定制化数据集和预训练模型 [[here]](https://zhijian.readthedocs.io/en/latest/tutorials/get_started.html)

> “执简驭繁”的意思是用简洁高效的方法驾驭纷繁复杂的事物。“繁”表示现有预训练模型和复用方法种类多、差异大、部署难，所以取名"执简"的意思是通过该工具包，能轻松地驾驭模型复用方法，易上手、快复用、稳精度，最大限度地唤醒预训练模型的知识。

&nbsp;

## 🕹️ 快速开始

1. 用 [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html "conda-env"), [venv](https://docs.python.org/3/library/venv.html), 或 [virtualenv](https://virtualenv.pypa.io/en/latest/) 部署 Python 3.7+ 的环境

2. 使用 pip 来安装 ZhiJian：
   ```bash
   $ pip install zhijian
   ```

   + [Option] Install with the newest version through GitHub:
      ```bash
      $ pip install git+https://github.com/ZhangYikaii/LAMDA-ZhiJian.git@main --upgrade
      ```

3. 打开 Python 控制台，输入
   ```python
   import zhijian
   print(zhijian.__version__)
   ```
   如果没有错误出现，则成功安装了执简工具包


&nbsp;

## 文档

📚 相关教程和API文档请点击 [ZhiJian.readthedocs.io](https://zhijian.readthedocs.io/)

&nbsp;

## 为什么使用执简工具包？

![architecture](https://github.com/zhangyikaii/LAMDA-ZhiJian/raw/main/assests/architecture.png?raw=true)

<table>
  <tr>
    <td colspan="9" style="border-bottom: 2px solid black;"></td>
  </tr>
  <tr>
    <td><b>Related Library</b></td>
    <td><b>GitHub Stars</b></td>
    <td><b># of Alg.<sup>(1)</sup></b></td>
    <td><b># of Model<sup>(1)</sup></b></td>
    <td><b># of Dataset<sup>(1)</sup></b></td>
    <td><b># of Fields<sup>(2)</sup></b></td>
    <td><b>LLM Supp.</b></td>
    <td><b>Docs.</b></td>
    <td><b>Last Update</b></td>
  </tr>
  <tr>
    <td><a href="https://github.com/huggingface/peft">PEFT</a></td>
    <td><a href="https://github.com/huggingface/peft/stargazers">
      <img src="https://img.shields.io/github/stars/huggingface/peft" alt="GitHub stars">
    </a></td>
    <td>6</td>
    <td>~15</td>
    <td>➖<sup>(3)</sup></td>
    <td>1<sup>(a)</sup></td>
    <td>✔️</td>
    <td>✔️</td>
    <td>
    <a>
      <img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/huggingface/peft?label=last%20update">
    </a>
    </td>
  </tr>
  <tr>
    <td><a href="https://github.com/adapter-hub/adapter-transformers">adapter-transformers</a></td>
    <td><a href="https://github.com/adapter-hub/adapter-transformers/stargazers">
      <img src="https://img.shields.io/github/stars/adapter-hub/adapter-transformers" alt="GitHub stars">
    </a></td>
    <td>10</td>
    <td>~15</td>
    <td>➖<sup>(3)</sup></td>
    <td>1<sup>(a)</sup></td>
    <td>❌</td>
    <td>✔️</td>
    <td>
    <a>
      <img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/adapter-hub/adapter-transformers?label=last%20update">
    </a>
    </td>
  </tr>
  <tr>
    <td><a href="https://github.com/hiyouga/LLaMA-Efficient-Tuning">LLaMA-Efficient-Tuning</a></td>
    <td><a href="https://github.com/hiyouga/LLaMA-Efficient-Tuning/stargazers">
      <img src="https://img.shields.io/github/stars/hiyouga/LLaMA-Efficient-Tuning" alt="GitHub stars">
    </a></td>
    <td>4</sup></td>
    <td>5</td>
    <td>~20</td>
    <td>1<sup>(a)</sup></td>
    <td>✔️</td>
    <td>❌</td>
    <td>
    <a>
      <img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/hiyouga/LLaMA-Efficient-Tuning?label=last%20update">
    </a></td>
  </tr>
  <tr>
    <td><a href="https://github.com/AberHu/Knowledge-Distillation-Zoo">Knowledge-Distillation-Zoo</a></td>
    <td><a href="https://github.com/AberHu/Knowledge-Distillation-Zoo/stargazers">
      <img src="https://img.shields.io/github/stars/AberHu/Knowledge-Distillation-Zoo" alt="GitHub stars">
    </a></td>
    <td>20</td>
    <td>2</td>
    <td>2</td>
    <td>1<sup>(b)</sup></td>
    <td>❌</td>
    <td>❌</td>
    <td>
    <a>
      <img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/AberHu/Knowledge-Distillation-Zoo?label=last%20update">
    </a></td>
  </tr>
  <tr>
    <td><a href="https://github.com/sicara/easy-few-shot-learning">Easy Few-Shot Learning</a></td>
    <td><a href="https://github.com/sicara/easy-few-shot-learning/stargazers">
      <img src="https://img.shields.io/github/stars/sicara/easy-few-shot-learning" alt="GitHub stars">
    </a></td>
    <td>10</td>
    <td>3</td>
    <td>2</td>
    <td>1<sup>(b)</sup></td>
    <td>❌</td>
    <td>❌</td>
    <td>
    <a>
      <img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/sicara/easy-few-shot-learning?label=last%20update">
    </a></td>
  </tr>
  <tr>
    <td><a href="https://github.com/mlfoundations/model-soups">Model soups</a></td>
    <td><a href="https://github.com/mlfoundations/model-soups/stargazers">
      <img src="https://img.shields.io/github/stars/mlfoundations/model-soups" alt="GitHub stars">
    </a></td>
    <td>3</sup></td>
    <td>3</td>
    <td>5</td>
    <td>1<sup>(c)</sup></td>
    <td>❌</td>
    <td>❌</td>
    <td>
    <a>
      <img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/mlfoundations/model-soups?label=last%20update">
    </a></td>
  </tr>
  <tr>
    <td><a href="https://github.com/samuela/git-re-basin">Git Re-Basin</a></td>
    <td><a href="https://github.com/samuela/git-re-basin/stargazers">
      <img src="https://img.shields.io/github/stars/samuela/git-re-basin" alt="GitHub stars">
    </a></td>
    <td>3</sup></td>
    <td>5</td>
    <td>4</td>
    <td>1<sup>(c)</sup></td>
    <td>❌</td>
    <td>❌</td>
    <td>
    <a>
      <img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/samuela/git-re-basin?label=last%20update">
    </a></td>
  </tr>
  <tr>
    <td colspan="9" style="border-bottom: 2px solid grey;"></td>
  </tr>
  </tr>
    <tr>
    <td><b>ZhiJian</b></td>
    <!-- <td><a href="https://github.com/adapter-hub/adapter-transformers/stargazers">
      <img src="https://img.shields.io/github/stars/zhangyikaii/LAMDA-ZhiJian" alt="GitHub stars">
    </a></td> -->
    <td>🙌</td>
    <td>30+</td>
    <td>~50</td>
    <td>19</td>
    <td>3<sup>(a,b,c)</sup></td>
    <td>✔️</td>
    <td>✔️</td>
    <td>
    <a>
      <img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/zhangyikaii/LAMDA-ZhiJian?label=last%20update">
    </a></td>
  </tr>

</table>


<sup><b>(1)</b>: 更新日期: 2023-08-05</sup>
<sup><b>(2)</b>: 涉及领域：(a) 构建者模块；(b) 微调者模块；(c) 融合者模块；</sup>

### 📦 复现 SoTA 结果

**ZhiJian** 固定了随机种子，以确保复现结果，仅在不同设备间存在微小不同

&nbsp;

## 如何贡献

ZhiJian 目前正在积极地开发中，欢迎任何形式的贡献。无论您是否对预训练模型、目标数据或创新的重用方法有哪些见解，我们都热切地邀请您加入我们，共同使 ZhiJian 变得更加优秀。如果您希望提交宝贵的贡献，请点击 [这里](https://zhijian.readthedocs.io/en/latest/contributing.html)。


&nbsp;

## 引用 ZhiJian

```latex
@misc{zhang2023zhijian,
  title={ZhiJian: A Unifying and Rapidly Deployable Toolbox for Pre-trained Model Reuse}, 
  author={Yi-Kai Zhang and Lu Ren and Chao Yi and Qi-Wei Wang and De-Chuan Zhan and Han-Jia Ye},
  year={2023},
  eprint={2308.09158},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}

@misc{zhijian2023,
  author = {ZhiJian Contributors},
  title = {LAMDA-ZhiJian},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/zhangyikaii/LAMDA-ZhiJian}}
}
```
