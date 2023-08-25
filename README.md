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
        A Unifying and Rapidly Deployable Toolbox for Pre-trained Model Reuse
    <p>
    <p>
        <a href="https://arxiv.org/abs/2308.09158">[Paper]</a> [<b>Code</b>] <a href="https://zhijian.readthedocs.io/en/latest/#">[Docs]</a>
    <p>
    <p>
        <b>English</b> |
        <a href="https://github.com/zhangyikaii/LAMDA-ZhiJian/blob/main/README_zh.md">中文</a>
    <p>
</h4>


**ZhiJian** ([**执简**驭繁](https://baike.baidu.com/item/%E6%89%A7%E7%AE%80%E9%A9%AD%E7%B9%81)) is a *comprehensive* and *user-friendly* `PyTorch`-based **Model Reuse toolbox** for leveraging **foundation pre-trained models** and their **fine-tuned counterparts** to *extract* knowledge and *expedite* learning in real-world tasks.

**The rapid progress** in deep learning has led to the emergence of **numerous open-source Pre-Trained Models (PTMs)** on platforms like PyTorch, TensorFlow, and HuggingFace Transformers. Leveraging these PTMs for specific tasks empowers them to handle objectives effectively, creating valuable resources for the machine-learning community. **Reusing PTMs is vital in enhancing target models' capabilities and efficiency**, achieved through adapting the architecture, customizing learning on target data, or devising optimized inference strategies to leverage PTM knowledge.

![overview](https://github.com/zhangyikaii/LAMDA-ZhiJian/raw/main/assests/overview.png?raw=true)

🔥 **To facilitate a holistic consideration of various model reuse strategies**, ZhiJian categorizes model reuse methods into *three* sequential modules: **Architect**, **Tuner**, and **Merger**, aligning with the stages of **model preparation**, **model learning**, and **model inference** on the target task, respectively. **The provided interface methods include**:

<details>
<summary style="margin-left: 2px;"><b>A</b>rchitect Module [<em>Click to Expand</em>]<p style="margin-left: 12px;">The Architect module involves <b>modifying the pre-trained model to fit the target task</b>, and reusing certain parts of the pre-trained model while introducing new learnable parameters with specialized structures.</p></summary>
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
<summary style="margin-left: 2px;"><b>T</b>uner Module [<em>Click to Expand</em>]<p style="margin-left: 12px;">The Tuner module focuses on <b>training the target model with guidance from pre-trained model knowledge</b> to expedite the optimization process, <em>e.g.</em>, via adjusting objectives, optimizers, or regularizers.</p></summary>
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
<summary style="margin-left: 2px;"><b>M</b>erger Module [<em>Click to Expand</em>]<p style="margin-left: 12px;">The Merger module influences <b>the inference phase</b> by either reusing pre-trained features or incorporating adapted logits from the pre-trained model.</p></summary>
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

💡 **ZhiJian** also has the following **highlights**:

+ **Support** reuse of various **pre-trained model zoo**, including:
  +  PyTorch [Torchvision](https://pytorch.org/vision/stable/models.html); OpenAI [CLIP](https://github.com/openai/CLIP); 🤗Hugging Face [PyTorch Image Models (timm)](https://github.com/huggingface/pytorch-image-models), [Transformers](https://github.com/huggingface/transformers)
  + Other popular projects, *e.g.*, [vit-pytorch](https://github.com/lucidrains/vit-pytorch) (stars [14k](https://github.com/lucidrains/vit-pytorch/stargazers)).
  + Large Language Model, including [baichuan](https://huggingface.co/baichuan-inc/baichuan-7B), [LLaMA](https://github.com/facebookresearch/llama), and [BLOOM](https://huggingface.co/bigscience/bloom).
+ **Extremely easy** to get started and **customize**
  + Get started with a 10 minute blitz [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Ho1R6h5FEg6zXBJVauXcBnSpBrfi6JmN?usp=sharing)
  + Customize datasets and pre-trained models with step-by-step instructions [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1PKy1U7DyAy5AJYIBv5VEoHWEDJ6NCwTZ?usp=sharing)
  + Feel free to create a novel approach for reusing pre-trained model [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1vHQjlaAGhoeiTVAwOrSQCraAlDvWOlh9?usp=sharing)
+ **Concise** things do **big**
  + Only ~5000 lines of the base code, with incorporating method like building *LEGO* blocks
  + **State-of-the-art** results on VTAB-M benchmark with approximately **10k** experiments [[here]](https://github.com/zhangyikaii/LAMDA-ZhiJian/tree/main/results)
  + Support friendly guideline and comprehensive documentation to custom dataset and pre-trained model [[here]](https://zhijian.readthedocs.io/en/latest/tutorials/get_started.html)

> "ZhiJian" in Chinese means handling complexity with concise and efficient methods. Given the variations in pre-trained models and the deployment overhead of full parameter fine-tuning, ZhiJian represents a solution that is easily reusable, maintains high accuracy, and maximizes the potential of pre-trained models.
> 
> “执简驭繁”的意思是用简洁高效的方法驾驭纷繁复杂的事物。“繁”表示现有预训练模型和复用方法种类多、差异大、部署难，所以取名"执简"的意思是通过该工具包，能轻松地驾驭模型复用方法，易上手、快复用、稳精度，最大限度地唤醒预训练模型的知识。

&nbsp;

## 🕹️ Quick Start

1. An environment with Python 3.7+ from [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html "conda-env"), [venv](https://docs.python.org/3/library/venv.html), or [virtualenv](https://virtualenv.pypa.io/en/latest/).

2. Install ZhiJian using pip:
   ```bash
   $ pip install zhijian
   ```

   + [Option] Install with the newest version through GitHub:
      ```bash
      $ pip install git+https://github.com/ZhangYikaii/LAMDA-ZhiJian.git@main --upgrade
      ```

3. Open your python console and type
   ```python
   import zhijian
   print(zhijian.__version__)
   ```
   If no error occurs, you have successfully installed ZhiJian.


&nbsp;

## Documentation

📚 The tutorials and API documentation are hosted on [ZhiJian.readthedocs.io](https://zhijian.readthedocs.io/)

&nbsp;

## Why ZhiJian?

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


<sup><b>(1)</b>: access date: 2023-08-05</sup>
<sup><b>(2)</b>: fields for (a) Architect; (b) Tuner; (c) Merger;</sup>

### 📦 Reproducible SoTA Results

**ZhiJian** fixed the random seed to ensure reproducibility of the results, with only minor variations across different devices.


| Method | Tuned Params | Mixed Mean | Caltech101 | CIFAR-100 | CLEVR-Count | CLEVR-Distance | Diabetic-Retinopathy | Dmlab | dSprites-Location | dSprites-Orientation | DTD | EuroSAT | KITTI | Oxford-Flowers-102 | Oxford-IIIT-Pet | PatchCamelyon | RESISC45 | smallNORB-Azimuth | smallNORB-Elevation | SVHN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| <div style="white-space: nowrap">**Adapter** </div> | 0.73/86.53(M) |  | <a href="./results/configs/ViT-B-16-VTAB-M-Caltech101-Adapter.json">84.16</a> | <a href="./results/configs/ViT-B-16-VTAB-M-CIFAR-100-Adapter.json">66.74</a> | <a href="./results/configs/ViT-B-16-VTAB-M-CLEVR-Count-Adapter.json">30.43</a> | <a href="./results/configs/ViT-B-16-VTAB-M-CLEVR-Distance-Adapter.json">22.97</a> | <a href="./results/configs/ViT-B-16-VTAB-M-Diabetic-Retinopathy-Adapter.json">75.92</a> | <a href="./results/configs/ViT-B-16-VTAB-M-Dmlab-Adapter.json">46.29</a> | <a href="./results/configs/ViT-B-16-VTAB-M-dSprites-Location-Adapter.json">3.76</a> | <a href="./results/configs/ViT-B-16-VTAB-M-dSprites-Orientation-Adapter.json">26.47</a> | <a href="./results/configs/ViT-B-16-VTAB-M-DTD-Adapter.json">68.03</a> | <a href="./results/configs/ViT-B-16-VTAB-M-EuroSAT-Adapter.json">95.13</a> | <a href="./results/configs/ViT-B-16-VTAB-M-KITTI-Adapter.json">49.09</a> | <a href="./results/configs/ViT-B-16-VTAB-M-Oxford-Flowers-102-Adapter.json">98.63</a> | <a href="./results/configs/ViT-B-16-VTAB-M-Oxford-IIIT-Pet-Adapter.json">91.47</a> | <a href="./results/configs/ViT-B-16-VTAB-M-PatchCamelyon-Adapter.json">79.21</a> | <a href="./results/configs/ViT-B-16-VTAB-M-RESISC45-Adapter.json">82.25</a> | <a href="./results/configs/ViT-B-16-VTAB-M-smallNORB-Azimuth-Adapter.json">7.99</a> | <a href="./results/configs/ViT-B-16-VTAB-M-smallNORB-Elevation-Adapter.json">23.20</a> | <a href="./results/configs/ViT-B-16-VTAB-M-SVHN-Adapter.json">76.71</a> |
| <div style="white-space: nowrap">**LoRA** </div> | 0.71/86.51(M) |  | <a href="./results/configs/ViT-B-16-VTAB-M-Caltech101-LoRA.json">84.75</a> | <a href="./results/configs/ViT-B-16-VTAB-M-CIFAR-100-LoRA.json">63.92</a> | <a href="./results/configs/ViT-B-16-VTAB-M-CLEVR-Count-LoRA.json">33.25</a> | <a href="./results/configs/ViT-B-16-VTAB-M-CLEVR-Distance-LoRA.json">27.85</a> | <a href="./results/configs/ViT-B-16-VTAB-M-Diabetic-Retinopathy-LoRA.json">76.37</a> | <a href="./results/configs/ViT-B-16-VTAB-M-Dmlab-LoRA.json">44.90</a> | <a href="./results/configs/ViT-B-16-VTAB-M-dSprites-Location-LoRA.json">4.54</a> | <a href="./results/configs/ViT-B-16-VTAB-M-dSprites-Orientation-LoRA.json">24.72</a> | <a href="./results/configs/ViT-B-16-VTAB-M-DTD-LoRA.json">68.56</a> | <a href="./results/configs/ViT-B-16-VTAB-M-EuroSAT-LoRA.json">94.33</a> | <a href="./results/configs/ViT-B-16-VTAB-M-KITTI-LoRA.json">50.91</a> | <a href="./results/configs/ViT-B-16-VTAB-M-Oxford-Flowers-102-LoRA.json">98.80</a> | <a href="./results/configs/ViT-B-16-VTAB-M-Oxford-IIIT-Pet-LoRA.json">91.66</a> | <a href="./results/configs/ViT-B-16-VTAB-M-PatchCamelyon-LoRA.json">82.57</a> | <a href="./results/configs/ViT-B-16-VTAB-M-RESISC45-LoRA.json">82.71</a> | <a href="./results/configs/ViT-B-16-VTAB-M-smallNORB-Azimuth-LoRA.json">5.92</a> | <a href="./results/configs/ViT-B-16-VTAB-M-smallNORB-Elevation-LoRA.json">27.00</a> | <a href="./results/configs/ViT-B-16-VTAB-M-SVHN-LoRA.json">74.30</a> |
| <div style="white-space: nowrap">**VPT / Deep**</div> | 0.45/86.24(M) |  | <a href="./results/configs/ViT-B-16-VTAB-M-Caltech101-VPT-Deep.json">83.15</a> | <a href="./results/configs/ViT-B-16-VTAB-M-CIFAR-100-VPT-Deep.json">52.39</a> | <a href="./results/configs/ViT-B-16-VTAB-M-CLEVR-Count-VPT-Deep.json">23.49</a> | <a href="./results/configs/ViT-B-16-VTAB-M-CLEVR-Distance-VPT-Deep.json">20.67</a> | <a href="./results/configs/ViT-B-16-VTAB-M-Diabetic-Retinopathy-VPT-Deep.json">75.13</a> | <a href="./results/configs/ViT-B-16-VTAB-M-Dmlab-VPT-Deep.json">39.37</a> | <a href="./results/configs/ViT-B-16-VTAB-M-dSprites-Location-VPT-Deep.json">2.84</a> | <a href="./results/configs/ViT-B-16-VTAB-M-dSprites-Orientation-VPT-Deep.json">23.06</a> | <a href="./results/configs/ViT-B-16-VTAB-M-DTD-VPT-Deep.json">66.12</a> | <a href="./results/configs/ViT-B-16-VTAB-M-EuroSAT-VPT-Deep.json">93.13</a> | <a href="./results/configs/ViT-B-16-VTAB-M-KITTI-VPT-Deep.json">42.33</a> | <a href="./results/configs/ViT-B-16-VTAB-M-Oxford-Flowers-102-VPT-Deep.json">97.82</a> | <a href="./results/configs/ViT-B-16-VTAB-M-Oxford-IIIT-Pet-VPT-Deep.json">90.00</a> | <a href="./results/configs/ViT-B-16-VTAB-M-PatchCamelyon-VPT-Deep.json">77.45</a> | <a href="./results/configs/ViT-B-16-VTAB-M-RESISC45-VPT-Deep.json">79.75</a> | <a href="./results/configs/ViT-B-16-VTAB-M-smallNORB-Azimuth-VPT-Deep.json">7.65</a> | <a href="./results/configs/ViT-B-16-VTAB-M-smallNORB-Elevation-VPT-Deep.json">18.02</a> | <a href="./results/configs/ViT-B-16-VTAB-M-SVHN-VPT-Deep.json">63.87</a> |
| <div style="white-space: nowrap">**Linear Probing** </div> | 0.42/86.22(M) |  | <a href="./results/configs/ViT-B-16-VTAB-M-Caltech101-Linear-Probing.json">80.93</a> | <a href="./results/configs/ViT-B-16-VTAB-M-CIFAR-100-Linear-Probing.json">37.15</a> | <a href="./results/configs/ViT-B-16-VTAB-M-CLEVR-Count-Linear-Probing.json">14.07</a> | <a href="./results/configs/ViT-B-16-VTAB-M-CLEVR-Distance-Linear-Probing.json">22.27</a> | <a href="./results/configs/ViT-B-16-VTAB-M-Diabetic-Retinopathy-Linear-Probing.json">74.68</a> | <a href="./results/configs/ViT-B-16-VTAB-M-Dmlab-Linear-Probing.json">35.32</a> | <a href="./results/configs/ViT-B-16-VTAB-M-dSprites-Location-Linear-Probing.json">3.29</a> | <a href="./results/configs/ViT-B-16-VTAB-M-dSprites-Orientation-Linear-Probing.json">18.51</a> | <a href="./results/configs/ViT-B-16-VTAB-M-DTD-Linear-Probing.json">60.69</a> | <a href="./results/configs/ViT-B-16-VTAB-M-EuroSAT-Linear-Probing.json">88.72</a> | <a href="./results/configs/ViT-B-16-VTAB-M-KITTI-Linear-Probing.json">40.08</a> | <a href="./results/configs/ViT-B-16-VTAB-M-Oxford-Flowers-102-Linear-Probing.json">97.59</a> | <a href="./results/configs/ViT-B-16-VTAB-M-Oxford-IIIT-Pet-Linear-Probing.json">88.09</a> | <a href="./results/configs/ViT-B-16-VTAB-M-PatchCamelyon-Linear-Probing.json">79.36</a> | <a href="./results/configs/ViT-B-16-VTAB-M-RESISC45-Linear-Probing.json">72.98</a> | <a href="./results/configs/ViT-B-16-VTAB-M-smallNORB-Azimuth-Linear-Probing.json">7.42</a> | <a href="./results/configs/ViT-B-16-VTAB-M-smallNORB-Elevation-Linear-Probing.json">15.09</a> | <a href="./results/configs/ViT-B-16-VTAB-M-SVHN-Linear-Probing.json">38.34</a> |
| <div style="white-space: nowrap">**Partial-1** </div> | 7.51/86.22(M) |  | <a href="./results/configs/ViT-B-16-VTAB-M-Caltech101-Partial-1.json">81.87</a> | <a href="./results/configs/ViT-B-16-VTAB-M-CIFAR-100-Partial-1.json">42.01</a> | <a href="./results/configs/ViT-B-16-VTAB-M-CLEVR-Count-Partial-1.json">25.50</a> | <a href="./results/configs/ViT-B-16-VTAB-M-CLEVR-Distance-Partial-1.json">24.34</a> | <a href="./results/configs/ViT-B-16-VTAB-M-Diabetic-Retinopathy-Partial-1.json">75.20</a> | <a href="./results/configs/ViT-B-16-VTAB-M-Dmlab-Partial-1.json">39.39</a> | <a href="./results/configs/ViT-B-16-VTAB-M-dSprites-Location-Partial-1.json">2.08</a> | <a href="./results/configs/ViT-B-16-VTAB-M-dSprites-Orientation-Partial-1.json">24.29</a> | <a href="./results/configs/ViT-B-16-VTAB-M-DTD-Partial-1.json">63.94</a> | <a href="./results/configs/ViT-B-16-VTAB-M-EuroSAT-Partial-1.json">91.37</a> | <a href="./results/configs/ViT-B-16-VTAB-M-KITTI-Partial-1.json">34.60</a> | <a href="./results/configs/ViT-B-16-VTAB-M-Oxford-Flowers-102-Partial-1.json">97.82</a> | <a href="./results/configs/ViT-B-16-VTAB-M-Oxford-IIIT-Pet-Partial-1.json">89.48</a> | <a href="./results/configs/ViT-B-16-VTAB-M-PatchCamelyon-Partial-1.json">79.50</a> | <a href="./results/configs/ViT-B-16-VTAB-M-RESISC45-Partial-1.json">77.57</a> | <a href="./results/configs/ViT-B-16-VTAB-M-smallNORB-Azimuth-Partial-1.json">7.65</a> | <a href="./results/configs/ViT-B-16-VTAB-M-smallNORB-Elevation-Partial-1.json">21.85</a> | <a href="./results/configs/ViT-B-16-VTAB-M-SVHN-Partial-1.json">50.35</a> | 


&nbsp;

## Contributing

**ZhiJian** is currently in active development, and we warmly welcome any contributions aimed at enhancing capabilities. Whether you have insights to share regarding pre-trained models, data, or innovative reuse methods, we eagerly invite you to join us in making **ZhiJian** even better. If you want to submit your valuable contributions, please click [here](https://zhijian.readthedocs.io/en/latest/contributing.html).

&nbsp;

## Citing ZhiJian

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
