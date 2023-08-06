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
    .customlyellow {
        color: #00ff73;
    }

    </style>




Config with ~1 Line Blitz
=================================

.. raw:: html

   <span style="font-size: 25px;">🌱</span>
   <p></p>


:lamdaorange:`Z`:lamdablue:`h`:lamdablue:`i`:lamdaorange:`J`:lamdablue:`i`:lamdablue:`a`:lamdablue:`n` is an **unifying** and **rapidly deployable** toolbox for **pre-trained model reuse**.

- **What** \& **Why** Reuse?

  - Performing downstream tasks **with the help of pre-trained model**, including model structures, weights, or other derived rules.
  - Significantly **accelerating convergence** and **improving downstream performance**.

In :lamdaorange:`Z`:lamdablue:`h`:lamdablue:`i`:lamdaorange:`J`:lamdablue:`i`:lamdablue:`a`:lamdablue:`n`, adding the **LoRA** module to the pre-trained model and adjusting which part of the parameters to fine-tune just require **about** :customlyellow:`one` **line of code.**


Overview
-------------------------

:customcolor5:`In` :customcolor4:`the` :customcolor2:`following` :customcolor1:`example`, we show how :lamdaorange:`Z`:lamdablue:`h`:lamdablue:`i`:lamdaorange:`J`:lamdablue:`i`:lamdablue:`a`:lamdablue:`n`:

  + Represent the modules of the pre-trained model
  + Config the extended add-in module with entry points


Modules of Pre-trained Model in One Line description
-------------------------

'(LoRA.adapt): ...->(blocks[0:12].attn.qkv){inout1}->...',

在Architect模块中，为了更方便地修改模型结构，为预训练模型添加额外的适配结构，ZhiJian接受单行序列化地表示基础的预训练模型，例如下图的timm库中的Vision Transformer模型，我们可以用如下的单行配置来表示它：

:lamdaorange:`Z`:lamdablue:`h`:lamdablue:`i`:lamdaorange:`J`:lamdablue:`i`:lamdablue:`a`:lamdablue:`n`

其中()括号内表示基础预训练模型的模块，在括号内支持用'.'访问符来TODO

箭头表示模块间的连接，...表示缺省的模块，允许用箭头连接部分结构，如上述代码所示，我们只表示了blocks TODO模块


Extended Add-in Module with Entry Points
-------------------------

我们用(): 来表示一个额外的适配结构，其中'.'之后是额外结构的主要forward函数，数据流入模块后将主要经过该方法。

我们用{}来表示额外结构对于预训练模型的接入点，包括源模型特征的入口和添加结构处理后特征的返回点


总结
-------------------------

如上配置后，ZhiJian支持无缝地修改预训练模型的结构，它将自动识别定义在:code:`zhijian\models\addin`中的额外结构，完成预训练网络的搭建
