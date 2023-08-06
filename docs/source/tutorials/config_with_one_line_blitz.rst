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

In the Architect module, to facilitate the modification of model structures, additional adaptive structures are incorporated into pre-trained models. :lamdaorange:`Z`:lamdablue:`h`:lamdablue:`i`:lamdaorange:`J`:lamdablue:`i`:lamdablue:`a`:lamdablue:`n` accepts a one-line serialized representation of the base pre-trained model, as exemplified in the Vision Transformer model from the :code:`timm` library in the following manner:

.. figure:: ../_static/images/tutorial_one_line_config.png
   :align: center


The modules within the parentheses :code:`()` represent the base pre-trained model, and the dot :code:`.` is used as a access operator.

The arrows :code:`->` indicate the connections between modules, and ellipsis :code:`...` represents default modules. Partial structures can be connected with arrows.

Extended Add-in Module with Entry Points
-------------------------

We use :code:`(): ` to denote an additional adaptive structure, where the part after the dot :code:`.` represents the main forward function of the extra structure. The data flows into the module and primarily passes through this method.

We use :code:`{}` to indicate the entry points of the extra structure into the pre-trained model, encompassing the entry of source model features and the return points of features after the added structure is processed.

With the aforementioned configuration, ZhiJian seamlessly supports the modification of pre-trained model structures. It automatically recognizes the additional structures defined in :code:`zhijian\models\addin`, enabling the construction of pre-trained models.