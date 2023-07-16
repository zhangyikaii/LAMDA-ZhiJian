from zhijian.models.configs.base import TQDM_BAR_FORMAT
from zhijian.models.utils import AverageMeter
from zhijian.models.backbone.timm.hooks import HOOKS
from zhijian.models.backbone.base import prepare_hook
from zhijian.models.regularization.base import prepare_reg_loss
from zhijian.trainers.finetune import Trainer as Base_Trainer

import time
import torch
from contextlib import suppress
import os
from copy import deepcopy

from tqdm import tqdm

def prepare_specific_trainer_parser(parser):
    parser.add_argument('--soup-mode', type=str, default=None)
    return parser


class Trainer(Base_Trainer):
    def __init__(
        self, args,
        model=None,
        model_args=None,
        train_loader=None,
        val_loader=None,
        num_classes=None,
        optimizer=None,
        lr_scheduler=None,
        criterion=None,
        device=None
        ):
        super().__init__(args, model, model_args, train_loader, val_loader, num_classes, optimizer, lr_scheduler, criterion, device)
