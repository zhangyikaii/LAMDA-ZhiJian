from zhijian.models.configs.base import TQDM_BAR_FORMAT
from zhijian.models.utils import AverageMeter
from zhijian.models.backbone.timm.hooks import HOOKS
from zhijian.models.backbone.base import prepare_hook
from zhijian.models.regularization.base import prepare_reg_loss
from zhijian.trainers.knowledge_distillation import TwinPTMTrainer as Base_Trainer

import time
import torch
from contextlib import suppress
from copy import deepcopy

from tqdm import tqdm


def prepare_specific_trainer_parser(parser):
    parser.add_argument('--t-config', type=str, default=None)
    parser.add_argument('--reg-mode', type=str, default=None)
    parser.add_argument('--reg-alpha', type=float, default=0.1)
    parser.add_argument('--reg-beta', type=float, default=1)

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
        self.t_model.eval()
        self.source_weights = self.t_model.state_dict()

        cur_model_hook = HOOKS[args.model.replace('timm.', '')]()
        prepare_hook(cur_model_hook.get_layer_feature_hook_info(), [cur_model_hook], self.model, 'hook')

        if args.reg_mode in ['delta']:
            self.init_model = deepcopy(self.model)
            with torch.no_grad():
                for param in self.init_model.parameters():
                    param.requires_grad = False
            self.init_model.eval()

        if criterion is not None:
            self.reg_criterion = criterion
        else:
            self.reg_criterion = prepare_reg_loss(self.args)


    def fit(self):
        if self.only_do_test:
            return
        best_val_acc1, best_epoch = 0, 0

        for epoch in range(self.max_epoch):
            self.model.train()

            end = time.time()
            num_batches_per_epoch = len(self.train_loader)
            batch_time_m, data_time_m, losses_m = AverageMeter(), AverageMeter(), AverageMeter()
            pbar = enumerate(self.train_loader)
            if self.verbose:
                self.logger.info(('\n' + '%11s' * 5) % ('Epoch', 'GPU Mem.', 'Time', 'Loss', 'LR'), only_print=True)
                pbar = tqdm(pbar, total=num_batches_per_epoch, unit='batch', unit_scale=True, bar_format=TQDM_BAR_FORMAT)
            for _, (input, target) in pbar:
                input, target = input.to(self.device), target.to(self.device)
                data_time_m.update(time.time() - end)

                with suppress():
                    outputs = self.model.reuse_callback(
                        self.model(
                            self.model.input_callback(
                                input
                            )
                        )
                    )
                    loss = self.criterion(outputs, target)
                    model_params = {k: v.detach() for k, v in self.model.named_parameters()}
                    if self.args.reg_mode == 'l2sp':
                        loss += self.reg_criterion(self.source_weights, model_params)
                    elif self.args.reg_mode == 'delta':
                        _ = self.init_model(input)
                        loss += self.reg_criterion(model_params, self.init_model.hook[0].get_layer_feature(), self.model.hook[0].get_layer_feature())
                    elif self.args.reg_mode == 'bss':
                        loss += self.reg_criterion(self.model.hook[0].get_layer_feature()[-1])
                    else:
                        raise NotImplementedError

                losses_m.update(loss.item(), input.size(0))

                self.optimizer.zero_grad()

                loss.backward()
                self.optimizer.step()

                batch_time_m.update(time.time() - end)

                end = time.time()
                if self.verbose:
                    mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                    pbar.set_description(('%11s' * 2 + '%11.4g' * 2 + '%11.5g') %
                                         (f'{epoch + 1}/{self.max_epoch}', mem, batch_time_m.avg, losses_m.avg, self.optimizer.param_groups[0]['lr']))

            self.lr_scheduler.step()

            val_acc1, val_acc5 = self.test(epoch)

            if val_acc1 > best_val_acc1:
                best_val_acc1 = val_acc1
                best_val_acc5 = val_acc5
                self.logger.save_model(
                    model=deepcopy(self.model).to('cpu'),
                    optimizer=self.optimizer,
                    epoch=epoch,
                    save_file='best.pt'
                )
