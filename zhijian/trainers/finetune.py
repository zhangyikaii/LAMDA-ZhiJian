from zhijian.models.backbone.base import prepare_model, prepare_hook, prepare_gradient, prepare_cuda, prepare_pretrained
from zhijian.models.addin.base import prepare_addins
from zhijian.data.base import prepare_vision_dataloader
from zhijian.models.configs.base import TQDM_BAR_FORMAT
from zhijian.models.utils import AverageMeter, accuracy, PrepareFunc, LogHandle, set_seed

import time
import torch
from contextlib import suppress
import os
from copy import deepcopy

from tqdm import tqdm

def prepare_specific_trainer_parser(parser):
    return parser

def get_model(args):
    model, model_args = prepare_model(args)
    addins, fixed_params = prepare_addins(args, model_args)

    prepare_hook(args.addins, addins, model, 'addin')
    prepare_gradient(args.reuse_keys, model)
    device = prepare_cuda(model)
    return model, model_args, device

class Trainer(object):
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
        set_seed(args.seed)
        self.logger = LogHandle(args)

        if None in [model, model_args, device]:
            self.model, self.model_args = prepare_model(args, self.logger)
            self.addins, self.fixed_params = prepare_addins(args, self.model_args)
            prepare_hook(args.addins, self.addins, self.model, 'addin')
            prepare_gradient(args.reuse_keys, self.model, self.logger)
            self.device = prepare_cuda(self.model)
            self.logger.info(f'Training with a single process on 1 device ({self.device})')
        else:
            self.model, self.model_args, self.device = model, model_args, device

        if None in [train_loader, val_loader, num_classes]:
            self.train_loader, self.val_loader, self.num_classes = prepare_vision_dataloader(args, self.model_args, self.logger)
        else:
            self.train_loader, self.val_loader, self.num_classes = train_loader, val_loader, num_classes

        if None in [optimizer, lr_scheduler]:
            prepare_optim_handle = PrepareFunc(args)
            self.optimizer, self.lr_scheduler = prepare_optim_handle.prepare_optimizer(self.model)
        else:
            self.optimizer, self.lr_scheduler = optimizer, lr_scheduler

        if criterion is None:
            prepare_optim_handle = PrepareFunc(args)
            self.criterion = prepare_optim_handle.prepare_criterion()
        else:
            self.criterion = criterion

        prepare_pretrained(self.model, args.pretrained_url, 'differential' if args.training_mode != 'model_soup' else args.soup_mode, self.logger)

        self.lr, self.batch_size = args.lr, args.batch_size
        self.verbose, self.max_epoch, self.only_do_test = args.verbose, args.max_epoch, args.only_do_test
        self.dataset = args.dataset

        self.args = args


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

    def test(self, epoch=0):
        batch_time_m, acc1_m, acc5_m = AverageMeter(), AverageMeter(), AverageMeter()

        pbar = enumerate(self.val_loader)
        if self.verbose:
            self.logger.info(('\n' + '%11s' * 5) % ('Epoch', 'GPU Mem.', 'Time', 'Acc@1', 'Acc@5'), only_print=True)
            pbar = tqdm(pbar, total=len(self.val_loader), unit='batch', unit_scale=True, bar_format=TQDM_BAR_FORMAT)
        self.model.eval()
        with torch.no_grad():
            end = time.time()
            for batch_idx, (input, target) in pbar:
                input, target = input.to(self.device), target.to(self.device)
                outputs = self.model.reuse_callback(
                    self.model(
                        self.model.input_callback(
                            input
                        )
                    )
                )
                acc1, acc5 = accuracy(outputs, target, topk=(1, 5))
                acc1_m.update(acc1.item())
                acc5_m.update(acc5.item())

                batch_time_m.update(time.time() - end)

                end = time.time()
                if self.verbose:
                    mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                    pbar.set_description(('%11s' * 2 + '%11.4g' * 3) %
                                         (f'{epoch + 1}/{self.max_epoch}', mem, batch_time_m.avg, acc1_m.avg, acc5_m.avg))

        self.logger.info(f'***   Best results: [Acc@1: {acc1_m.avg}], [Acc5: {acc5_m.avg}]')

        return acc1_m.avg, acc5_m.avg
