from zhijian.models.backbone.base import prepare_model, prepare_hook, prepare_cuda, prepare_pretrained
from zhijian.models.addin.base import prepare_addins
from zhijian.models.configs.base import TQDM_BAR_FORMAT

from zhijian.models.utils import AverageMeter
from zhijian.trainers.base import prepare_args
from zhijian.trainers.finetune import Trainer as Base_Trainer
from zhijian.trainers.nearest_class_mean import compute_prototypes, compute_logits
from zhijian.models.kd.base import prepare_kd_loss
from zhijian.models.backbone.timm.hooks import HOOKS

from copy import deepcopy

import time
import torch
from contextlib import suppress

from tqdm import tqdm

def prepare_specific_trainer_parser(parser):
    parser.add_argument('--t-config', type=str, default=None)
    parser.add_argument('--kd-mode', type=str, default='st')
    parser.add_argument('--kd-lambda', type=float, default=1)
    parser.add_argument('--temperature', type=float, default=1)
    return parser

class TwinPTMTrainer(Base_Trainer):
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

        args.t_args = prepare_args(args.t_args, update_default=True)
        self.t_model, t_model_args = prepare_model(args.t_args, self.logger)
        self.t_addins, self.t_fixed_params = prepare_addins(args.t_args, t_model_args)
        prepare_hook(args.t_args.addins, self.t_addins, self.t_model, 'addin')
        prepare_pretrained(self.t_model, args.t_args.pretrained_url, self.logger)


class Trainer(TwinPTMTrainer):
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
        cur_t_model_hook = HOOKS[args.t_args.model.replace('timm.', '')]()
        prepare_hook(cur_t_model_hook.get_feature_hook_info(), [cur_t_model_hook], self.t_model, 'hook')
        cur_model_hook = HOOKS[args.model.replace('timm.', '')]()
        prepare_hook(cur_model_hook.get_feature_hook_info(), [cur_model_hook], self.model, 'hook')
        for param in self.t_model.parameters():
            param.requires_grad = False
        self.t_device = prepare_cuda(self.t_model)

        self.t_model.eval()

        if criterion is not None:
            self.kd_criterion = criterion
        else:
            self.kd_criterion = prepare_kd_loss(self.args)

        self.kd_mode = args.kd_mode
        self.kd_lambda = args.kd_lambda
        self.t_training_mode = args.t_args.training_mode
        if self.t_training_mode == 'nearest_class_mean':
            self.t_shot_transform = args.t_args.shot_transform
            self.t_metric = args.t_args.metric
            self.t_temperature = args.t_args.temperature

    def fit(self):
        if self.only_do_test:
            return
        best_val_acc1, best_epoch = 0, 0
        if self.t_training_mode == 'nearest_class_mean':
            self.t_prototypes = compute_prototypes(self.model, self.train_loader, self.num_classes, self.device, self.verbose, self.logger)

        if self.kd_mode in ['refilled']:
            for epoch in range(self.max_epoch):
                self.logger.info(f'{epoch}th epoch warmup start: \n')
                self.model.train()
                pbar = enumerate(self.train_loader)
                for batch_idx, (input, target) in pbar:
                    input, target = input.to(self.device), target.to(self.device)
                    with suppress():
                        out_s = self.model.reuse_callback(self.model(self.model.input_callback(input)))
                        out_t = self.model.reuse_callback(self.t_model(self.t_model.input_callback(input)))
                        feat_s = self.model.hook[0].get_feature()
                        feat_t = self.t_model.hook[0].get_feature()
                        kd_loss_stage1 = self.kd_criterion[0](feat_s, feat_t, target)
                    self.optimizer.zero_grad()
                    kd_loss_stage1.backward()
                    self.optimizer.step()
                self.lr_scheduler.step()
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
                    out_s = self.model.reuse_callback(self.model(self.model.input_callback(input)))
                    feat_s = self.model.hook[0].get_feature()
                    out_t = self.t_model(input)
                    feat_t = self.t_model.hook[0].get_feature()
                    if self.t_training_mode == 'nearest_class_mean':
                        out_t = compute_logits(feat_t, self.t_prototypes, self.t_shot_transform, self.t_metric, self.t_temperature)
                    stem_t, rb1_t, rb2_t, rb3_t = None, None, None, None
                    stem_s, rb1_s, rb2_s, rb3_s = None, None, None, None
                    cls_loss = self.criterion(out_s, target)

                    if self.kd_mode in ['logits', 'st']:
                        kd_loss = self.kd_criterion(out_s, out_t.detach()) * self.kd_lambda
                    elif self.kd_mode in ['nst']:
                        kd_loss = self.kd_criterion(rb3_s[1], rb3_t[1].detach()) * self.kd_lambda
                    elif self.kd_mode in ['at', 'sp']:
                        kd_loss = (
                            self.kd_criterion(rb1_s[1], rb1_t[1].detach()) +
                            self.kd_criterion(rb2_s[1], rb2_t[1].detach()) +
                            self.kd_criterion(rb3_s[1], rb3_t[1].detach())
                            ) / 3.0 * self.kd_lambda
                    elif self.kd_mode in ['pkt', 'rkd', 'cc', 'fitnet']:
                        kd_loss = self.kd_criterion(feat_s, feat_t.detach()) * self.kd_lambda
                    elif self.kd_mode in ['fsp']:
                        kd_loss = (
                            self.kd_criterion(stem_s[1], rb1_s[1], stem_t[1].detach(), rb1_t[1].detach()) +
                            self.kd_criterion(rb1_s[1],  rb2_s[1], rb1_t[1].detach(),  rb2_t[1].detach()) +
                            self.kd_criterion(rb2_s[1],  rb3_s[1], rb2_t[1].detach(),  rb3_t[1].detach())
                            ) / 3.0 * self.kd_lambda
                    elif self.kd_mode in ['ab']:
                        kd_loss = (
                            self.kd_criterion(rb1_s[0], rb1_t[0].detach()) +
                            self.kd_criterion(rb2_s[0], rb2_t[0].detach()) +
                            self.kd_criterion(rb3_s[0], rb3_t[0].detach())
                            ) / 3.0 * self.kd_lambda
                    elif self.kd_mode in ['sobolev']:
                        kd_loss = self.kd_criterion(out_s, out_t, input, target) * self.kd_lambda
                    elif self.kd_mode in ['lwm']:
                        kd_loss = self.kd_criterion(out_s, rb2_s[1], out_t, rb2_t[1], target) * self.kd_lambda
                    elif self.kd_mode in ['irg']:
                        kd_loss = self.kd_criterion(
                            [rb2_s[1], rb3_s[1], feat_s, out_s],
                            [rb2_t[1].detach(), rb3_t[1].detach(), feat_t.detach(), out_t.detach()]
                            ) * self.kd_lambda
                    elif self.kd_mode in ['vid', 'afd']:
                        kd_loss = (
                            self.kd_criterion[1](rb1_s[1], rb1_t[1].detach()) +
                            self.kd_criterion[2](rb2_s[1], rb2_t[1].detach()) +
                            self.kd_criterion[3](rb3_s[1], rb3_t[1].detach())
                            ) / 3.0 * self.kd_lambda
                    elif self.kd_mode in ['ofd']:
                        kd_loss = (
                            self.kd_criterion[1](rb1_s[0], rb1_t[0].detach()) +
                            self.kd_criterion[2](rb2_s[0], rb2_t[0].detach()) +
                            self.kd_criterion[3](rb3_s[0], rb3_t[0].detach())
                            ) / 3.0 * self.kd_lambda
                    elif self.kd_mode in ['refilled']:
                        kd_loss = self.kd_criterion[1](out_s, out_t, self.num_classes, target)
                    elif self.kd_mode == 'customized':
                        kd_loss = self.kd_criterion(out_s, out_t)
                    else:
                        raise Exception('Invalid kd mode...')
                    loss = cls_loss + kd_loss

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
