import torch

from tqdm import tqdm

from zhijian.models.configs.base import TQDM_BAR_FORMAT
from zhijian.models.utils import ProtoAverageMeter, AverageMeter, accuracy, pairwise_metric, SHOT_TRANSFORMS
from zhijian.models.backbone.timm.hooks import HOOKS
from zhijian.models.backbone.base import prepare_hook
from zhijian.trainers.finetune import Trainer as Base_Trainer

import time
import torch

from tqdm import tqdm

def prepare_specific_trainer_parser(parser):
    parser.add_argument('--shot-transform', type=str, default='l2n')
    parser.add_argument('--metric', type=str, default='euclidean')
    parser.add_argument('--temperature', type=float, default=1)
    return parser

def compute_logits(features, prototypes, shot_transform='l2n', metric='euclidean', temperature=1):
    cur_logits = pairwise_metric(
        x=SHOT_TRANSFORMS[shot_transform](features),
        y=SHOT_TRANSFORMS[shot_transform](prototypes),
        matching_fn=metric,
        temperature=temperature,
        is_distance=False
        )
    return cur_logits


def compute_prototypes(model, data_loader, num_classes, device, verbose=False, logger=None):
    prototypes = [ProtoAverageMeter() for _ in range(num_classes)]
    pbar = enumerate(data_loader)

    if logger is not None:
        logger.info('Computing class means..', only_print=True)
    if verbose:
        pbar = tqdm(pbar, total=len(data_loader), unit='batch', unit_scale=True, bar_format=TQDM_BAR_FORMAT)

    model.eval()
    with torch.no_grad():
        for idx, (input, target) in pbar:
            input = input.to(device)
            target = target.to(device)

            target_indices = {}
            for i_class in range(num_classes):
                target_indices[i_class] = (target == i_class)

            _ = model(input)
            features = model.hook[0].get_feature()

            for i_class in range(num_classes):
                if torch.sum(target_indices[i_class]) != 0:
                    prototypes[i_class].update(features[target_indices[i_class]].detach().cpu())

    for i in range(len(prototypes)):
        if prototypes[i].avg is None:
            raise Exception

    return torch.stack([i.avg for i in prototypes]).to(device)


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
        cur_ncm_hook = HOOKS[args.model.replace('timm.', '')]()
        prepare_hook(cur_ncm_hook.get_feature_hook_info(), [cur_ncm_hook], self.model, 'hook')

        
        self.shot_transform = self.args.shot_transform
        self.metric = self.args.metric
        self.temperature = self.args.temperature

    def fit(self):
        self.prototypes = compute_prototypes(self.model, self.train_loader, self.num_classes, self.device, self.verbose, self.logger)
        self.test()

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

                _ = self.model.reuse_callback(
                    self.model(
                        self.model.input_callback(
                            input
                        )
                    )
                )
                features = self.model.hook[0].get_feature()
                outputs = compute_logits(features, self.prototypes, self.shot_transform, self.metric, self.temperature)

                acc1, acc5 = accuracy(outputs, target, topk=(1, 5))
                acc1_m.update(acc1.item())
                acc5_m.update(acc5.item())

                batch_time_m.update(time.time() - end)

                end = time.time()
                if self.verbose:
                    mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                    pbar.set_description(('%11s' * 2 + '%11.4g' * 3) %
                                         (f'{epoch + 1}/{self.max_epoch}', mem, batch_time_m.avg, acc1_m.avg, acc5_m.avg))


        is_updated = False

        if acc1_m.avg >= self.results.val_best_acc1:
            self.results.update(
                val_best_acc1=acc1_m.avg,
                val_best_acc5=acc5_m.avg,
                val_best_epoch=epoch
                )
            is_updated = True

        self.logger.info(f'***   Epoch {epoch + 1} results: [Acc@1: {acc1_m.avg}], [Acc@5: {acc5_m.avg}]')

        if self.alchemy:
            self.logger.log_one_line(str(self.results), 'results.csv')
        return is_updated
