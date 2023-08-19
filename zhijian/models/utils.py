
import os
import os.path as osp
from pathlib import Path

import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.backends.cudnn as cudnn


from argparse import ArgumentParser

import logging
from logging.config import dictConfig

from collections import defaultdict, OrderedDict

import numpy as np
import time
import random
import argparse

import json
from enum import Enum

import hashlib

import importlib

from torchvision.datasets.folder import default_loader

from dataclasses import asdict

from zhijian.trainers.llm_config import ModelArguments, DataTrainingArguments, FinetuningArguments
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments

class ProtoAverageMeter(object):
    def __init__(self):
        self.avg = None
        self.count = 0

    def update(self, val):
        if self.count == 0:
            self.avg = torch.mean(val, dim=0)
        else:
            self.avg = torch.sum(torch.cat([(self.avg.unsqueeze(0) * self.count), val]), dim=0) / (self.count + len(val))
        self.count += len(val)


def pairwise_metric(x, y, matching_fn, temperature=1, is_distance=True):
    n_x = x.shape[0]
    n_y = y.shape[0]

    if matching_fn == 'euclidean':
        result_metric = -(
            x.unsqueeze(1).expand(n_x, n_y, *x.shape[1:]) -
            y.unsqueeze(0).expand(n_x, n_y, *x.shape[1:])
        ).pow(2).sum(dim=-1)
        if is_distance:
            result_metric = -result_metric

    elif matching_fn == 'cosine':
        EPSILON = 1e-8
        normalised_x = x / (x.pow(2).sum(dim=-1, keepdim=True).sqrt() + EPSILON)
        normalised_y = y / (y.pow(2).sum(dim=-1, keepdim=True).sqrt() + EPSILON)

        expanded_x = normalised_x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = normalised_y.unsqueeze(0).expand(n_x, n_y, -1)

        result_metric = (expanded_x * expanded_y).sum(dim=-1)
        if is_distance:
            result_metric = 1 - result_metric

    elif matching_fn == 'dot':
        expanded_x = x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = y.unsqueeze(0).expand(n_x, n_y, -1)

        result_metric = (expanded_x * expanded_y).sum(dim=2)
        if is_distance:
            result_metric = -result_metric

    else:
        raise ValueError('Unsupported similarity function')

    return result_metric / temperature


def save_pickle(file_name, data):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.DEFAULT_PROTOCOL)

def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


class MyImageFolderDataset(Dataset):
    def __init__(self, samples, transform):
        super().__init__()
        self.transform = transform
        self.samples = samples

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = default_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def __len__(self):
        return len(self.samples)



def gpu_state(gpu_id, get_return=False, logger=None):
    qargs = ['index', 'gpu_name', 'memory.used', 'memory.total']
    cmd = 'nvidia-smi --query-gpu={} --format=csv,noheader'.format(','.join(qargs))

    results = os.popen(cmd).readlines()
    gpu_id_list = gpu_id.split(",")
    gpu_space_available = {}
    for cur_state in results:
        cur_state = cur_state.strip().split(", ")
        for i in gpu_id_list:
            if i == cur_state[0]:
                if not get_return:
                    if logger is not None:
                        logger.info(f'GPU {i} {cur_state[1]}: Memory-Usage {cur_state[2]} / {cur_state[3]}.')
                else:
                    gpu_space_available[i] = int("".join(list(filter(str.isdigit, cur_state[3])))) - int("".join(list(filter(str.isdigit, cur_state[2]))))
    if get_return:
        return gpu_space_available


def set_gpu(x, space_hold=1000):
    assert torch.cuda.is_available()
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    torch.backends.cudnn.benchmark = True
    gpu_available = 0
    while gpu_available < space_hold:
        gpu_space_available = gpu_state(x, get_return=True)
        for gpu_id, space in gpu_space_available.items():
            gpu_available += space
        if gpu_available < space_hold:
            gpu_available = 0
            raise Exception
            time.sleep(1800)
    gpu_state(x)


def set_seed(seed):
    np.random.seed(seed=seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def get_command_line_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='VTAB-1k.CIFAR-100')
    parser.add_argument('--dataset-dir', type=str, default=None)

    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--config', type=str, default='')
    parser.add_argument('--config-blitz', type=str, default=None)
    parser.add_argument('--pretrained-url', nargs='*', default=[])
    parser.add_argument('--training-mode', type=str, default='finetune')
    parser.add_argument('--reuse-keys', nargs='*', default=None)
    parser.add_argument('--reuse-keys-blitz', type=str, default=None)

    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--max-epoch', type=int, default=30)

    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--wd', type=float, default=1e-5)
    parser.add_argument('--mom', type=float, default=0.9)

    parser.add_argument('--lr-scheduler', type=str, default='cosine')
    parser.add_argument('--eta-min', type=float, default=0)

    parser.add_argument('--criterion', type=str, default='cross-entropy')

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=str, default='3')
    parser.add_argument('--num-workers', type=int, default=8)


    parser.add_argument('--test-interval', type=int, default=30)
    parser.add_argument('--log-url', type=str, default='your/log/directory')
    parser.add_argument('--time-str', type=str, default='')
    parser.add_argument('--verbose', action='store_true', default=False)

    parser.add_argument('--only-do-test', action='store_true', default=False)
    parser.add_argument('--alchemy', action='store_true', default=False)


    args, _ = parser.parse_known_args()
    parser = ArgumentParser(parents=[parser], add_help=False)

    return args, parser


import pprint
_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)




class ConfigEncoder(json.JSONEncoder):
    def default(self, o):
        encode_classes = (DataTrainingArguments, ModelArguments, FinetuningArguments, Seq2SeqTrainingArguments)
        if isinstance(o, encode_classes):
            return asdict(o)
        elif isinstance(o, type):
            return {'$class': o.__module__ + "." + o.__name__}
        elif isinstance(o, Enum):
            return {
                '$enum': o.__module__ + "." + o.__class__.__name__ + '.' + o.name
            }
        elif isinstance(o, argparse.Namespace):
            return vars(o)
        elif callable(o):
            return {
                '$function': o.__module__ + "." + o.__name__
            }
        return json.JSONEncoder.default(self, o)


def listdir_nohidden(path, sort=False):
    """List non-hidden items in a directory.

    Args:
         path (str): directory path.
         sort (bool): sort the items.
    """
    items = [f for f in os.listdir(path) if not f.startswith(".")]
    if sort:
        items.sort()
    return items



def default_trans(x):
    return x
def l1n(x):
    return F.normalize(x, p=1, dim=1, eps=1e-5)
def l2n(x):
    return F.normalize(x, p=2, dim=1, eps=1e-5)
def z_score(x):
    cur_std = torch.std(x, dim=0, unbiased=False, keepdim=True)
    return (x - torch.mean(x, dim=0, keepdim=True)) / (cur_std + 1e-5)
def tky(x):
    beta = 1 / 3
    flip_flag = False
    if beta < 1:
        sign_x = torch.sign(x)
        x = torch.mul(x, sign_x)
        flip_flag = True
    if beta == 0:
        x = torch.log(x)
    else:
        x = torch.pow(x, beta)
    if flip_flag:
        x = torch.mul(x, sign_x)
    return x

def boxcox(x):
    lmbda = 1 / 3
    flip_flag = False
    if lmbda < 1:
        sign_x = torch.sign(x)
        x = torch.mul(x, sign_x)
        flip_flag = True
    if lmbda == 0:
        x = torch.log(x)
    else:
        x = (torch.pow(x, lmbda) - 1) / lmbda
    if flip_flag:
        x = torch.mul(x, sign_x)
    return x


def avg_pooling(x):
    return F.adaptive_avg_pool2d(x, 1)

def max_pooling(x):
    return F.adaptive_max_pool2d(x, 1)


SHOT_TRANSFORMS = {
    'default': default_trans,
    'l1n': l1n,
    'l2n': l2n,
    'tky': tky,
    'z_score': z_score,
    'boxcox': boxcox,
    'avg_pooling': avg_pooling,
    'max_pooling': max_pooling
    }


def get_judge_list(model):
    judge_list = {'module': hasattr(model, 'module')}
    if judge_list['module']:
        judge_list.update({
            'forward_features': hasattr(model.module, 'forward_features'),
            'get_intermediate_layers': hasattr(model.module, 'get_intermediate_layers'),
            'forward_return_n_last_blocks': hasattr(model.module, 'forward_return_n_last_blocks')
            })
    else:
        judge_list.update({
            'forward_features': hasattr(model, 'forward_features'),
            'get_intermediate_layers': hasattr(model, 'get_intermediate_layers'),
            'forward_return_n_last_blocks': hasattr(model, 'forward_return_n_last_blocks')
            })
    return judge_list


def _pil_interp(method):
    from torchvision.transforms import InterpolationMode
    if method == 'bicubic':
        return InterpolationMode.BICUBIC
    elif method == 'lanczos':
        return InterpolationMode.LANCZOS
    elif method == 'hamming':
        return InterpolationMode.HAMMING
    else:
        # default bilinear, do we want to allow nearest?
        return InterpolationMode.BILINEAR

def init_device(args):
    set_gpu(args.gpu)
    set_seed(args.seed)
    torch.cuda.set_device(int(args.gpu))
    torch.autograd.set_detect_anomaly(True)


def compute_md5(file_name):
    hash_md5 = hashlib.md5()
    with open(file_name, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class BestInfo(object):
    def __init__(self, args, **kwargs):
        self.val_best_acc1 = 0 if 'val_best_acc1' not in kwargs.keys() else kwargs['val_best_acc1']
        self.val_best_acc5 = 0 if 'val_best_acc5' not in kwargs.keys() else kwargs['val_best_acc5']
        self.val_best_epoch = 0 if 'val_best_epoch' not in kwargs.keys() else kwargs['val_best_epoch']
        self.args = args

    def update(self, **kwargs):
        for k in kwargs.keys():
            if kwargs[k] >= getattr(self, k):
                setattr(self, k, kwargs[k])

    def __str__(self):
        return f'{self.args.model},{self.args.config},{self.args.dataset},{self.args.training_mode},{self.args.optimizer},{self.args.lr},{self.args.wd},{self.args.log_url},{self.val_best_epoch},{self.val_best_acc1},{self.val_best_acc5}'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = min(max(topk), target.max() + 1)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            if correct.shape[0] < k:
                res.append(torch.tensor(0))
            else:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append((correct_k.mul_(100.0 / batch_size)))

        return res


class PrepareFunc(object):
    def __init__(self, args):
        self.args = args

    def prepare_optimizer(self, model):
        def set_optimizer(cur_type, cur_encoder):
            if cur_type == 'adam':
                return optim.Adam(
                    cur_encoder.parameters(),
                    lr=self.args.lr,
                    weight_decay=self.args.wd
                    )
            elif cur_type == 'sgd':
                return optim.SGD(
                    cur_encoder.parameters(),
                    lr=self.args.lr,
                    momentum=self.args.mom,
                    weight_decay=self.args.wd,
                    nesterov = True
                    )
            elif cur_type == 'adamw':
                return optim.AdamW(
                    cur_encoder.parameters(),
                    lr=self.args.lr,
                    weight_decay=self.args.wd
                    )

        optimizer = set_optimizer(self.args.optimizer, model)

        def set_lr_scheduler(cur_type, optmz):
            if cur_type == 'step':
                return optim.lr_scheduler.StepLR(
                    optmz,
                    step_size=int(self.args.step_size),
                    gamma=self.args.gamma
                    )
            elif cur_type == 'multistep':
                return optim.lr_scheduler.MultiStepLR(
                    optmz,
                    milestones=[int(_) for _ in self.args.step_size.split(',')],
                    gamma=self.args.gamma,
                    )
            elif cur_type == 'cosine':
                return optim.lr_scheduler.CosineAnnealingLR(
                    optmz,
                    self.args.max_epoch,
                    eta_min=self.args.eta_min   # a tuning parameter
                    )
            elif cur_type == 'plateau':
                return optim.lr_scheduler.ReduceLROnPlateau(
                    optmz,
                    mode='min',
                    factor=self.args.gamma,
                    patience=5
                    )
            else:
                raise ValueError('No Such Scheduler')

        lr_scheduler = set_lr_scheduler(self.args.lr_scheduler, optimizer)

        return optimizer, lr_scheduler

    def prepare_criterion(self, **kwargs):
        if self.args.criterion == 'cross-entropy':
            return nn.CrossEntropyLoss(**kwargs)
        elif self.args.criterion == 'nll':
            return nn.NLLLoss(**kwargs)
        elif self.args.criterion == 'mse':
            return nn.MSELoss(**kwargs)
        elif self.args.criterion == 'multi-label-soft-margin':
            return nn.MultiLabelSoftMarginLoss(**kwargs)
        else:
            raise NotImplementedError


class Logger(object):
    def __init__(self, args, log_dir, level, **kwargs):
        from tensorboardX import SummaryWriter
        self.logger_path = osp.join(log_dir, 'scalars.json')
        self.tb_logger = SummaryWriter(
            logdir=osp.join(log_dir, 'tflogger'),
            **kwargs,
        )
        self.scalars = defaultdict(OrderedDict)

        self.log_config(vars(args))

        self.set_logging(level, log_dir)
        logging.info(f'Log files are recorded in: {log_dir}')

    def add_scalar(self, key, value, counter):
        assert self.scalars[key].get(counter, None) is None, 'counter should be distinct'
        self.scalars[key][counter] = value
        self.tb_logger.add_scalar(key, value, counter)

    def log_config(self, variant_data):
        config_filepath = osp.join(osp.dirname(self.logger_path), 'configs.json')
        with open(config_filepath, "w") as fd:
            json.dump(variant_data, fd, indent=2, sort_keys=True, cls=ConfigEncoder)

    def dump(self):
        with open(self.logger_path, 'w') as fd:
            json.dump(self.scalars, fd, indent=2)

    def set_logging(self, level, work_dir):
        log_file = os.path.join(work_dir if work_dir is not None else ".", 'train.log')
        LOGGING = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                },
                "simple": {
                    "format": "%(message)s"
                },
            },
            "handlers": {
                "console": {
                    "level": level,
                    "class": "logging.StreamHandler",
                    "formatter": "simple",
                },
                'file': {
                    'level': level,
                    'formatter': 'detailed',
                    'class': 'logging.FileHandler',
                    'filename': log_file,
                    'mode': 'a',
                },
            },
            "loggers": {
                "": {
                    "level": level,
                    "handlers": ["console", "file"] if work_dir is not None else ["console"],
                },
            },
        }
        dictConfig(LOGGING)
        logging.info(f"Log level set to: {level}")

    def warning(self, message):
        logging.warning(message)

    def info(self, message):
        logging.info(message)



class LogHandle(object):
    def __init__(self, args):
        self.log_url = args.log_url
        self.time_str = args.time_str
        self.save_path = Path(os.path.join(args.log_url, args.time_str))
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.logger = Logger(args, self.save_path, 'INFO')

        self.args = args

    def save_model(self, model, optimizer, epoch, save_file='best.pt'):
        state = {
            'model': model.state_dict(),
            'addin': model.addin.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'time_str': self.args.time_str
        }
        torch.save(state, os.path.join(self.save_path, save_file))

    def warning(self, message):
        self.logger.warning(message)

    def info(self, message, only_print=False, end='\n'):
        if only_print:
            print(message, end=end)
        else:
            self.logger.info(message)

    def log_per_epoch(self, epoch, contents, save_file):
        with open(os.path.join(self.save_path, save_file), 'a') as f:
            for i in contents:
                f.write(f'{epoch},{i}\n')

    def log_one_line(self, content, save_csv_file):
        with open(os.path.join(self.log_url, save_csv_file), 'a') as f:
            f.write(f'{self.time_str},{content}\n')
    
    def log_pt(self, data, save_pt_file):
        torch.save(data, os.path.join(self.save_path, save_pt_file))

    def add_scalar(self, key, value, counter):
        self.logger.add_scalar(key, value, counter)

    def log_pickle(self, contents, save_file):
        save_pickle(os.path.join(self.save_path, save_file), contents)


def nan_assert(x):
    assert torch.any(torch.isnan(x)) == False


class OnlineDict(object):
    def __init__(self, pkl_file_name):
        self.data = load_pickle(pkl_file_name) if os.path.isfile(pkl_file_name) else {}
        self.pkl_file_name = pkl_file_name
    def update(self, k, v):
        if k not in self.data.keys():
            self.data[k] = v
            save_pickle(self.pkl_file_name, self.data)
    def add(self, v):
        k = f'c{len(self.data)}'
        self.update(k, v)
        return k
    def get(self, k):
        return self.data.get(k, None)

def dict2args(d):
    cur_args = argparse.Namespace()
    cur_args.__dict__.update(d)
    return cur_args

def safe_update(d, d_prime):
    for k, v in d_prime.items():
        if k not in d.keys():
            d.update({k: v})

def get_class_from_module(module_name, class_name):
    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except ModuleNotFoundError:
        return None

def select_from_input(prompt_for_select, valid_selections):
    selections2print = '\n\t'.join([f'[{idx + 1}] {i}' for idx, i in enumerate(valid_selections)])
    while True:
        selected = input(f"Please input a {prompt_for_select}, type 'help' to show the options: ")

        if selected == 'help':
            print(f"Available {prompt_for_select}(s):\n\t{selections2print}")
        elif selected.isdigit() and int(selected) >= 1 and int(selected) <= len(valid_selections):
            selected = valid_selections[int(selected) - 1]
            break
        elif selected in valid_selections:
            break
        else:
            print("Sorry, input not support.")
            print(f"Available {prompt_for_select}(s):\n\t{selections2print}")

    return selected