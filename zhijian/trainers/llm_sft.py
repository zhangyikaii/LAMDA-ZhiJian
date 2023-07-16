from zhijian.models.backbone.base import prepare_llm, prepare_hook, prepare_gradient, prepare_cuda, prepare_pretrained
from zhijian.models.addin.base import prepare_addins
from zhijian.data.base import prepare_llm_dataset, preprocess_llm_dataset, DynamicDataCollatorWithPadding
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


class Trainer(object):
    def __init__(
        self, args,
        model=None,
        tokenizer=None,
        dataset=None,
        data_collator=None,
        val_loader=None,
        num_classes=None,
        optimizer=None,
        lr_scheduler=None,
        criterion=None,
        device=None
        ):
        set_seed(args.seed)
        self.logger = LogHandle(args)

        if dataset is None:
            self.dataset = prepare_llm_dataset(args.model_args, args.data_args)
        if None in [model, tokenizer]:
            self.model, self.tokenizer = prepare_llm(args.model_args, args.finetuning_args, not args.only_do_test, stage="sft")
            self.dataset = preprocess_llm_dataset(self.dataset, self.tokenizer, args.data_args, args.training_args, stage="sft")
            self.data_collator = DynamicDataCollatorWithPadding(
                tokenizer=tokenizer,
                ignore_pad_token_for_loss=(args.data_args.ignore_pad_token_for_loss and not args.training_args.predict_with_generate)
            )

        # Override the decoding parameters of Seq2SeqTrainer
        args.training_args.generation_max_length = args.training_args.generation_max_length if \
                    args.training_args.generation_max_length is not None else args.data_args.max_target_length
        args.training_args.generation_num_beams = args.data_args.eval_num_beams if \
                    args.data_args.eval_num_beams is not None else args.training_args.generation_num_beams

        # Split the dataset
        if args.training_args.do_train:
            if args.data_args.dev_ratio > 1e-6:
                dataset = dataset.train_test_split(test_size=args.data_args.dev_ratio)
                trainer_kwargs = {"train_dataset": dataset["train"], "eval_dataset": dataset["test"]}
            else:
                trainer_kwargs = {"train_dataset": dataset}
        else: # do_eval or do_predict
            trainer_kwargs = {"eval_dataset": dataset}
