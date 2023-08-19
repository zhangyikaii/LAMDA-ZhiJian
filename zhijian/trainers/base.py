from zhijian.models.utils import get_class_from_module, get_command_line_parser
from zhijian.models.addin.base import addin_config_compile, reuse_keys_config_compile

import os
import yaml
import argparse
import datetime

import sys
import torch
from typing import Tuple
from typing_extensions import Literal

import transformers
from transformers import HfArgumentParser, Seq2SeqTrainingArguments

from zhijian.trainers.llm_config import (
    ModelArguments,
    DataTrainingArguments,
    FinetuningArguments,
)

def get_args(**kwargs):
    args, parser = get_command_line_parser()
    args.__dict__.update(kwargs)
    return prepare_args(args, parser)

def update_args(args, **kwargs):
    args.__dict__.update(kwargs)
    return prepare_args(args)

def prepare_trainer(args, **kwargs):
    return get_class_from_module(f'zhijian.trainers.{args.training_mode}', 'Trainer')(args, **kwargs)

def prepare_llm_args(
    args, stage: Literal["pt", "sft", "rm", "ppo"], logger=None
) -> Tuple[ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments, FinetuningArguments]:

    import datasets
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments, FinetuningArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"): # Provide arguments with a json file.
        model_args, data_args, training_args, finetuning_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, finetuning_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    # Setup logging
    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Check arguments (do not check finetuning_args since it may be loaded from checkpoints)
    data_args.init_for_training(args)

    assert stage == "sft" or (not training_args.predict_with_generate), \
        "`predict_with_generate` cannot be set as True at PT, RM and PPO stages."

    assert not (training_args.do_train and training_args.predict_with_generate), \
        "`predict_with_generate` cannot be set as True while training."

    assert (not training_args.do_predict) or training_args.predict_with_generate, \
        "Please enable `predict_with_generate` to save model predictions."

    assert model_args.quantization_bit is None or finetuning_args.finetuning_type == "lora", \
        "Quantization is only compatible with the LoRA method."

    if model_args.checkpoint_dir is not None:
        if finetuning_args.finetuning_type != "lora":
            assert len(model_args.checkpoint_dir) == 1, "Only LoRA tuning accepts multiple checkpoints."
        else:
            assert model_args.quantization_bit is None or len(model_args.checkpoint_dir) == 1, \
                "Quantized model only accepts a single checkpoint."

    if logger is not None:
        if model_args.quantization_bit is not None and (not training_args.do_train):
            logger.warning("Evaluating model in 4/8-bit mode may cause lower scores.")

        if training_args.do_train and (not training_args.fp16):
            logger.warning("We recommend enable fp16 mixed precision training.")

        if data_args.prompt_template == "default":
            logger.warning("Please specify `prompt_template` if you are using other pre-trained models.")

    if training_args.local_rank != -1 and training_args.ddp_find_unused_parameters is None:
        if logger is not None:
            logger.warning("`ddp_find_unused_parameters` needs to be set as False in DDP training.")
        training_args.ddp_find_unused_parameters = False

    training_args.optim = "adamw_torch" if training_args.optim == "adamw_hf" else training_args.optim # suppress warning

    if model_args.quantization_bit is not None:
        if training_args.fp16:
            model_args.compute_dtype = torch.float16
        elif training_args.bf16:
            model_args.compute_dtype = torch.bfloat16
        else:
            model_args.compute_dtype = torch.float32

    # Log on each process the small summary:
    if logger is not None:
        logger.info(
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}\n"
            + f"  distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
        )
        logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    transformers.set_seed(training_args.seed)

    return model_args, data_args, training_args, finetuning_args

def prepare_args(args, parser=None, update_default=False):
    def merge_b2a_wo_overwrite(b, a):
        for key, value in vars(b).items():
            if not hasattr(a, key):
                setattr(a, key, value)
        return a

    def update_args(cur_parser_fn):
        nonlocal parser, args
        if parser is not None:
            parser = cur_parser_fn(parser)
        else:
            default_parser = argparse.ArgumentParser()
            default_parser = cur_parser_fn(default_parser)
            default_args = default_parser.parse_args([])
            args = merge_b2a_wo_overwrite(default_args, args)

    if (parser is not None or (parser is None and update_default)) and args.model is not None:
        model_base_name = args.model.split('.', 1)[0]
        backbone_parser_fn = get_class_from_module(f'zhijian.models.backbone.{model_base_name}.arguments', 'prepare_specific_parser')
        if backbone_parser_fn is not None:
            update_args(backbone_parser_fn)

    yaml_params = {}
    if hasattr(args, 'config') and os.path.isfile(args.config):
        with open(args.config, 'r') as f:
            yaml_params = yaml.safe_load(f)

    args.addins = yaml_params.get('addins', [])

    if args.config_blitz is not None and not args.addins:
        cur_config_blitz = [i.strip() for i in args.config_blitz.split(',')]
        for cur_s in cur_config_blitz:
            args.addins.extend(addin_config_compile(cur_s))

    if parser is not None or (parser is None and update_default):
        flag_first_prepare_parser = {}
        for cur_addin in args.addins:
            cur_addin_name = cur_addin['name']
            if cur_addin_name in flag_first_prepare_parser.keys():
                continue

            addin_parser_fn = get_class_from_module(f'zhijian.models.addin.module.{cur_addin_name.lower()}', 'prepare_specific_addin_parser')
            if addin_parser_fn is not None:
                flag_first_prepare_parser[cur_addin_name] = True
                update_args(addin_parser_fn)

        trainer_parser_fn = get_class_from_module(f'zhijian.trainers.{args.training_mode}', 'prepare_specific_trainer_parser')
        update_args(trainer_parser_fn)

    if parser is not None:
        temp_args_dict = args.__dict__
        args, unknown_args = parser.parse_known_args()
        args.__dict__.update(temp_args_dict)

    params_overwrite_from_yaml = []
    for k, v in yaml_params.items():
        if hasattr(args, k):
            params_overwrite_from_yaml.append(k)
        setattr(args, k, v)

    if hasattr(args, 'reuse_keys_blitz') and args.reuse_keys_blitz is not None:
        if not hasattr(args, 'reuse_keys') or args.reuse_keys is None:
            args.reuse_keys = []
        cur_reuse_keys_blitz = [i.strip() for i in args.reuse_keys_blitz.split(',')]
        for cur_s in cur_reuse_keys_blitz:
            args.reuse_keys.extend(reuse_keys_config_compile(cur_s))
    if args.only_do_test:
        args.reuse_keys = []
        args.max_epoch = 1

    if not hasattr(args, 'time_str') or args.time_str == '':
        args.time_str = datetime.datetime.now().strftime('%m%d-%H-%M-%S-%f')[:-3]

    if args.training_mode in ['knowledge_distillation', 'regularization']:
        with open(args.t_config, "r") as t_f:
            t_yaml_params = yaml.safe_load(t_f)
 
        args.t_args = argparse.Namespace()
        for k, v in t_yaml_params.items():
            setattr(args.t_args, k, v)            
        for k in ['dataset']:
            if k not in t_yaml_params.keys():
                setattr(args.t_args, k, getattr(args, k))

    if not hasattr(args, 'pretrained_url'):
        args.pretrained_url = []

    if 'llm' in args.training_mode:
        args.model_args, args.data_args, args.training_args, args.finetuning_args = prepare_llm_args(args, stage="sft")
        args.training_args._n_gpu = 1
        args.training_args.ddp_find_unused_parameters = None
        args.training_args.dataset = args.dataset
        args.training_args.dataset_dir = args.dataset_dir

    return args
