import torch
import torch.nn as nn

from zhijian.models.utils import dict2args, get_class_from_module, safe_update
from zhijian.models.configs.base import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from zhijian.models.backbone.wrapper import ModelWrapper

from zhijian.data.config import DATASET2NUM_CLASSES

from collections import defaultdict, OrderedDict

from copy import deepcopy

import os
from zhijian.trainers.llm_config import FinetuningArguments
from transformers import AutoTokenizer, AutoConfig, BitsAndBytesConfig, AutoModelForCausalLM
from typing import Literal, Optional, Tuple
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils.versions import require_version


def get_module_from_location(model, locations):
    cur_module = model
    for l in locations:
        if isinstance(l, str):
            if ':' in l:
                bef_idx, aft_idx = int(l.split(':')[0]), int(l.split(':')[1])
                cur_module = cur_module[bef_idx:aft_idx]
            else:
                cur_module = getattr(cur_module, l)
        elif isinstance(l, int):
            cur_module = cur_module[l]
        else:
            raise NotImplementedError
    return cur_module


def prepare_gradient(reuse_keys, model, logger=None):
    cur_reuse_keys, cur_reuse_modules = [], []
    if reuse_keys:
        for i_loc in reuse_keys:            
            cur_reuse_modules.append('.'.join([f'{i}' for i in i_loc]))
            cur_module = get_module_from_location(model, i_loc)
            cur_reuse_keys.extend(['.'.join([f'{i}' for i in i_loc]) + f'.{i}' for i, _ in cur_module.named_parameters()])
    elif isinstance(reuse_keys, list) and len(reuse_keys) == 0:
        if logger is not None:
            logger.info(f"All parameters are frozen")
    else:
        if logger is not None:
            logger.info(f"All parameters are fine-tunable")
        cur_reuse_keys.extend([i for i, _ in model.named_parameters()])
        cur_reuse_modules = [i for i, _ in model.named_children()]

    if logger is not None and cur_reuse_keys:
        logger.info(f"Reuse parameters: [{', '.join(cur_reuse_keys)}]")

    model.reuse_keys = cur_reuse_keys
    model.reuse_modules = cur_reuse_modules

    for k, p in model.named_parameters():
        if k in model.reuse_keys:
            continue
        p.requires_grad = False
    model = prepare_train_fn(model)


def prepare_cuda(model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    return device


def prepare_train_fn(model):
    def core(self, mode=True):
        # set train status for this class: disable all but the prompt-related modules
        if mode:
            # training:
            for name, module in self.named_modules():
                if any(i in name for i in self.reuse_modules):
                    module.train()
                else:
                    module.eval()
        else:
            # eval:
            for module in self.children():
                module.eval()
    import types
    model.train = types.MethodType(core, model) # Monkey Patch
    return model


def prepare_hook(addin_info, addins, model, module_name):
    setattr(model, module_name, nn.ModuleList())

    assert len(addins) == len(addin_info)

    for idx, (cur_addin, cur_info) in enumerate(zip(addins, addin_info)):
        assert cur_info['name'] == cur_addin.__class__.__name__
        assert len(cur_info['location']) == len(cur_info['hook'])
        cur_module = getattr(model, module_name)
        cur_module.append(cur_addin)

        for i_loc, i_hook in zip(cur_info['location'], cur_info['hook']):
            add_layer = get_module_from_location(model, i_loc)
            cur_hook = getattr(cur_module[-1], i_hook[0])
            if i_hook[1] == 'pre':
                add_layer.register_forward_pre_hook(cur_hook)
            elif i_hook[1] == 'post':
                add_layer.register_forward_hook(cur_hook)
            else:
                raise NotImplementedError


def prepare_pretrained(model, pretrained_urls, logger=None):
    updated_keys = defaultdict(set)
    model_dict = model.state_dict()
    num_pretrained = len(pretrained_urls)
    if num_pretrained == 0:
        return
    for idx, cur_pretrained_url in enumerate(pretrained_urls):
        pretrained_dict = torch.load(cur_pretrained_url, map_location='cpu')
        if isinstance(pretrained_dict, dict):
            keys_to_check = ['model', 'state_dict', 'model_state_dict']
            matching_key = next((key for key in keys_to_check if key in pretrained_dict), None)
            if matching_key is not None:
                pretrained_dict = pretrained_dict[matching_key]
        
        pretrained_dict = {k: v.to(next(model.parameters()).device) for k, v in pretrained_dict.items()}

        def _fine_tune_params_dict(cur_d):
            ret_d = deepcopy(model_dict)
            for k_idx, k in enumerate(model_dict.keys()):
                if k in cur_d.keys() and cur_d[k].shape == model_dict[k].shape:
                    if not torch.equal(cur_d[k], model_dict[k]):
                        if idx != 0 and k in updated_keys[idx - 1]:
                            logger.warning(f'{k} is overwritten in the {idx} iteration')
                        ret_d[k] = cur_d[k]
                        updated_keys[idx].add(k)

            return ret_d

        model_dict = _fine_tune_params_dict(pretrained_dict)

    model.load_state_dict(model_dict)
    for i in updated_keys.keys():
        logger.info(f"Loaded parameters (file {i}): [{', '.join(updated_keys[i])}]")




def prepare_model(args, logger=None, **kwargs):
    model_args = {}
    cur_dataset = args.dataset.replace('VTAB.','')

    if args.model.startswith('timm.'):
        from timm.models import create_model

        model = create_model(
            args.model.replace('timm.', ''),
            pretrained=args.pretrained,
            num_classes=DATASET2NUM_CLASSES[cur_dataset],
            drop_rate=args.drop,
            drop_connect_rate=args.drop_connect,
            drop_path_rate=args.drop_path,
            drop_block_rate=args.drop_block,
            global_pool=args.gp,
            bn_momentum=args.bn_momentum,
            bn_eps=args.bn_eps,
            scriptable=args.torchscript,
            checkpoint_path=args.initial_checkpoint
        )

        if args.num_classes is None:
            assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
            args.num_classes = model.num_classes  # FIXME handle model default vs config num_classes more elegantly
        if args.grad_checkpointing:
            model.set_grad_checkpointing(enable=True)

        model_args.update(model.default_cfg)

        if args.model == 'timm.vit_base_patch16_224_in21k':
            model_args.update({
                'patches_size': model.patch_embed.patch_size,
                'hidden_size': model.embed_dim,
                'transformer_num_layers': len(model.blocks),
            })
            
    elif args.model.startswith('huggingface.'):
        _, base_module_name, model_name = args.model.split('.', 2) # huggingface.base_module_name.model_name
        base_module = get_class_from_module('transformers', base_module_name)
        # .from_pretrained(num_labels=DATASET2NUM_CLASSES[cur_dataset], ignore_mismatched_sizes=True) for (transformers.AutoModelForImageClassification.google/vit-base-patch16-224-in21k)
        model = base_module.from_pretrained(
            model_name,
            **kwargs
        )

        model_args.update({
            'input_size': (model.config.image_size, model.config.image_size),
        })

        def _reuse_callback(outputs):
            return outputs.logits
        model.reuse_callback = _reuse_callback

    elif args.model.startswith('vit_pytorch.'):
        image_size = 224
        if args.model == 'vit_pytorch.ViT':
            from vit_pytorch import ViT
            model = ViT(
                image_size=image_size,
                patch_size=32,
                num_classes=DATASET2NUM_CLASSES[cur_dataset],
                dim=1024,
                depth=6,
                heads=16,
                mlp_dim=2048,
                dropout=0.1,
                emb_dropout=0.1
            )
        model_args.update({
            'input_size': (image_size, image_size)
        })

    elif args.model.startswith('torchvision.'):
        if args.model == 'torchvision.vision_transformer.vit_b_16':
            from torchvision.models.vision_transformer import vit_b_16
            from collections import OrderedDict
            import math
            model = vit_b_16(pretrained=True)

            heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
            if model.representation_size is None:
                heads_layers["head"] = nn.Linear(model.hidden_dim, DATASET2NUM_CLASSES[cur_dataset])
            else:
                heads_layers["pre_logits"] = nn.Linear(model.hidden_dim, model.representation_size)
                heads_layers["act"] = nn.Tanh()
                heads_layers["head"] = nn.Linear(model.representation_size, DATASET2NUM_CLASSES[cur_dataset])

            model.heads = nn.Sequential(heads_layers)

            if hasattr(model.heads, "pre_logits") and isinstance(model.heads.pre_logits, nn.Linear):
                fan_in = model.heads.pre_logits.in_features
                nn.init.trunc_normal_(model.heads.pre_logits.weight, std=math.sqrt(1 / fan_in))
                nn.init.zeros_(model.heads.pre_logits.bias)

            if isinstance(model.heads.head, nn.Linear):
                nn.init.zeros_(model.heads.head.weight)
                nn.init.zeros_(model.heads.head.bias)

        model_args.update({
            'input_size': (model.image_size, model.image_size)
        })


    elif args.model.startswith('clip.'):
        import clip
        from zhijian.models.backbone.clip.utils import CLIPModelWrapper
        model, preprocess = clip.load(args.model.replace('clip.', ''))

        model = CLIPModelWrapper(model)
        if args.model == 'clip.ViT-B/32':
            model_args.update({
                # 'dtype': model.dtype,
                # 'ln_final_weight': model.ln_final_weight,
                # 'visual_input_size': model.visual_input_size,
                'preprocess': preprocess,
            })
 
    elif args.model.startswith('customized.'):
        from zhijian.models.backbone.customized import MyModels 
        model = MyModels[args.model.replace('customized.', '')](
            args,
            num_classes=DATASET2NUM_CLASSES[cur_dataset],
            **{k: v for k, v in kwargs.items() if k not in ['model_args']}
        )

        if hasattr(model, "image_size"):
            model_args.update({
                'input_size': (model.image_size, model.image_size)
            })

    else:
        assert False, f'Unkown model type {args.model}'

    if not args.model.startswith('clip.'):
        model = ModelWrapper(model)

    if 'model_args' in kwargs.keys():
        model_args.update(kwargs['model_args'])

    if 'mean' not in model_args.keys() and 'std' not in model_args.keys():
        model_args.update({
            
        })
    safe_update(
        model_args,
        {'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 'resize_size': 224}
    )

    model_args = dict2args(model_args)

    return model, model_args


def prepare_llm(
    model_args,
    finetuning_args,
    is_trainable: Optional[bool] = False,
    stage: Optional[Literal["pt", "sft", "rm", "ppo"]] = "sft",
    logger=None
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    r"""
    Loads pretrained model and tokenizer.

    Support both training and inference.
    """

    if (not is_trainable) and model_args.checkpoint_dir is None:
        if logger is not None:
            logger.info("Checkpoint is not found at evaluation, load the original model.")
        finetuning_args = FinetuningArguments(finetuning_type="none")

    assert stage in ["pt", "sft"] or finetuning_args.finetuning_type == "lora", \
        "RM and PPO training can only be performed with the LoRA method."

    config_kwargs = {
        "trust_remote_code": True,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model,
        use_fast=model_args.use_fast_tokenizer,
        padding_side="left",
        **config_kwargs
    )
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == 64000: # 64000 for baichuan model (older version)
        tokenizer.pad_token_id = 0 # set as the <unk> token

    config = AutoConfig.from_pretrained(model_args.model, **config_kwargs)
    is_mergeable = True

    # Quantization configurations (using bitsandbytes library).
    if model_args.quantization_bit is not None:
        if model_args.quantization_bit == 8:
            require_version("bitsandbytes>=0.37.0", "To fix: pip install bitsandbytes>=0.37.0")
            config_kwargs["load_in_8bit"] = True
            config_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
        elif model_args.quantization_bit == 4:
            require_version("bitsandbytes>=0.39.0", "To fix: pip install bitsandbytes>=0.39.0")
            require_version("transformers>=4.30.1", "To fix: pip install transformers>=4.30.1")
            require_version("accelerate>=0.20.3", "To fix: pip install accelerate>=0.20.3")
            require_version("peft>=0.4.0.dev0", "To fix: pip install git+https://github.com/huggingface/peft.git")
            config_kwargs["load_in_4bit"] = True
            config_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=model_args.compute_dtype,
                bnb_4bit_use_double_quant=model_args.double_quantization,
                bnb_4bit_quant_type=model_args.quantization_type
            )
        config_kwargs["device_map"] = {"": int(os.environ.get("LOCAL_RANK", "0"))}
        if logger is not None:
            logger.info("Quantizing model to {} bit.".format(model_args.quantization_bit))

    if not is_trainable: # `device_map=auto` should be used for inference only
        config_kwargs["device_map"] = "auto"

    if model_args.checkpoint_dir is not None and finetuning_args.finetuning_type == "full":
        model_to_load = model_args.checkpoint_dir[0]
    else:
        model_to_load = model_args.model

    # Load and prepare pretrained models (without valuehead).
    model = AutoModelForCausalLM.from_pretrained(
        model_to_load,
        config=config,
        torch_dtype=torch.bfloat16 if model_args.compute_dtype == torch.bfloat16 else torch.float16,
        low_cpu_mem_usage=True,
        **config_kwargs
    )
    # model = prepare_model_for_training(model, finetuning_args.finetuning_type) if is_trainable else model
    # model = _init_adapter(model, model_args, finetuning_args, is_trainable, is_mergeable)

    if stage == "rm" or stage == "ppo": # add value head
        from trl import AutoModelForCausalLMWithValueHead
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model)

        if stage == "rm" and model_args.checkpoint_dir is not None: # load valuehead weights to evaluate reward model
            if logger is not None:
                logger.info("Only the last checkpoint containing valuehead will be loaded as the valuehead.")
            def load_valuehead_params(model: torch.nn.Module, checkpoint_dir: os.PathLike) -> bool:
                VALUE_HEAD_FILE_NAME = "value_head.bin"
                valuehead_file = os.path.join(checkpoint_dir, VALUE_HEAD_FILE_NAME)
                if not os.path.exists(valuehead_file):
                    if logger is not None:
                        logger.warning("Provided path ({}) does not contain valuehead weights.".format(checkpoint_dir))
                    return False
                valuehead_state_dict = torch.load(valuehead_file, map_location="cpu")
                model.register_buffer("reward_head_weight", valuehead_state_dict["summary.weight"])
                model.register_buffer("reward_head_bias", valuehead_state_dict["summary.bias"])
                model.register_buffer("default_head_weight", torch.zeros_like(valuehead_state_dict["summary.weight"]))
                model.register_buffer("default_head_bias", torch.zeros_like(valuehead_state_dict["summary.bias"]))
                return True
            if load_valuehead_params(model, model_args.checkpoint_dir[-1]):
                model.v_head.load_state_dict({
                    "summary.weight": getattr(model, "reward_head_weight"),
                    "summary.bias": getattr(model, "reward_head_bias")
                })

        if stage == "ppo": # load reward model
            assert is_trainable, "PPO stage cannot be performed at evaluation."
            assert model_args.reward_model is not None, "Reward model is necessary for PPO training."
            if logger is not None:
                logger.info("Load reward model from {}".format(model_args.reward_model))
            model.pretrained_model.load_adapter(model_args.reward_model, "reward", is_trainable=False)
            assert load_valuehead_params(model, model_args.reward_model), "Reward model is not correctly loaded."

    if not is_trainable:
        model.requires_grad_(False) # fix all model params
        model = model.half() if model_args.quantization_bit is None else model # cast from fp32 to fp16

    return model, tokenizer

