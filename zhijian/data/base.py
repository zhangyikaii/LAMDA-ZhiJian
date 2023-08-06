import os
from zhijian.models.utils import load_pickle, save_pickle, MyImageFolderDataset
from zhijian.models.configs.base import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from zhijian.data.config import DATA_PATH_SUB_DIR, DATASET2NUM_CLASSES, IGNORE_INDEX
from zhijian.data.template import Template

import torch
from torch.utils.data.sampler import Sampler
import random

from tqdm import tqdm

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets.folder import ImageFolder, default_loader

import re

from typing import Dict, Optional, Sequence, Union, List, Literal

from transformers import DataCollatorWithPadding, BatchEncoding
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import Seq2SeqTrainingArguments


from itertools import chain
import hashlib

from datasets import Dataset, concatenate_datasets, load_dataset



class TestAugTransform:
    def __init__(self, transform, aug_times):
        self.transform = transform
        self.aug_times = aug_times
    def __call__(self, x):
        return [self.transform(x) for _ in range(self.aug_times)]

def get_in_train_transform(dataset, crop_size, mean, std):
    if dataset in ['CLEVR-Distance', 'CLEVR-Count', 'smallNORB-Azimuth', 'smallNORB-Elevation', 'dSprites-Orientation', 'dSprites-Location', 'KITTI']:
         return transforms.Compose([
            transforms.Resize((224, 224), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    else:
        return transforms.Compose([
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=mean, std=std
            )
        ])

def get_in_test_transform(dataset, resize_size, crop_size, mean, std):
    if dataset in ['CLEVR-Distance', 'CLEVR-Count', 'smallNORB-Azimuth', 'smallNORB-Elevation', 'dSprites-Orientation', 'dSprites-Location', 'KITTI']:
        return transforms.Compose([
            transforms.Resize((224, 224), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)])
    else:
        if resize_size is None and crop_size is None:
            return transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=mean, std=std),
            ])
        return transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=mean, std=std
            )
        ])

def prepare_vision_dataloader(args, model_args, logger=None):
    remove_pattern = '|'.join(['VTAB-1k.', 'VTAB-tuning'])
    raw_dataset = re.sub(remove_pattern, '', args.dataset)

    if hasattr(model_args, 'preprocess'):
        train_transform = model_args.preprocess
        val_transform = model_args.preprocess
    else:
        train_transform = get_in_train_transform(raw_dataset, model_args.input_size[1], model_args.mean, model_args.std)
        val_transform = get_in_test_transform(raw_dataset, model_args.resize_size, model_args.input_size[1], model_args.mean, model_args.std)

    train_dataset, val_dataset, num_classes = get_dataset(args.dataset, os.path.join(args.dataset_dir, DATA_PATH_SUB_DIR[args.dataset]), train_transform, val_transform, logger)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=True
    ) if train_dataset is not None else None

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False
    ) if val_dataset is not None else None


    return train_loader, val_loader, num_classes


class general_vtab_dataset(ImageFolder):
    def __init__(self, name, root, train=True, transform=None, **kwargs):
        """write the description of the function here, like this is the ...

        write the description of the function here

        :param a1: write parameter a1 description here

        :return: write return thing here
        """
        self.dataset_root = root
        self.loader = default_loader
        self.target_transform = None
        self.transform = transform

        if 'VTAB-1k' in name:
            train_list_path = os.path.join(self.dataset_root, 'train800val200.txt')
            test_list_path = os.path.join(self.dataset_root, 'test.txt')
        elif 'VTAB-tuning' in name:
            train_list_path = os.path.join(self.dataset_root, 'train800.txt')
            test_list_path = os.path.join(self.dataset_root, 'val200.txt')
        else:
            raise NotImplementedError

        self.samples = []
        if train:
            with open(train_list_path, 'r') as f:
                for line in f:
                    img_name = line.split(' ')[0]
                    label = int(line.split(' ')[1])
                    self.samples.append((os.path.join(root,img_name), label))
        else:
            with open(test_list_path, 'r') as f:
                for line in f:
                    img_name = line.split(' ')[0]
                    label = int(line.split(' ')[1])
                    self.samples.append((os.path.join(root,img_name), label))

class imagenet_dataset(ImageFolder):
    def __init__(self, root, train=True, transform=None):
        if train:
            root = os.path.join(root, 'train')
        else:
            root = os.path.join(root, 'val')
        super().__init__(root, transform)

class general_imageneta_dataset(ImageFolder):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, transform)


def get_dataset(name, data_path, train_transform, val_transform, logger=None):
    def imagefolder_dataset(train_prefix, test_prefix):
        return torchvision.datasets.ImageFolder(os.path.join(data_path, train_prefix), transform=train_transform), torchvision.datasets.ImageFolder(os.path.join(data_path, test_prefix), transform=val_transform)

    if 'VTAB-1k' in name or 'VTAB-tuning' in name:
        train_dataset, val_dataset = general_vtab_dataset(name, data_path, train=True, transform=train_transform), general_vtab_dataset(name, data_path, train=False, transform=val_transform)
        num_classes = DATASET2NUM_CLASSES[name]

    elif name == 'ImageNet':
        train_dataset, val_dataset = imagenet_dataset(data_path, train = True, transform= train_transform), imagenet_dataset(data_path, train=False, transform=val_transform)
        num_classes = 1000
    elif name == 'ImageNet-A':
        train_dataset = None
        val_dataset = general_imageneta_dataset(data_path, transform=val_transform)
        num_classes = 200
    elif name == 'ImageNet-R':
        train_dataset = None
        val_dataset = general_imageneta_dataset(data_path, transform=val_transform)
        num_classes = 200
    elif name == 'CIFAR-10':
        train_dataset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=False, transform=train_transform)
        val_dataset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=False, transform=val_transform)
        num_classes = 10
    elif name == 'CIFAR-100':
        train_dataset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=False, transform=train_transform)
        val_dataset = torchvision.datasets.CIFAR100(root=data_path, train=False, download=False, transform=val_transform)
        num_classes = 100
    elif name == 'Aircraft':
        from zhijian.data.aircraft import Aircraft
        train_dataset = Aircraft(data_path, transform=train_transform, train=True, download=False)
        val_dataset = Aircraft(data_path, transform=val_transform, train=False, download=False)
        num_classes = 100
    elif name == 'Caltech101':
        from zhijian.data.caltech101 import Caltech101
        train_dataset = Caltech101(data_path, transform=train_transform, train=True)
        val_dataset = Caltech101(data_path, transform=val_transform, train=False)
        num_classes = 101
    elif name == 'Cars':
        from zhijian.data.cars import Cars
        train_dataset = Cars(data_path, transform=train_transform, train=True, download=False)
        val_dataset = Cars(data_path, transform=val_transform, train=False, download=False)
        num_classes = 196
    elif name == 'CUB2011':
        from zhijian.data.cub2011 import CUB2011
        train_dataset = CUB2011(data_path, transform=train_transform, train=True, download=False)
        val_dataset = CUB2011(data_path, transform=val_transform, train=False, download=False)
        num_classes = 200
    elif name == 'Dogs':
        from zhijian.data.dogs import Dogs
        train_dataset = Dogs(data_path, transform=train_transform, train=True, download=False)
        val_dataset = Dogs(data_path, transform=val_transform, train=False, download=False)
        num_classes = 120
    elif name == 'DTD':
        from zhijian.data.dtd import DTD
        train_dataset = DTD(data_path, transform=train_transform, train=True)
        val_dataset = DTD(data_path, transform=val_transform, train=False)
        num_classes = 47
    elif name == 'EuroSAT':
        from zhijian.data.eurosat import EuroSAT
        train_dataset = EuroSAT(data_path, transform=train_transform, train=True)
        val_dataset = EuroSAT(data_path, transform=val_transform, train=False)
        num_classes = 10
    elif name == 'Flowers':
        from zhijian.data.flowers import Flowers
        train_dataset = Flowers(data_path, transform=train_transform, train=True)
        val_dataset = Flowers(data_path, transform=val_transform, train=False)
        num_classes = 102
    elif name == 'Food':
        train_dataset, val_dataset = imagefolder_dataset('train', 'test')
        num_classes = 101
    elif name == 'Pet':
        from zhijian.data.oxford_iiit_pet import OxfordIIITPet
        train_dataset = OxfordIIITPet(root=data_path, split='trainval', download=False, transform=train_transform)
        val_dataset = OxfordIIITPet(root=data_path, split='test', download=False, transform=val_transform)
        num_classes = 37
    elif name == 'STL10':
        train_dataset = torchvision.datasets.STL10(root=data_path, split='train', download=False, transform=train_transform)
        val_dataset = torchvision.datasets.STL10(root=data_path, split='test', download=False, transform=val_transform)
        num_classes = 10
    elif name == 'SVHN':
        train_dataset = torchvision.datasets.SVHN(root=data_path, split='train', download=False, transform=train_transform)
        val_dataset = torchvision.datasets.SVHN(root=data_path, split='test', download=False, transform=val_transform)
        num_classes = 10
    elif name == 'SUN397':
        from zhijian.data.sun397 import SUN397
        train_dataset = SUN397(data_path, transform=train_transform, train=True)
        val_dataset = SUN397(data_path, transform=val_transform, train=False)
        num_classes = 397
    elif name in ['OfficeHome', 'PACS', 'DomainNet', 'VLCS']:
        if name == 'OfficeHome':
            raise Exception('train test do not split')
            from udatasets.officehome import OfficeHome
            cur_dataset_class = OfficeHome
            domains = ["Art", "Clipart", "Product", "Real World"]
            num_classes = 65
        elif name == 'PACS':
            from zhijian.data.pacs import PACS
            cur_dataset_class = PACS
            domains = ["art_painting", "cartoon", "photo", "sketch"]
            num_classes = 7
        elif name == 'DomainNet':
            from zhijian.data.domainnet import DomainNet
            cur_dataset_class = DomainNet
            domains = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
            num_classes = 345
        from zhijian.data.domain import DatasetWrapper
        full_dataset = cur_dataset_class(data_path, domains, domains)
        train_dataset = DatasetWrapper(full_dataset.train_x, transform=train_transform)
        val_dataset = DatasetWrapper(full_dataset.test, transform=val_transform)
    elif 'DomainNet-' in name:
        from zhijian.data.domainnet import DomainNet
        from zhijian.data.domain import DatasetWrapper
        domains = [name[name.find('DomainNet-') + len('DomainNet-'):]]
        num_classes = 345
        full_dataset = DomainNet(data_path, domains, domains)
        train_dataset = DatasetWrapper(full_dataset.train_x, transform=train_transform)
        val_dataset = DatasetWrapper(full_dataset.test, transform=val_transform)
    elif name == 'NABirds':
        from zhijian.data.nabirds import NABirds
        train_dataset = NABirds(data_path, transform=train_transform, train=True, download=False)
        val_dataset = NABirds(data_path, transform=val_transform, train=False, download=False)
        num_classes = 555
    elif name == 'SmallNORB':
        from zhijian.data.smallnorb import SmallNORB
        train_dataset = SmallNORB(data_path, transform=train_transform, train=True, download=True)
        val_dataset = SmallNORB(data_path, transform=val_transform, train=False, download=True)
        num_classes = 5
    elif name == 'PCAM':
        from zhijian.data.pcam import PCAM
        train_dataset = PCAM(root=data_path, split='train', transform=train_transform, download=False)
        val_dataset = PCAM(root=data_path, split='test', transform=val_transform, download=False)
        num_classes = 2
    elif name == 'Resisc45' or name == 'AID':
        if not os.path.isfile(os.path.join(data_path, 'train.pkl')) and not os.path.isfile(os.path.join(data_path, 'test.pkl')):
            raise Exception('split file error')
            traintest_dataset = torchvision.datasets.ImageFolder(data_path)
            n_traintest = len(traintest_dataset)
            traintest_indices = list(range(n_traintest))
            random.shuffle(traintest_indices)
            split_point = int(n_traintest * 0.8)
            save_pickle(os.path.join(data_path, 'train.pkl'), [(traintest_dataset.samples[i][0].replace(f'{data_path}/', ''), traintest_dataset.samples[i][1]) for i in traintest_indices[: split_point]])
            save_pickle(os.path.join(data_path, 'test.pkl'), [(traintest_dataset.samples[i][0].replace(f'{data_path}/', ''), traintest_dataset.samples[i][1]) for i in traintest_indices[split_point: ]])

        train_samples = load_pickle(os.path.join(data_path, 'train.pkl'))
        train_samples = [(os.path.join(data_path, i[0]), i[1]) for i in train_samples]
        val_samples = load_pickle(os.path.join(data_path, 'test.pkl'))
        val_samples = [(os.path.join(data_path, i[0]), i[1]) for i in val_samples]
        train_dataset = MyImageFolderDataset(train_samples, transform=train_transform)
        val_dataset = MyImageFolderDataset(val_samples, transform=val_transform)
        if name == 'Resisc45':
            num_classes = 45
        elif name == 'AID':
            num_classes = 30
    else:
        raise NotImplementedError

    if logger is not None:
        logger.info(f'Dataset: {name} - [train {len(train_dataset) if train_dataset is not None else 0}] [test {len(val_dataset)}] [num_classes {num_classes}]')
    return train_dataset, val_dataset, num_classes


class ClassUniformlySampler(Sampler):
    '''
    Arguments:
        data_source (Dataset): data_loader to sample from
        class_position (int): which one is used as class
        k (int): sample k images of each class
    '''
 
    def __init__(self, data_source, pkl_file, class_position, k):
        super().__init__(data_source)
        self.class_position = class_position
        self.k = k
        self.pkl_file = pkl_file

        self.samples = data_source
        self.class_dict = self._tuple2dict(self.samples)
        self.sample_list = self._generate_list(self.class_dict)
 
    def __iter__(self):
        return iter(self.sample_list)
 
    def __len__(self):
        return len(self.sample_list)
 
    def _tuple2dict(self, inputs):
        '''
        :param inputs: list with tuple elemnts, [(image_path1, class_index_1), (imagespath_2, class_index_2), ...]
        :return class2indices, {class_index_i: [samples_index1, samples_index2, ...]}
        '''
        if os.path.isfile(self.pkl_file):
            class2indices = load_pickle(self.pkl_file)
        else:
            class2indices = {}
            for index, each_input in enumerate(tqdm(inputs)):
                class_index = each_input[self.class_position]
                if class_index not in list(class2indices.keys()):
                    class2indices[class_index] = [index]
                else:
                    class2indices[class_index].append(index)
            save_pickle(self.pkl_file, class2indices)
        return class2indices
 
    def _generate_list(self, class2indices):
        '''
        :param class2indices:  {class_index_i: [samples_index1, samples_index2, ...]}
        :return [samples_index1, samples_index3, samples_index2, ...]
        '''
        sample_list = []
 
        class2indices_copy = class2indices.copy()
        cur_keys = list(class2indices_copy.keys()) # [class_index_0, class_index_1, ...]
        random.shuffle(cur_keys)
        class_num_dict = {}
        cur_min = 99999
        for key in cur_keys:
            value = class2indices_copy[key] # [samples_index1, samples_index2, ...]
            class_num_dict[key] = len(value)
            cur_min = min(cur_min, len(value))
            if len(value) >= self.k:
                random.shuffle(value)
                sample_list.extend(value[0 : self.k])
            else:
                random.shuffle(value)
                sample_list.extend(value)
        return sample_list


class DatasetPercentSampler(ClassUniformlySampler):
    def __init__(self, data_source, pkl_file, class_position, p):
        self.p = p
        super().__init__(data_source, pkl_file, class_position, -2)
    def __iter__(self):
        return iter(self.sample_list)

    def __len__(self):
        return len(self.sample_list)

    def _generate_list(self, class2indices):
        sample_list = []

        class2indices_copy = class2indices.copy()
        cur_keys = list(class2indices_copy.keys()) # [class_index_0, class_index_1, ...]
        random.shuffle(cur_keys)
        for key in cur_keys:
            value = class2indices_copy[key] # [samples_index1, samples_index2, ...]
            random.shuffle(value)
            sample_list.append(value[0])

        sample_num = int(self.p * len(self.samples))
        if len(sample_list) < sample_num:
            rest_sample_num = sample_num - len(sample_list)
            rest_indices = list(set(list(range(len(self.samples)))) - set(sample_list))
            sampled = random.sample(rest_indices, min(len(rest_indices), rest_sample_num))
            sample_list.extend(sampled)

        return sample_list


def prepare_llm_dataset(
    model_args,
    data_args,
    logger=None
) -> Dataset:

    def checksum(file_path, hash):
        with open(file_path, "rb") as datafile:
            binary_data = datafile.read()
        sha1 = hashlib.sha1(binary_data).hexdigest()
        if sha1 != hash and logger is not None:
            logger.warning("Checksum failed for {}. It may vary depending on the platform.".format(file_path))

    ext2type = {
        "csv": "csv",
        "json": "json",
        "jsonl": "json",
        "txt": "text"
    }

    max_samples = data_args.max_samples
    all_datasets: List[Dataset] = [] # support multiple datasets

    for dataset_attr in data_args.dataset_list:

        logger.info("Loading dataset {}...".format(dataset_attr))

        if dataset_attr.load_from == "hf_hub":
            data_path = dataset_attr.dataset_name
            data_files = None
        elif dataset_attr.load_from == "script":
            data_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
            data_files = None
        elif dataset_attr.load_from == "file":
            data_path = None
            data_files: List[str] = []

            if os.path.isdir(os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)):
                for file_name in os.listdir(os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)):
                    data_files.append(os.path.join(data_args.dataset_dir, dataset_attr.dataset_name, file_name))

                    if data_path is None:
                        data_path = ext2type.get(data_files[0].split(".")[-1], None)
                    else:
                        assert data_path == ext2type.get(data_files[-1].split(".")[-1], None), "file type does not match."
            elif os.path.isfile(os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)):
                data_files.append(os.path.join(data_args.dataset_dir, dataset_attr.dataset_name))
                data_path = ext2type.get(data_files[0].split(".")[-1], None)
            else:
                raise ValueError("File not found.")

            assert data_path, "File extension must be txt, csv, json or jsonl."

            if len(data_files) == 1 and dataset_attr.dataset_sha1 is not None:
                checksum(data_files[0], dataset_attr.dataset_sha1)
            else:
                if logger is not None:
                    logger.warning("Checksum failed: missing SHA-1 hash value in dataset_info.json or too many files.")
        else:
            raise NotImplementedError

        raw_datasets = load_dataset(
            data_path,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None
        )
        dataset = raw_datasets[data_args.split]

        if max_samples is not None:
            max_samples_temp = min(len(dataset), max_samples)
            dataset = dataset.select(range(max_samples_temp))

        dummy_data = [None] * len(dataset)
        prefix_data = [dataset_attr.source_prefix] * len(dataset)
        for column_name, target_name in [
            ("prompt_column", "prompt"),
            ("query_column", "query"),
            ("response_column", "response"),
            ("history_column", "history")
        ]: # every dataset will have 4 columns same as each other
            if getattr(dataset_attr, column_name) != target_name:
                if getattr(dataset_attr, column_name):
                    dataset = dataset.rename_column(getattr(dataset_attr, column_name), target_name)
                else: # None or empty string
                    dataset = dataset.add_column(target_name, dummy_data)
        dataset = dataset.add_column("prefix", prefix_data)
        all_datasets.append(dataset)

    if len(data_args.dataset_list) == 1:
        all_datasets = all_datasets[0]
    else:
        all_datasets = concatenate_datasets(all_datasets)

    return all_datasets


class DynamicDataCollatorWithPadding(DataCollatorWithPadding):
    r"""
    Inherits DataCollatorWithPadding. It is capable of dynamically padding for batched data.
    """
    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            ignore_pad_token_for_loss: Optional[bool] = False
    ):
        super().__init__(tokenizer, padding=True)
        self.label_pad_token_id = IGNORE_INDEX if ignore_pad_token_for_loss else tokenizer.pad_token_id

    def get_attention_masks(self, input_ids: torch.Tensor, device: torch.device) -> torch.Tensor:
        r"""
        Generates attention masks for left-padded sequences.
        """
        batch_size, seq_length = input_ids.size()
        attention_mask = torch.ones((batch_size, seq_length), device=device)

        for i, seq in enumerate(input_ids):
            attention_mask[i, :(seq != self.tokenizer.pad_token_id).nonzero()[0].item()] = 0 # padding

        attention_mask = attention_mask.bool()
        return attention_mask

    def __call__(self, features: Sequence[Dict[str, Union[torch.Tensor, Sequence[int]]]]) -> BatchEncoding:
        r"""
        Pads batched data to the longest sequence in the batch.

        We adopt left-padding in both training and evaluation.
        """
        if isinstance(features[0]["input_ids"], torch.Tensor):
            input_ids = [feature["input_ids"].clone().detach().flip(0) for feature in features]
        else:
            input_ids = [torch.tensor(feature["input_ids"]).flip(0) for feature in features]

        if "labels" in features[0]:
            if isinstance(features[0]["labels"], torch.Tensor):
                labels = [feature["labels"].clone().detach().flip(0) for feature in features]
            else:
                labels = [torch.tensor(feature["labels"]).flip(0) for feature in features]
            input_ids = input_ids + labels # pad them to the same length

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        ).flip(-1)

        batch = {}

        if "labels" in features[0]:
            input_ids, labels = input_ids.split(len(features), dim=0)
            labels = torch.where(labels != self.tokenizer.pad_token_id, labels, self.label_pad_token_id)
            batch["labels"] = labels

        batch["input_ids"] = input_ids
        batch["attention_mask"] = self.get_attention_masks(input_ids, device=input_ids.device)

        return BatchEncoding(batch)


def preprocess_llm_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    data_args,
    training_args: Seq2SeqTrainingArguments,
    stage: Literal["pt", "sft", "rm", "ppo"],
    logger=None
) -> Dataset:

    column_names = list(dataset.column_names)
    prompt_template = Template(data_args.prompt_template)

    # support question with a single answer or multiple answers
    def get_dialog(examples):
        for i in range(len(examples["prompt"])):
            if examples["prompt"][i] and examples["response"][i]:
                query, answer = examples["prompt"][i], examples["response"][i]
                query = query + "\n" + examples["query"][i] if examples["query"][i] else query
                prefix = examples["prefix"][i] if examples["prefix"][i] else ""
                dialog = prompt_template.get_dialog(query, answer, examples["history"][i], prefix)
                yield dialog

    def preprocess_pretrain_dataset(examples):
        # build grouped texts with format `<bos> X1 X2 X3 ...` (without <eos>)
        text_ids = tokenizer(examples["prompt"], add_special_tokens=False)["input_ids"]
        concatenated_ids = list(chain(*text_ids))
        total_length = len(concatenated_ids)
        block_size = data_args.max_source_length - 1
        # we drop the small remainder, and if the total_length < block_size, we exclude this batch
        total_length = (total_length // block_size) * block_size
        # split by chunks of max_source_length
        result = [[tokenizer.bos_token_id] + concatenated_ids[i: i + block_size]
                  for i in range(0, total_length, block_size)]
        return {
            "input_ids": result,
            "labels": result.copy()
        }

    def preprocess_supervised_dataset(examples):
        # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
        # for input with history, we build multiple input-label pairs just like:
        # https://github.com/lm-sys/FastChat/blob/f17c092f64840fa6354ed52789dccb2daa793d0b/fastchat/train/train.py#L112
        model_inputs = {"input_ids": [], "labels": []}
        for dialog in get_dialog(examples):
            input_ids, labels = [], []

            for i in range(len(dialog) // 2):
                source_ids = tokenizer.encode(text=dialog[2*i], add_special_tokens=False)
                target_ids = tokenizer.encode(text=dialog[2*i+1], add_special_tokens=False)

                if len(source_ids) > data_args.max_source_length - 1: # bos token
                    source_ids = source_ids[:data_args.max_source_length - 1]
                if len(target_ids) > data_args.max_target_length - 1: # eos token
                    target_ids = target_ids[:data_args.max_target_length - 1]

                input_ids += [tokenizer.bos_token_id] + source_ids + target_ids + [tokenizer.eos_token_id]
                labels += [IGNORE_INDEX] * (len(source_ids) + 1) + target_ids + [tokenizer.eos_token_id]

            if len(input_ids) > data_args.max_source_length + data_args.max_target_length:
                input_ids = input_ids[:data_args.max_source_length + data_args.max_target_length]
            if len(labels) > data_args.max_source_length + data_args.max_target_length:
                labels = labels[:data_args.max_source_length + data_args.max_target_length]

            model_inputs["input_ids"].append(input_ids)
            model_inputs["labels"].append(labels)
        return model_inputs

    def preprocess_unsupervised_dataset(examples):
        # build inputs with format `<bos> X` and labels with format `<bos> Y`
        model_inputs = {"input_ids": [], "labels": []}
        for dialog in get_dialog(examples):
            prompt, answer = "".join(dialog[:-1]), dialog[-1]

            source_ids = tokenizer.encode(text=prompt, add_special_tokens=False)
            target_ids = tokenizer.encode(text=answer, add_special_tokens=False)

            if len(source_ids) > data_args.max_source_length - 1: # bos token
                source_ids = source_ids[:data_args.max_source_length - 1]
            if len(target_ids) > data_args.max_target_length - 1: # bos token
                target_ids = target_ids[:data_args.max_target_length - 1]

            input_ids = [tokenizer.bos_token_id] + source_ids
            labels = [tokenizer.bos_token_id] + target_ids

            model_inputs["input_ids"].append(input_ids)
            model_inputs["labels"].append(labels)
        return model_inputs

    def preprocess_pairwise_dataset(examples):
        # build input pairs with format `<bos> X Y1 <eos>` and `<bos> X Y2 <eos>`
        model_inputs = {"accept_ids": [], "reject_ids": []}
        for dialog in get_dialog(examples):
            prompt, answer = "".join(dialog[:-1]), dialog[-1]

            source_ids = tokenizer.encode(text=prompt, add_special_tokens=False)
            accept_ids = tokenizer.encode(text=answer[0], add_special_tokens=False)
            reject_ids = tokenizer.encode(text=answer[1], add_special_tokens=False)

            if len(source_ids) > data_args.max_source_length - 1: # bos token
                source_ids = source_ids[:data_args.max_source_length - 1]
            if len(accept_ids) > data_args.max_target_length - 1: # eos token
                accept_ids = accept_ids[:data_args.max_target_length - 1]
            if len(reject_ids) > data_args.max_target_length - 1: # eos token
                reject_ids = reject_ids[:data_args.max_target_length - 1]

            accept_ids = [tokenizer.bos_token_id] + source_ids + accept_ids + [tokenizer.eos_token_id]
            reject_ids = [tokenizer.bos_token_id] + source_ids + reject_ids + [tokenizer.eos_token_id]

            model_inputs["accept_ids"].append(accept_ids)
            model_inputs["reject_ids"].append(reject_ids)
        return model_inputs

    def print_supervised_dataset_example(example):
        logger.info("input_ids:\n{}".format(example["input_ids"]))
        logger.info("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        logger.info("label_ids:\n{}".format(example["labels"]))
        logger.info("labels:\n{}".format(
            tokenizer.decode([d if d != IGNORE_INDEX else tokenizer.pad_token_id for d in example["labels"]],
                             skip_special_tokens=False)
        ))

    def print_pairwise_dataset_example(example):
        logger.info("accept_ids:\n{}".format(example["accept_ids"]))
        logger.info("accepts:\n{}".format(tokenizer.decode(example["accept_ids"], skip_special_tokens=False)))
        logger.info("reject_ids:\n{}".format(example["reject_ids"]))
        logger.info("rejects:\n{}".format(tokenizer.decode(example["reject_ids"], skip_special_tokens=False)))

    def print_unsupervised_dataset_example(example):
        logger.info("input_ids:\n{}".format(example["input_ids"]))
        logger.info("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))

    if stage == "pt":
        preprocess_function = preprocess_pretrain_dataset
    elif stage == "sft":
        preprocess_function = preprocess_unsupervised_dataset \
            if training_args.predict_with_generate else preprocess_supervised_dataset
    elif stage == "rm":
        preprocess_function = preprocess_pairwise_dataset
    elif stage == "ppo":
        preprocess_function = preprocess_unsupervised_dataset

    with training_args.main_process_first(desc="dataset map pre-processing"):
        dataset = dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset"
        )

        if stage == "pt":
            print_unsupervised_dataset_example(dataset[0])
        elif stage == "sft":
            print_supervised_dataset_example(dataset[0])
        elif stage == "rm":
            print_pairwise_dataset_example(dataset[0])
        elif stage == "ppo":
            print_unsupervised_dataset_example(dataset[0])

        return dataset
