IN1K_CLASSES = 1000

TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}'  # tqdm bar format

DATA_PATH_PREFIX = '/data/zhangyk/data'
DATA_PATHS = {
    'ImageNet': f'{DATA_PATH_PREFIX}/imagenet',
    'CIFAR10': f'{DATA_PATH_PREFIX}/cifar10',
    'CIFAR100': f'{DATA_PATH_PREFIX}/cifar100',
    'Aircraft': f'{DATA_PATH_PREFIX}/aircraft',
    'Caltech101': f'{DATA_PATH_PREFIX}/caltech101/caltech-101',
    'Cars': f'{DATA_PATH_PREFIX}/cars',
    'CUB2011': f'{DATA_PATH_PREFIX}/cub2011',
    'Dogs': f'{DATA_PATH_PREFIX}/dogs',
    'DomainNet': f'{DATA_PATH_PREFIX}/domainnet',
    'DTD': f'{DATA_PATH_PREFIX}/dtd',
    'EuroSAT': f'{DATA_PATH_PREFIX}/eurosat',
    'Flowers': f'{DATA_PATH_PREFIX}/flowers',
    'Food': f'{DATA_PATH_PREFIX}/food/food-101',
    'NABirds': f'{DATA_PATH_PREFIX}/nabirds',
    'OfficeHome': f'{DATA_PATH_PREFIX}/officehome',
    'PACS': f'{DATA_PATH_PREFIX}/pacs',
    'Pet': f'{DATA_PATH_PREFIX}/pet',
    'SmallNORB': f'{DATA_PATH_PREFIX}/smallnorb',
    'STL10': f'{DATA_PATH_PREFIX}/stl10',
    'SUN397': f'{DATA_PATH_PREFIX}/sun397',
    'SVHN': f'{DATA_PATH_PREFIX}/svhn',
    'FFCV-ImageNet': f'{DATA_PATH_PREFIX}/imagenet',
    'PCAM': f'{DATA_PATH_PREFIX}/pcam',
    'Resisc45': f'{DATA_PATH_PREFIX}/NWPU-RESISC45',
    'AID': f'{DATA_PATH_PREFIX}/AID',
    'PCAM': f'{DATA_PATH_PREFIX}/pcam',
    'DomainNet-clipart': f'{DATA_PATH_PREFIX}/domainnet',
    'DomainNet-infograph': f'{DATA_PATH_PREFIX}/domainnet',
    'DomainNet-painting': f'{DATA_PATH_PREFIX}/domainnet',
    'DomainNet-quickdraw': f'{DATA_PATH_PREFIX}/domainnet',
    'DomainNet-real': f'{DATA_PATH_PREFIX}/domainnet',
    'DomainNet-sketch': f'{DATA_PATH_PREFIX}/domainnet',
}

DEFAULT_CROP_PCT = 0.875
DEFAULT_CROP_MODE = 'center'
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)
IMAGENET_DPN_MEAN = (124 / 255, 117 / 255, 104 / 255)
IMAGENET_DPN_STD = tuple([1 / (.0167 * 255)] * 3)
OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

try:
    from torchvision.transforms.functional import InterpolationMode
    has_interpolation_mode = True
except ImportError:
    has_interpolation_mode = False

if has_interpolation_mode:
    _torch_interpolation_to_str = {
        InterpolationMode.NEAREST: 'nearest',
        InterpolationMode.BILINEAR: 'bilinear',
        InterpolationMode.BICUBIC: 'bicubic',
        InterpolationMode.BOX: 'box',
        InterpolationMode.HAMMING: 'hamming',
        InterpolationMode.LANCZOS: 'lanczos',
    }
    _str_to_torch_interpolation = {b: a for a, b in _torch_interpolation_to_str.items()}
else:
    _pil_interpolation_to_torch = {}
    _torch_interpolation_to_str = {}

from PIL import Image

if hasattr(Image, "Resampling"):
    _pil_interpolation_to_str = {
        Image.Resampling.NEAREST: 'nearest',
        Image.Resampling.BILINEAR: 'bilinear',
        Image.Resampling.BICUBIC: 'bicubic',
        Image.Resampling.BOX: 'box',
        Image.Resampling.HAMMING: 'hamming',
        Image.Resampling.LANCZOS: 'lanczos',
    }
else:
    _pil_interpolation_to_str = {
        Image.NEAREST: 'nearest',
        Image.BILINEAR: 'bilinear',
        Image.BICUBIC: 'bicubic',
        Image.BOX: 'box',
        Image.HAMMING: 'hamming',
        Image.LANCZOS: 'lanczos',
    }

_str_to_pil_interpolation = {b: a for a, b in _pil_interpolation_to_str.items()}

def str_to_pil_interp(mode_str):
    return _str_to_pil_interpolation[mode_str]

def str_to_interp_mode(mode_str):
    if has_interpolation_mode:
        return _str_to_torch_interpolation[mode_str]
    else:
        return _str_to_pil_interpolation[mode_str]
