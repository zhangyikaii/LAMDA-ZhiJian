import os
from typing import Callable, Optional

from torchvision import models, datasets, transforms

class EuroSAT(datasets.ImageFolder):
    """RGB version of the `EuroSAT <https://github.com/phelber/eurosat>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``root/eurosat`` exists.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again. Default is False.
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ) -> None:
        self.root = os.path.expanduser(root)
        self._base_folder = self.root
        self._data_folder = os.path.join(self._base_folder, "2750")

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        super().__init__(self._data_folder, transform=transform, target_transform=target_transform)
        self.root = os.path.expanduser(root)

    def __len__(self) -> int:
        return len(self.samples)

    def _check_exists(self) -> bool:
        return os.path.exists(self._data_folder)

root = '/data/zhangyk/data/eurosat'
dataset = EuroSAT(root)
images = [i[0].replace(f'{root}/2750/', '') for i in dataset.samples]
labels = [i[1] for i in dataset.samples]

from sklearn.model_selection import train_test_split

train_x, val_x, train_y, val_y = train_test_split(images, labels, test_size=0.2, random_state=42)
trainval, test = [(train_x[i], train_y[i]) for i in range(len(train_x))], [(val_x[i], val_y[i]) for i in range(len(val_x))]

import pickle
def save_pickle(file_name, data):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.DEFAULT_PROTOCOL)

save_pickle(f'{root}/annotations.pkl', {'trainval': trainval, 'test': test})
