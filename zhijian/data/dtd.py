from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path

class DTD(Dataset):
    def __init__(self, root, train, partition=1, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self._split = ['train', 'val'] if train else ['test']

        self._image_files = []
        classes = []

        self._base_folder = Path(self.root)
        self._data_folder = self._base_folder / "dtd"
        self._meta_folder = self._data_folder / "labels"
        self._images_folder = self._data_folder / "images"
        for split in self._split:
            with open(self._meta_folder / f"{split}{partition}.txt") as file:
                for line in file:
                    cls, name = line.strip().split("/")
                    self._image_files.append(self._images_folder.joinpath(cls, name))
                    classes.append(cls)

        self.classes = sorted(set(classes))
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))
        self._labels = [self.class_to_idx[cls] for cls in classes]

        self.samples = [(i, j) for i, j in zip(self._image_files, self._labels)]


    def __getitem__(self, idx):
        image_file, label = self.samples[idx]
        image = Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return len(self.samples)
