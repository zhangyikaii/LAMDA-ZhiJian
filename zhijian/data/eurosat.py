import os

from PIL import Image
from torch.utils.data import Dataset
from zhijian.models.utils import load_pickle


class EuroSAT(Dataset):
    def __init__(self, root, train, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.split = 'trainval' if train else 'test'
        data = load_pickle(os.path.join(self.root, 'annotations.pkl'))[self.split]
        self.samples = [(os.path.join(self.root, '2750', i[0]), i[1]) for i in data]

    def __getitem__(self, i):
        image, label = self.samples[i]
        image = Image.open(image).convert('RGB')
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def __len__(self):
        return len(self.samples)
