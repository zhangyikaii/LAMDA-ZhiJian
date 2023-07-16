import os

from PIL import Image
from torch.utils.data import Dataset
from zhijian.models.utils import load_pickle

# Splits from "Meta-Datasets", Triantafillou et al, 2020
SPLITS = {
    'train': [90, 38, 80, 30, 29, 12, 43, 27, 4, 64, 31, 99, 8, 67, 95, 77,
              78, 61, 88, 74, 55, 32, 21, 13, 79, 70, 51, 69, 14, 60, 11, 39,
              63, 37, 36, 28, 48, 7, 93, 2, 18, 24, 6, 3, 44, 76, 75, 72, 52,
              84, 73, 34, 54, 66, 59, 50, 91, 68, 100, 71, 81, 101, 92, 22,
              33, 87, 1, 49, 20, 25, 58],
    'validation': [10, 16, 17, 23, 26, 47, 53, 56, 57, 62, 82, 83, 86, 97, 102],
    'test': [5, 9, 15, 19, 35, 40, 41, 42, 45, 46, 65, 85, 89, 94, 96, 98],
    'all': list(range(1, 103))
}


class Flowers(Dataset):
    def __init__(self, root, train, mode='all', transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.split = 'trainval' if train else 'test'
        self.load_data(mode)

    def _check_exists(self):
        return os.path.exists(self.root)

    def load_data(self, mode):
        # images_path = os.path.join(self.root, 'jpg')
        # labels_path = os.path.join(self.root, 'imagelabels.mat')
        # labels_mat = scipy.io.loadmat(labels_path)
        # image_labels = []
        # split = SPLITS[mode]
        # for idx, label in enumerate(labels_mat['labels'][0], start=1):
        #     if label in split:
        #         image = str(idx).zfill(5)
        #         image = f'image_{image}.jpg'
        #         image = os.path.join(images_path, image)
        #         label = split.index(label)
        #         image_labels.append((image, label))
        # self.data = image_labels
        data = load_pickle(os.path.join(self.root, 'annotations.pkl'))[self.split]
        self.samples = [(os.path.join(self.root, 'jpg', i[0]), i[1]) for i in data]

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
