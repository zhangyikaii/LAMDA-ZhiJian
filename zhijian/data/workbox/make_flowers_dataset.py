
import os
from scipy import io
root = '/data/zhangyk/data/flowers'

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

images_path = os.path.join(root, 'jpg')
labels_path = os.path.join(root, 'imagelabels.mat')
labels_mat = io.loadmat(labels_path)
image_labels = []
split = SPLITS['all']
images, labels = [], []
for idx, label in enumerate(labels_mat['labels'][0], start=1):
    if label in split:
        image = str(idx).zfill(5)
        image = f'image_{image}.jpg'
        # image = os.path.join(images_path, image)
        label = split.index(label)
        images.append(image)
        labels.append(label)

from sklearn.model_selection import train_test_split

train_x, val_x, train_y, val_y = train_test_split(images, labels, test_size=0.2, random_state=42)
trainval, test = [(train_x[i], train_y[i]) for i in range(len(train_x))], [(val_x[i], val_y[i]) for i in range(len(val_x))]

import pickle
def save_pickle(file_name, data):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.DEFAULT_PROTOCOL)

save_pickle(f'{root}/annotations.pkl', {'trainval': trainval, 'test': test})
