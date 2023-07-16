import os
from typing import Any, Callable, List, Optional, Union, Tuple

root = '/data/zhangyk/data/caltech101/caltech-101'

categories = sorted(os.listdir(os.path.join(root, "101_ObjectCategories")))
categories.remove("BACKGROUND_Google")

index: List[int] = []
y = []
for (i, c) in enumerate(categories):
    n = len(os.listdir(os.path.join(root, "101_ObjectCategories", c)))
    index.extend(range(1, n + 1))
    y.extend(n * [i])

images = []
labels = []
for i in range(len(index)):
    images.append(os.path.join(categories[y[i]], "image_{:04d}.jpg".format(index[i])))
    labels.append(y[i])

from sklearn.model_selection import train_test_split
train_x, val_x, train_y, val_y = train_test_split(images, labels, test_size=0.2, random_state=42)
trainval, test = [(train_x[i], train_y[i]) for i in range(len(train_x))], [(val_x[i], val_y[i]) for i in range(len(val_x))]

import pickle
def save_pickle(file_name, data):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.DEFAULT_PROTOCOL)

save_pickle(f'{root}/annotations.pkl', {'trainval': trainval, 'test': test})

