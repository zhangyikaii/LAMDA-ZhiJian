from collections import defaultdict
import os
from shutil import copy

def prepare_data(filepath, src, dest):
    classes_images = defaultdict(list)
    with open(filepath, 'r') as txt:
        paths = [read.strip() for read in txt.readlines()]
        for p in paths:
            food = p.split('/')
            classes_images[food[0]].append(food[1] + '.jpg')

    for food in classes_images.keys():
        print("\nCopying images into ", food)
        if not os.path.exists(os.path.join(dest, food)):
            os.makedirs(os.path.join(dest, food))
        for i in classes_images[food]:
            copy(os.path.join(src, food, i), os.path.join(dest, food, i))
    print("Copying Done!")


FOOD_PATH = "/data/zhangyk/data/food/food-101"
IMG_PATH = FOOD_PATH+"/images"
META_PATH = FOOD_PATH+"/meta"
TRAIN_PATH = FOOD_PATH+"/train"
TEST_PATH = FOOD_PATH+"/test"

prepare_data(f'{META_PATH}/train.txt', IMG_PATH, TRAIN_PATH)
prepare_data(f'{META_PATH}/test.txt', IMG_PATH, TEST_PATH)
