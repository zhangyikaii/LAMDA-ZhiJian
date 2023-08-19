from zhijian.data.config import DATA_PATH_SUB_DIR
import os
import json

datasets = [
    'CIFAR-100',
    'CLEVR-Count',
    'CLEVR-Distance',
    'Caltech101',
    'DTD',
    'Diabetic-Retinopathy',
    'Dmlab',
    'EuroSAT',
    'KITTI',
    'Oxford-Flowers-102',
    'Oxford-IIIT-Pet',
    'PatchCamelyon',
    'RESISC45',
    'SVHN',
    'dSprites-Location',
    'dSprites-Orientation',
    'smallNORB-Azimuth',
    'smallNORB-Elevation'
]

data_url = '/data/zhangyk/data/petl'

class_map = {}
base_idx = 0
for i_dataset in sorted(datasets):
    print(i_dataset)
    train_txt = os.path.join(data_url, DATA_PATH_SUB_DIR[i_dataset], 'train800val200.txt')
    labels = []
    with open(train_txt, 'r') as f:
        for line in f:
            img_name = line.split(' ')[0]
            label = int(line.split(' ')[1])
            labels.append(label)
    
    cur_map = {int(k): v for k, v in zip(sorted(set(labels)), range(base_idx, base_idx + len(set(labels))))}
    base_idx = base_idx + len(set(labels))
    class_map[i_dataset] = cur_map

import pickle
def save_pickle(file_name, data):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.DEFAULT_PROTOCOL)

save_pickle('vmt_class_map.pkl', class_map)

with open('vmt_class_map.json', 'w') as f:
    json.dump(class_map, f)


merged = []
for i_dataset in sorted(datasets):
    train_txt = os.path.join(data_url, DATA_PATH_SUB_DIR[i_dataset], 'train800val200.txt')
    with open(train_txt, 'r') as f:
        for line in f:
            img_name = f'{DATA_PATH_SUB_DIR[i_dataset]}/{line.split(" ")[0]}'
            label = int(line.split(' ')[1])
            merged.append((img_name, class_map[i_dataset][label]))

with open('train19x800val19x200.txt', 'w') as f:
    for i in merged:
        f.write(f'{i[0]} {i[1]}\n')

test_merged = []
for i_dataset in sorted(datasets):
    test_txt = os.path.join(data_url, DATA_PATH_SUB_DIR[i_dataset], 'test.txt')
    with open(test_txt, 'r') as f:
        for line in f:
            img_name = f'{DATA_PATH_SUB_DIR[i_dataset]}/{line.split(" ")[0]}'
            label = int(line.split(' ')[1])
            # print(img_name, class_map[i_dataset][label])
            test_merged.append((img_name, class_map[i_dataset][label]))

with open('test.txt', 'w') as f:
    for i in test_merged:
        f.write(f'{i[0]} {i[1]}\n')
