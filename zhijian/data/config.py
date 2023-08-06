def add_prefix_to_dict(d, prefix_list):
    d_to_add = {}
    for i_prefix in prefix_list:
        d_to_add.update({f'{i_prefix}{key}': value for key, value in d.items()})
    d.update(d_to_add)

DATA_PATH_SUB_DIR = {
    'CIFAR-100': 'cifar',
    'CLEVR-Count': 'clevr_count',
    'CLEVR-Distance': 'clevr_dist',
    'Caltech101': 'caltech101',
    'DTD': 'dtd',
    'Diabetic-Retinopathy': 'diabetic_retinopathy',
    'Dmlab': 'dmlab',
    'EuroSAT': 'eurosat',
    'KITTI': 'kitti',
    'Oxford-Flowers-102': 'oxford_flowers102',
    'Oxford-IIIT-Pet': 'oxford_iiit_pet',
    'PatchCamelyon': 'patch_camelyon',
    'RESISC45': 'resisc45',
    'SUN397': 'sun397',
    'SVHN': 'svhn',
    'dSprites-Location': 'dsprites_loc',
    'dSprites-Orientation': 'dsprites_ori',
    'smallNORB-Azimuth': 'smallnorb_azi',
    'smallNORB-Elevation': 'smallnorb_ele',
}

add_prefix_to_dict(DATA_PATH_SUB_DIR, ['VTAB-1k.', 'VTAB-tuning.'])

DATA_PATH_SUB_DIR.update({
    'ImageNet': 'imagenet',
    'ImageNet-A': 'imagenet-a',
    'ImageNet-R': 'imagenet-r',
    'CIFAR-10': 'cifar10',
    'Aircraft': 'aircraft',
    'Stanford-Cars': 'cars',
    'CUB-200-2011': 'cub2011',
    'Dogs': 'dogs',
    'Food': 'food/food-101',
    'NABirds': 'nabirds',
    'OfficeHome': 'officehome',
    'PACS': 'pacs',
    'STL10': 'stl10',
    'FFCV-ImageNet': 'imagenet',
    'AID': 'AID',
    'DomainNet': 'domainnet',
    'DomainNet-clipart': 'domainnet',
    'DomainNet-infograph': 'domainnet',
    'DomainNet-painting': 'domainnet',
    'DomainNet-quickdraw': 'domainnet',
    'DomainNet-real': 'domainnet',
    'DomainNet-sketch': 'domainnet',
})

DATASET2SUB_CLASSES = {
    'ImageNet-A': [
        6, 11, 13, 15, 17, 22, 23, 27, 30, 37, 39, 42, 47, 50, 57, 70, 71, 76, 79, 89, 90, 94, 96, 97, 99, 105, 107,
        108, 110,
        113, 124, 125, 130, 132, 143, 144, 150, 151, 207, 234, 235, 254, 277, 283, 287, 291, 295, 298, 301, 306, 307,
        308, 309,
        310, 311, 313, 314, 315, 317, 319, 323, 324, 326, 327, 330, 334, 335, 336, 347, 361, 363, 372, 378, 386, 397,
        400, 401,
        402, 404, 407, 411, 416, 417, 420, 425, 428, 430, 437, 438, 445, 456, 457, 461, 462, 470, 472, 483, 486, 488,
        492, 496,
        514, 516, 528, 530, 539, 542, 543, 549, 552, 557, 561, 562, 569, 572, 573, 575, 579, 589, 606, 607, 609, 614,
        626, 627,
        640, 641, 642, 643, 658, 668, 677, 682, 684, 687, 701, 704, 719, 736, 746, 749, 752, 758, 763, 765, 768, 773,
        774, 776,
        779, 780, 786, 792, 797, 802, 803, 804, 813, 815, 820, 823, 831, 833, 835, 839, 845, 847, 850, 859, 862, 870,
        879, 880,
        888, 890, 897, 900, 907, 913, 924, 932, 933, 934, 937, 943, 945, 947, 951, 954, 956, 957, 959, 971, 972, 980,
        981, 984,
        986, 987, 988],
    'ImageNet-R': [
        1, 2, 4, 6, 8, 9, 11, 13, 22, 23, 26, 29, 31, 39, 47, 63, 71, 76, 79, 84, 90, 94, 96, 97, 99, 100, 105, 107,
        113, 122,
        125, 130, 132, 144, 145, 147, 148, 150, 151, 155, 160, 161, 162, 163, 171, 172, 178, 187, 195, 199, 203,
        207, 208, 219,
        231, 232, 234, 235, 242, 245, 247, 250, 251, 254, 259, 260, 263, 265, 267, 269, 276, 277, 281, 288, 289,
        291, 292, 293,
        296, 299, 301, 308, 309, 310, 311, 314, 315, 319, 323, 327, 330, 334, 335, 337, 338, 340, 341, 344, 347,
        353, 355, 361,
        362, 365, 366, 367, 368, 372, 388, 390, 393, 397, 401, 407, 413, 414, 425, 428, 430, 435, 437, 441, 447,
        448, 457, 462,
        463, 469, 470, 471, 472, 476, 483, 487, 515, 546, 555, 558, 570, 579, 583, 587, 593, 594, 596, 609, 613,
        617, 621, 629,
        637, 657, 658, 701, 717, 724, 763, 768, 774, 776, 779, 780, 787, 805, 812, 815, 820, 824, 833, 847, 852,
        866, 875, 883,
        889, 895, 907, 928, 931, 932, 933, 934, 936, 937, 943, 945, 947, 948, 949, 951, 953, 954, 957, 963, 965,
        967, 980, 981,
        983, 988]
}


DATASET2NUM_CLASSES = {
    'CIFAR-10': 10,
    'CIFAR-100': 100,
    'CLEVR-Count': 8,
    'CLEVR-Distance': 6,
    'Caltech101': 102,
    'DTD': 47,
    'Diabetic-Retinopathy': 5,
    'Dmlab': 6,
    'EuroSAT': 10,
    'KITTI': 4,
    'Oxford-Flowers-102': 102,
    'Oxford-IIIT-Pet': 37,
    'PatchCamelyon': 2,
    'RESISC45': 45,
    'SUN397': 397,
    'SVHN': 10,
    'dSprites-Location': 16,
    'dSprites-Orientation': 16,
    'smallNORB-Azimuth': 18,
    'smallNORB-Elevation': 9,
}


add_prefix_to_dict(DATASET2NUM_CLASSES, ['VTAB-1k.', 'VTAB-tuning.'])

DATASET2NUM_CLASSES.update({
    'ImageNet': 1000,
    'Customized': 5
})

IGNORE_INDEX = -100
