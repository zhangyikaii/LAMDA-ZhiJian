from setuptools import setup, find_packages

setup(
    name='zhijian',
    version='0.0.1',
    description='ZhiJian: A Unifying and Rapidly Deployable Toolbox for Pre-trained Model Reuse',
    author='ZhiJian Contributors',
    author_email='yumzhangyk@gmail.com',
    packages=find_packages(),
    keywords='pytorch pretrained model reuse',
    url='https://github.com/zhangyikaii/LAMDA-ZhiJian',
    install_requires=[
        'datasets',
        'numpy',
        'pytorch_metric_learning',
        'PyYAML',
        'tensorboardX',
        'torch>=1.7',
        'torchvision',
        'tqdm',
        'transformers',
        'timm',
    ],
    python_requires='>=3.7',
)