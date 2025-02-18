# Order-Robust Class Incremental Learning via Graph-Driven Dynamic Similarity Grouping

## Official code repository for CVPR 2025 Published Paper:

Guannan Lai, Yujie Li, Xiangkun Wang, Junbo Zhang, Tianrui Li, Xing Yang(2025). "Order-Robust Class Incremental Learning via Graph-Driven Dynamic Similarity Grouping."

Contact: <aignlai@163.com>

## Environment

This repository is tested in an Anaconda environment. To reproduce exactly, create your environment as follows:

```bash
conda create -y -n GDDSG python=3.9
conda activate GDDSG
conda install -y pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -y -c anaconda pandas==1.5.2
pip install tqdm==4.65.0 
pip install timm==0.6.12
pip install easydict
pip install scikit-learn lightgbm imbalanced-learn optuna
```

## To reproduce results run code of the form

```bash
python main.py -d cifar224
python main.py -d dogs
python main.py -d dogs
python main.py -d dogs
```
