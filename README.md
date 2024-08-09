## Environment

This repository is tested in an Anaconda environment. To reproduce exactly, create your environment as follows:

```
conda install -y pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -y -c anaconda pandas==1.5.2
pip install tqdm==4.65.0 
pip install timm==0.6.12
pip install easydict
```

## To reproduce results run code of the form

```
python main.py -d cifar224
python main.py -d dogs
python main.py -d cub
python main.py -d omnibenchmark
```

- except for cifar224, data will need to be downloaded and moved to relative paths at "./data/dataset_name/train/" and "./data/dataset_name/test/" -- see data.py
