# Computational Efficiency Radar Object Detection based on Densely Connected Residual Block
Chih-Chung Hsu, et. al.

Official implementation of our DCSN-radar object detector.
Our source is based on [RODNet](https://github.com/yizhou-wang/RODNet), thanks for the authors' contribution and great effort.

[[Paper (soon)]]()
[[Dataset]](https://www.cruwdataset.org)

![Training loss](https://cchsu.info/files/training%20loss%20of%20DCSN-radar.png)
<center>Fig.1. Training loss curves of the proposed DCSN and RODNet comparison (DCSN: Red, RODNet: Orange)</center>

## Installation

Create a conda environment for dcsn
```
conda create -n dcsn python=3.7 -y
conda activate dcsn
```

Install pytorch.
```
conda install pytorch torchvision -c pytorch
```

Install `cruw-devkit` package (needed for accessing the dataset)
Please refer to [`cruw-devit`](https://github.com/yizhou-wang/cruw-devkit) repository for detailed instructions.
```
git clone https://github.com/yizhou-wang/cruw-devkit.git
cd cruw-devkit
pip install -e .
cd ..
```

Setup our DCSN package (same as RODNet does).
```
pip install -e .
```

## Prepare data for our DCSN (same as RODNet does).
Note that you should set the configuration files like [configs/DCSN.py](configs/DCSN.py) to adopt DCSN architecture. 

```
python tools/prepare_dataset/prepare_data.py \
        --config configs/<CONFIG_FILE> \
        --data_root <DATASET_ROOT> \
        --split train,test \
        --out_data_dir data/<DATA_FOLDER_NAME>
```

## Train models

```
python tools/train.py --config configs/<CONFIG_FILE> \
        --data_dir data/<DATA_FOLDER_NAME> \
        --log_dir checkpoints/
```

## Inference

```
python tools/test.py --config configs/<CONFIG_FILE> \
        --data_dir data/<DATA_FOLDER_NAME> \
        --checkpoint <CHECKPOINT_PATH> \
        --res_dir results/
```
