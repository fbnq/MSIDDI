# MSIDDI
## Overview
This repository contains source code for our paper "Graph neural network with sampled mix-hop subgraph for drug–drug interaction prediction"

## Environment Setting
This code is based on Pytorch 1.4.0. You need prepare your virtual enviroment firstly.
Requirement:
```
torch_geometric==1.4.2
scipy==1.4.1
networkx==2.4
torch==1.4.0
chemprop==1.1.0
rdkit==2009.Q1-1
scikit_learn==0.23.2
```

## Pretrain the local representation learning module
In order to avoid inaccurate local representation learning negatively impacts the training of global representation learning module, you can running the pretrain.py file to pretrain the local representation learning module.

You can use the following code to finish pretraining process:
```python
python pretrain.py
```
## Running the code
You can run the following command to re-implement our work:
```python
python main.py
```
