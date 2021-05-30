# DGL Implementation of KPConv

This DGL example implements the GNN model proposed in the paper [KPConv: Flexible and Deformable Convolution for Point Clouds](https://arxiv.org/abs/1904.08889). For the original implementation, see [here](https://github.com/HuguesTHOMAS/KPConv-PyTorch).

Contributor: [xnuohz](https://github.com/xnuohz)

### Requirements
The codebase is implemented in Python 3.7. For version requirement of packages, see below.

```
dgl 0.6.0.post1
torch 1.7.0
logzero 1.7.0
```

### The dataset used in this example

ModelNet10 for classification. Dataset summary:

* Number of point clouds: 3,991(train), 908(test)
* Number of classes: 10

ModelNet40 for classification. Dataset summary:

* Number of point clouds: 9,843(train), 2,468(test)
* Number of classes: 40

### Usage

**Note: we only support KPConv rigid in this example.**

Train a model which follows the original hyperparameters
```bash
# ModelNet10
python main.py --epochs 150

# ModelNet40
python main.py --data-type large --epochs 150
```

### Performance

|    Dataset     | ModelNet10 | ModelNet40 |
| :------------: | :--------: | :--------: |
| Result(Paper)  |     -      |    92.9    |
| Result(Author) |            |    91.6    |
|  Result(DGL)   |   90.09    |            |

### Speed

|    Dataset     | ModelNet10 | ModelNet40 |
| :------------: | :--------: | :--------: |
| Result(Author) |            |   59.87    |
|  Result(DGL)   |   363.70   |            |
