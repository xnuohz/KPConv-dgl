# DGL Implementation of KPConv

This DGL example implements the GNN model proposed in the paper [KPConv: Flexible and Deformable Convolution for Point Clouds](https://arxiv.org/abs/1904.08889). For the original implementation, see [here](https://github.com/HuguesTHOMAS/KPConv-PyTorch).

Contributor: [xnuohz](https://github.com/xnuohz)

### Requirements
The codebase is implemented in Python 3.7. For version requirement of packages, see below.

```
dgl 0.6.0
torch 1.7.0
```

### DataLoader

- grid sampling
- get item
  * augmentation
  * classification inputs
    - batch grid subsampling
    - batch neighbors
- batch neighbors
  * kd tree
- as layer goes deeper
  * points -> grid_sampling(points)
  * batch_neighbors(points) -> batch_neighbors(grid_sampling(points))

### Tips

- subsampling is used to reduce both the total number of points and the number of points in the radius neighbors.
- augmentation made the grid subsampling happening on a randomly oriented grid
- limits?
- before each KPConv:
  * batch neighbors is needed for every points in each point cloud
  * batch grid subsampling is needed while going to the next layer
  * conv_i is neighbors of stacked_points
  * pool_i is neighbors of center points which is grid subsampled by stacked_points
  * stacked_lengths and pool_b are both batch information
- currently, batch_len only support on cpu
