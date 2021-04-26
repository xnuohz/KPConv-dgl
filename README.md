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