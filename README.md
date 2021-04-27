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