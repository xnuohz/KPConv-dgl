import torch
import os
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt


def square_distance(src, dst):
    r"""

    Description
    -----------
    Computes distances for src points -> dst points

    Parameters
    -----------
    src: torch.Tensor
        [N, 3]
    dst: torch.Tensor
        [N, 3]
    
    Return
    -----------
    dist: torch.Tensor
        [N, N]
    """
    N, _ = src.shape
    M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(1, 0))
    dist += torch.sum(src ** 2, -1).view(N, 1)
    dist += torch.sum(dst ** 2, -1).view(1, M)
    return dist


def batch_neighbors(queries,
                    supports,
                    q_batches,
                    s_batches,
                    radius):
    r"""

    Description
    -----------
    Computes neighbors for a batch of point clouds
    
    Parameters
    -----------
    queries: torch.Tensor
        [N, 3], batched point clouds(center node)
    supports: torch.Tensor
        [N, 3], batched point clouds(neighbors set)
    q_batches: torch.Tensor
        [B,] the list of lengths of batched point clouds
    s_batches: torch.Tensor
        [B,] the list of lengths of batched point clouds
    radius: float32

    Return
    -----------
    stacked_src: list
        [[N1,], [N2,], ...]
    stacked_dst: list
        [[N1,], [N2,], ...]
    """
    
    stacked_src, stacked_dst = [], []
    
    for i in range(1, len(q_batches)):
        query_cloud = queries[int(q_batches[i - 1]):int(q_batches[i])]  # [N, 3]
        support_cloud = supports[int(s_batches[i - 1]):int(s_batches[i])]  # [M, 3]
        N, M = len(query_cloud), len(support_cloud)
        # get neighbors for each point
        dists = square_distance(query_cloud, support_cloud)  # [N, M]
        group_idx = torch.arange(M, dtype=torch.long).view(1, -1).repeat(N, 1)
        group_idx[dists > radius ** 2] = M
        # get edges idx
        dst, src = torch.where(group_idx != M)
        # we should include self-loop edge
        stacked_src.append(src)
        stacked_dst.append(dst)
    
    return stacked_src, stacked_dst


def get_bbox(points):
    r"""
    
    Description
    -----------
    Get the bounding box of a point cloud
    
    Parameters
    ----------
    points: torch.Tensor
        [N, 3] matrix of input points
    """
    min_point = torch.min(points, 0)[0].data
    max_point = torch.max(points, 0)[0].data
    
    return min_point, max_point


def grid_subsampling(points, dl, features=None):
    r"""
    Description
    -----------
    Grid subsampling is implemented by C++ in the author's PyTorch code
    
    Parameters
    ----------
    points: torch.Tensor
        [N, 3] matrix of input points
    dl: float
        the size of grid voxels
    features: torch.Tensor
        [N, d] matrix of input features

    Return
    ----------
    subsampled points and features
    """
    n_points = len(points)
    min_corner, max_corner = get_bbox(points)
    # is this nessesary?
    min_corner = torch.floor(torch.div(min_corner, dl)) * dl
    
    sample_nx = int((max_corner[0] - min_corner[0]) / dl) + 1
    sample_ny = int((max_corner[1] - min_corner[1]) / dl) + 1

    data = defaultdict(list)

    if features is not None:
        points = torch.cat([points, features], dim=-1)

    for p in points:
        idx_x = int((p[0] - min_corner[0]) / dl)
        idx_y = int((p[1] - min_corner[1]) / dl)
        idx_z = int((p[2] - min_corner[2]) / dl)
        idx = idx_x + sample_nx * idx_y + sample_nx * sample_ny * idx_z
        data[idx].append(p.view(1, -1))
    
    subsampled_data = []
    
    for _, v in data.items():
        v = torch.cat(v)
        subsampled_data.append(torch.mean(v, dim=0).view(1, -1))

    subsampled_data = torch.cat(subsampled_data)

    if features is not None:
        return subsampled_data[:, :3], subsampled_data[:, 3:]
    else:
        return subsampled_data


def batch_grid_subsampling(stacked_points, stacked_lengths, dl, offset=5):
    r"""
    
    Description
    -----------
    Create barycenters from batched points for the next layer by batch grid subsampling

    Parameters
    -----------
    stacked_points: torch.Tensor
        [N, 3], batched point clouds
    stacked_lengths: torch.Tensor
        [B,] the list of lengths of batched point clouds
    dl: float32
        the size of grid voxels

    Return
    -----------
    pool_p: torch.Tensor
        [[N1, 3], [N2, 3], ...]
    pool_b: torch.Tensor
        [N1, N2, ...] 
    """
    # cumsum -> not cumsum
    batch_length = stacked_lengths[1:] - stacked_lengths[:-1]
    # assume that stacked_points, stacked_lengths are on CPU before training
    # +offset -> grid subsampling simultaneously -> -offset
    offsets = torch.arange(0, len(batch_length) * offset, offset)
    # each offset will be repeated by the number of each point cloud
    stacked_offsets = offsets.repeat_interleave(batch_length).reshape(-1, 1)  # [batch, 1]
    stacked_offset_points = stacked_points + stacked_offsets  # [batch, 3]
    pool_points = grid_subsampling(stacked_offset_points, dl)
    # calculate pool batch length
    tmp_points = torch.cat([pool_points, torch.zeros(1, pool_points.size()[1]).fill_(float('inf'))], dim=0)
    # assume that there exists a gap between each point cloud
    gap = torch.abs(tmp_points[1:, :] - tmp_points[:-1, :]) >= offset - 2
    pool_cumsum_batch = torch.cat([torch.zeros(1), torch.where(gap[:, 0] == True)[0] + 1])
    pool_batch_len = (pool_cumsum_batch[1:] - pool_cumsum_batch[:-1]).long()
    # back to the origin scale
    pool_offsets = torch.arange(0, len(pool_batch_len) * offset, offset)
    pool_batch_offsets = pool_offsets.repeat_interleave(pool_batch_len).reshape(-1, 1)
    pool_points = pool_points - pool_batch_offsets
    # back to cumsum for the following layer
    pool_batch_len = torch.cumsum(torch.cat([torch.LongTensor([0]), pool_batch_len]), dim=0)

    return pool_points, pool_batch_len


def kernel_point_optimization_debug(radius, num_points, num_kernels=1, dimension=3,
                                    fixed='center', ratio=0.66, verbose=0):
    """
    From the author's code
    Creation of kernel point via optimization of potentials.
    :param radius: Radius of the kernels
    :param num_points: points composing kernels
    :param num_kernels: number of wanted kernels
    :param dimension: dimension of the space
    :param fixed: fix position of certain kernel points ('none', 'center' or 'verticals')
    :param ratio: ratio of the radius where you want the kernels points to be placed
    :param verbose: display option
    :return: points [num_kernels, num_points, dimension]
    """

    #######################
    # Parameters definition
    #######################

    # Radius used for optimization (points are rescaled afterwards)
    radius0 = 1
    diameter0 = 2

    # Factor multiplicating gradients for moving points (~learning rate)
    moving_factor = 1e-2
    continuous_moving_decay = 0.9995

    # Gradient threshold to stop optimization
    thresh = 1e-5

    # Gradient clipping value
    clip = 0.05 * radius0

    #######################
    # Kernel initialization
    #######################

    # Random kernel points
    kernel_points = np.random.rand(num_kernels * num_points - 1, dimension) * diameter0 - radius0
    while (kernel_points.shape[0] < num_kernels * num_points):
        new_points = np.random.rand(num_kernels * num_points - 1, dimension) * diameter0 - radius0
        kernel_points = np.vstack((kernel_points, new_points))
        d2 = np.sum(np.power(kernel_points, 2), axis=1)
        kernel_points = kernel_points[d2 < 0.5 * radius0 * radius0, :]
    kernel_points = kernel_points[:num_kernels * num_points, :].reshape((num_kernels, num_points, -1))

    # Optionnal fixing
    if fixed == 'center':
        kernel_points[:, 0, :] *= 0
    if fixed == 'verticals':
        kernel_points[:, :3, :] *= 0
        kernel_points[:, 1, -1] += 2 * radius0 / 3
        kernel_points[:, 2, -1] -= 2 * radius0 / 3

    #####################
    # Kernel optimization
    #####################

    # Initialize figure
    if verbose>1:
        fig = plt.figure()

    saved_gradient_norms = np.zeros((10000, num_kernels))
    old_gradient_norms = np.zeros((num_kernels, num_points))
    step = -1
    while step < 10000:

        # Increment
        step += 1

        # Compute gradients
        # *****************

        # Derivative of the sum of potentials of all points
        A = np.expand_dims(kernel_points, axis=2)
        B = np.expand_dims(kernel_points, axis=1)
        interd2 = np.sum(np.power(A - B, 2), axis=-1)
        inter_grads = (A - B) / (np.power(np.expand_dims(interd2, -1), 3/2) + 1e-6)
        inter_grads = np.sum(inter_grads, axis=1)

        # Derivative of the radius potential
        circle_grads = 10*kernel_points

        # All gradients
        gradients = inter_grads + circle_grads

        if fixed == 'verticals':
            gradients[:, 1:3, :-1] = 0

        # Stop condition
        # **************

        # Compute norm of gradients
        gradients_norms = np.sqrt(np.sum(np.power(gradients, 2), axis=-1))
        saved_gradient_norms[step, :] = np.max(gradients_norms, axis=1)

        # Stop if all moving points are gradients fixed (low gradients diff)

        if fixed == 'center' and np.max(np.abs(old_gradient_norms[:, 1:] - gradients_norms[:, 1:])) < thresh:
            break
        elif fixed == 'verticals' and np.max(np.abs(old_gradient_norms[:, 3:] - gradients_norms[:, 3:])) < thresh:
            break
        elif np.max(np.abs(old_gradient_norms - gradients_norms)) < thresh:
            break
        old_gradient_norms = gradients_norms

        # Move points
        # ***********

        # Clip gradient to get moving dists
        moving_dists = np.minimum(moving_factor * gradients_norms, clip)

        # Fix central point
        if fixed == 'center':
            moving_dists[:, 0] = 0
        if fixed == 'verticals':
            moving_dists[:, 0] = 0

        # Move points
        kernel_points -= np.expand_dims(moving_dists, -1) * gradients / np.expand_dims(gradients_norms + 1e-6, -1)

        if verbose:
            print('step {:5d} / max grad = {:f}'.format(step, np.max(gradients_norms[:, 3:])))
        if verbose > 1:
            plt.clf()
            plt.plot(kernel_points[0, :, 0], kernel_points[0, :, 1], '.')
            circle = plt.Circle((0, 0), radius, color='r', fill=False)
            fig.axes[0].add_artist(circle)
            fig.axes[0].set_xlim((-radius*1.1, radius*1.1))
            fig.axes[0].set_ylim((-radius*1.1, radius*1.1))
            fig.axes[0].set_aspect('equal')
            plt.draw()
            plt.pause(0.001)
            plt.show(block=False)
            print(moving_factor)

        # moving factor decay
        moving_factor *= continuous_moving_decay

    # Remove unused lines in the saved gradients
    if step < 10000:
        saved_gradient_norms = saved_gradient_norms[:step+1, :]

    # Rescale radius to fit the wanted ratio of radius
    r = np.sqrt(np.sum(np.power(kernel_points, 2), axis=-1))
    kernel_points *= ratio / np.mean(r[:, 1:])

    # Rescale kernels with real radius
    return kernel_points * radius, saved_gradient_norms


def load_kernels(radius, kernel_size, p_dim, fixed):
    kernel_dir = 'kernels'

    if not os.path.exists(kernel_dir):
        os.makedirs(kernel_dir)
    
    kernel_file = f'{kernel_dir}/kp_{kernel_size}_{p_dim}_{radius}_{fixed}.npy'

    if not os.path.exists(kernel_file):
        # Create kernels
        kernel_points, grad_norms = kernel_point_optimization_debug(1.0,
                                                                    kernel_size,
                                                                    num_kernels=100,
                                                                    dimension=p_dim,
                                                                    fixed=fixed)
        # Find best candidate
        best_k = np.argmin(grad_norms[-1, :])

        # Save points
        kernel_points = kernel_points[best_k, :, :]
        np.save(kernel_file, kernel_points)
    else:
        kernel_points = np.load(kernel_file)
    
    # Random rorations for the kernel
    R = np.eye(p_dim)
    theta = np.random.rand() * 2 * np.pi
    # currently we only support that fixed is 'center', p_dim is 3 and kernel_size <= 30
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    # Add a small noise
    kernel_points = kernel_points + np.random.normal(scale=0.01, size=kernel_points.shape)
    # Scale kernels
    kernel_points = radius * kernel_points
    # Rotate kernels
    kernel_points = np.matmul(kernel_points, R)
    
    return kernel_points