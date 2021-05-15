import torch
from collections import defaultdict


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
    
    for i in range(len(q_batches) - 1):
        query_cloud = queries[int(q_batches[i]):int(q_batches[i + 1])]  # [N, 3]
        support_cloud = supports[int(s_batches[i]):int(s_batches[i + 1])]  # [M, 3]
        N, M = len(query_cloud), len(support_cloud)
        # get neighbors for each point
        dists = square_distance(query_cloud, support_cloud)  # [N, M]
        group_idx = torch.arange(M, dtype=torch.long).view(1, -1).repeat(N, 1)
        group_idx[dists > self.radius ** 2] = M
        # get edges idx
        src, dst = torch.where(group_idx != num_points)
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


def batch_grid_subsampling(stacked_points, stack_lengths, dl, offset=5):
    r"""
    
    Description
    -----------
    Create barycenters from batched points for the next layer by batch grid subsampling

    Parameters
    -----------
    stacked_points: torch.Tensor
        [N, 3], batched point clouds
    stack_lengths: torch.Tensor
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
    # assume that stacked_points, stack_lengths are on CPU before training
    # +offset -> grid subsampling simultaneously -> -offset
    offsets = torch.arange(0, len(stack_lengths) * offset, offset)
    # each offset will be repeated by the number of each point cloud
    stacked_offsets = offsets.repeat_interleave(stack_lengths).reshape(-1, 1)  # [batch, 1]
    stacked_offset_points = stacked_points + stacked_offsets  # [batch, 3]
    
    pool_points = grid_subsampling(stacked_offset_points, dl)
    # calculate pool batch length
    tmp_points = torch.cat([pool_points, torch.zeros(1, pool_points.size()[1])], dim=0)
    # assume that there exists a gap between each point cloud
    gap = torch.abs(tmp_points[1:, :] - tmp_points[:-1, :]) >= offset - 2
    pool_cumsum_batch = torch.cat([torch.zeros(1), torch.where(gap[:, 0] == True)[0] + 1])
    pool_batch_len = (pool_cumsum_batch[1:] - pool_cumsum_batch[:-1]).long()
    # back to the origin scale
    pool_offsets = torch.arange(0, len(pool_batch_len) * offset, offset)
    pool_batch_offsets = pool_offsets.repeat_interleave(pool_batch_len).reshape(-1, 1)
    pool_points = pool_points - pool_batch_offsets

    return pool_points, pool_batch_len
