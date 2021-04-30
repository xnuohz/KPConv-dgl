import numpy as np
import torch.nn as nn
import dgl
from dataset import grid_subsampling


def square_distance(points):
    r"""
    points: [n, 3] -> [n, n]
    """
    N, _ = points.shape
    dist = -2 * torch.matmul(points, points.permute(1, 0))
    dist += torch.sum(points ** 2, -1).view(N, 1)
    dist += torch.sum(points ** 2, -1).view(1, N)
    return dist


class FixedRadiusNNGraph(nn.Module):
    r"""
    Build batched nn graph
    """
    def __init__(self, radius):
        super(FixedRadiusNNGraph, self).__init__()
        self.radius = radius

    def forward(self, batch_points, batch_feats, batch_len):
        batch_g = []

        for i in range(len(batch_len) - 1):
            current_cloud = batch_points[batch_len[i]:batch_len[i + 1]]
            num_points = len(current_cloud)
            # get neighbors for each point
            dists = square_distance(current_cloud)
            group_idx = torch.arange(num_points, dtype=torch.long).view(1, -1).repeat(num_points, 1)
            group_idx[dists > self.radius ** 2] = num_points
            # create undirected graph
            src, dst = torch.where(group_idx != num_points)
            g = dgl.graph((src, dst))
            g.ndata['pos'] = current_cloud
            g.ndata['feat'] = batch_feats[batch_len[i]:batch_len[i + 1]]
            g = dgl.to_bidirected(g)
            batch_g.append(g)
        
        return dgl.batch(batch_g)


class BatchGridSubsampling(nn.Module):
    r"""
    Create barycenters from batched points for the next layer by batch grid subsampling
    """
    def __init__(self, dl):
        super(BatchGridSubsampling, self).__init__()
        self.dl = dl

    def forward(self, batch_points, batch_feats, batch_len):
        pool_points, pool_feats, pool_batch = [], [], []

        for i in range(len(batch_len) - 1):
            current_cloud = batch_points[batch_len[i]:batch_len[i + 1]]
            current_feat = batch_feats[batch_len[i]:batch_len[i + 1]]
            ps, feats = grid_subsampling(current_cloud, current_feat, self.dl)
            pool_points.append(ps)
            pool_feats.append(feats)
            pool_batch.append(len(ps))
        
        pool_points = torch.FloatTensor(np.concatenate(pool_points))
        pool_feats = torch.FloatTensor(np.concatenate(pool_feats))
        pool_batch = np.concatenate([[0], np.cumsum(pool_batch)])

        return pool_points, pool_feats, pool_batch


class KPConv(nn.Module):
    def __init__(self):
        super(KPConv, self).__init__()
        pass

    def forward(self):
        pass


if __name__ == '__main__':
    import torch
    from dataset import grid_subsampling
    p = torch.Tensor([[1, 2, 3], [4, 5, 6], [23, 45, 30]])
    feat = torch.arange(9).view(3, -1)
    print(square_distance(p))

    fnn = BatchGridSubsampling(30)
    x = fnn(p, feat, [0, 3])
    print(x)