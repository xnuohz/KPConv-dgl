import math
import numpy as np
import torch
import torch.nn as nn
import dgl
import dgl.function as fn
from dataset import grid_subsampling
from utils import load_kernels


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
        batch_len = torch.cat([torch.zeros(1).to(batch_len.device), torch.cumsum(batch_len, dim=0)])

        for i in range(len(batch_len) - 1):
            current_cloud = batch_points[int(batch_len[i]):int(batch_len[i + 1])]
            num_points = len(current_cloud)
            # get neighbors for each point
            dists = square_distance(current_cloud)
            group_idx = torch.arange(num_points, dtype=torch.long).view(1, -1).repeat(num_points, 1)
            group_idx[dists > self.radius ** 2] = num_points
            # create undirected graph
            src, dst = torch.where(group_idx != num_points)
            g = dgl.graph((src, dst))
            g = dgl.to_bidirected(g)
            g = g.to(batch_points.device)
            g.ndata['pos'] = current_cloud
            g.ndata['feat'] = batch_feats[int(batch_len[i]):int(batch_len[i + 1])]
            batch_g.append(g)
        
        return dgl.batch(batch_g)


class BatchGridSubsampling(nn.Module):
    r"""
    Create barycenters from batched points for the next layer by batch grid subsampling
    """
    def __init__(self, dl, offset=5):
        super(BatchGridSubsampling, self).__init__()
        self.dl = dl
        self.offset = offset

    def forward(self, batch_points, batch_feats, batch_len):
        # +offset -> gs simultaneously -> -offset
        device = batch_points.device
        offsets = np.arange(0, len(batch_len) * self.offset, self.offset)
        # each offset will be repeated by the number of each point cloud
        offsets = offsets.repeat(batch_len)
        batch_offsets = torch.FloatTensor(offsets).reshape(-1, 1).to(device)  # [batch, 1]
        batch_offset_points = batch_points + batch_offsets  # [batch, 3]
        
        pool_points, pool_feats = grid_subsampling(batch_offset_points, batch_feats, self.dl)
        # calculate pool batch length
        tmp_points = torch.cat([pool_points, torch.zeros(1, pool_points.size()[1]).to(device)], dim=0)
        # assume that there exists a gap between each point cloud
        gap = torch.abs(tmp_points[1:, :] - tmp_points[:-1, :]) >= self.offset - 1
        pool_cumsum_batch = torch.cat([torch.zeros(1).to(device), torch.where(gap[:, 0] == True)[0] + 1])
        pool_batch = pool_cumsum_batch[1:] - pool_cumsum_batch[:-1]

        return pool_points, pool_feats, pool_batch


class KPConv(nn.Module):
    r"""
    
    """
    def __init__(self,
                 k,
                 p_dim,
                 in_dim,
                 out_dim,
                 KP_extent,
                 radius,
                 fixed_kernel_points='center'):
        super(KPConv, self).__init__()
        
        self.out_dim = out_dim
        self.KP_extent = KP_extent

        # kernel points weight
        self.weights = nn.Parameter(torch.FloatTensor(k, in_dim, out_dim), requires_grad=True)
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        # kernel points position
        kp = load_kernels(radius, k, p_dim, fixed_kernel_points)
        self.kernel_points = nn.Parameter(torch.FloatTensor(kp), requires_grad=False)
        # h in equation (2)
        self.relu = nn.ReLU()

    def msg_fn(self, edge):
        y = edge.src['pos'] - edge.dst['pos']  # centerize every neighborhood
        y = y.unsqueeze(1) - self.kernel_points  # [n_edges, K, p_dim]
        h = self.relu(1 - torch.sqrt(torch.sum(y ** 2, dim=-1)) / self.KP_extent)  # [n_edges, K]
        h = h.unsqueeze(-1).unsqueeze(-1)  # [n_edges, K, 1, 1]
        m = torch.sum(h * self.weights, dim=1)  # [n_edges, K, in_dim, out_dim] -> [n_edges, in_dim, out_dim]
        f = edge.src['feat'].unsqueeze(1)  # [n_edges, 1, in_dim]
        return {'m': (f @ m).squeeze(1)}

    def forward(self, g, feats):
        with g.local_scope():
            g.ndata['feat'] = feats
            g.update_all(self.msg_fn, fn.sum('m', 'h'))
            return g.ndata['h']
