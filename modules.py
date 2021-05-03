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
            g = dgl.to_bidirected(g, copy_ndata=True)
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
        print(edge.data['y'].size(), self.kernel_points.size())
        y = edge.data['y'].unsqueeze(1) - self.kernel_points  # [n_edges, K, p_dim]
        h = self.relu(1 - torch.sqrt(torch.sum(y ** 2, dim=-1)) / self.KP_extent)  # [n_edges, K]
        h = h.unsqueeze(-1).unsqueeze(-1)  # [n_edges, K, 1, 1]
        m = torch.sum(h * self.weights, dim=1)  # [n_edges, K, in_dim, out_dim] -> [n_edges, in_dim, out_dim]
        f = edge.src['feat'].unsqueeze(1)  # [n_edges, 1, in_dim]
        return {'m': (f @ m).squeeze(1)}

    def forward(self, g):
        # Center every neighborhood
        g.apply_edges(fn.u_sub_v('pos', 'pos', 'y'))
        g.update_all(self.msg_fn, fn.sum('m', 'h'))
        return g
