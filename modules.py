import math
import torch
import torch.nn as nn
import dgl.function as fn
from dgl.nn.pytorch.glob import AvgPooling
from utils import load_kernels


class KPConv(nn.Module):
    r"""

    Description
    -----------
    Kernel point convolution

    Parameters
    -----------
    k: int
        Number of kernel points.
    p_dim: int
        Dimension of the point space.
    in_dim: int
        Dimension of input features.
    out_dim: int
        Dimension of output features.
    KP_extent: float
        Influence radius of each kernel point.(sigma in equation 3)
    radius: float
        Radius used for kernel point init.
    fixed_kernel_points: str
        Fix position of certain kernel points ('none', 'center' or 'verticals').
    """
    def __init__(self, k, p_dim, in_dim, out_dim, KP_extent, radius, fixed_kernel_points='center'):
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
            g.srcdata['feat'] = feats
            g.update_all(self.msg_fn, fn.sum('m', 'h'))
            return g.dstdata['h']


def block_decider(block_name, radius, in_dim, out_dim, config):
    if block_name == 'simple':
        return SimpleBlock(in_dim, out_dim, radius, config)
    elif block_name in ['resnetb', 'resnetb_strided']:
        return ResnetBottleneckBlock(block_name, in_dim, out_dim, radius, config)
    elif block_name == 'global_average':
        return GlobalAverageBlock()
    else:
        raise ValueError(f'Unknown block name {block_name}')


class SimpleBlock(nn.Module):
    r"""

    Description
    -----------
    Initialize a simple convolution block with its ReLU and BatchNorm.

    Parameters
    -----------
    in_dim: int
        Dimension of input features.
    out_dim: int
        Dimension of output features.
    radius: float
        Radius used for kernel point init.
    config:
        Model parameters.
    """
    def __init__(self, in_dim, out_dim, radius, config):
        super(SimpleBlock, self).__init__()
        
        # get KP_extent from current radius
        current_extent = radius * config.KP_extent / config.conv_radius

        self.kpconv = KPConv(k=config.num_kernel_points,
                             p_dim=config.p_dim,
                             in_dim=in_dim,
                             out_dim=out_dim,
                             KP_extent=current_extent,
                             radius=radius)
        
        # Other opperations
        self.bn = nn.BatchNorm1d(out_dim, momentum=config.bn_momentum)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, g, feats):
        with g.local_scope():
            feats = self.kpconv(g, feats)
            feats = self.leaky_relu(self.bn(feats))
            return feats


class ResnetBottleneckBlock(nn.Module):
    r"""

    Description
    -----------
    Initialize a resnet bottleneck block.

    Parameters
    -----------
    block_name: str
        'resnetb' or 'resnetb_strided'.
    in_dim: int
        Dimension of input features.
    out_dim: int
        Dimension of output features.
    radius: float
        Radius used for kernel point init.
    config:
        Model parameters.
    """
    def __init__(self, block_name, in_dim, out_dim, radius, config):
        super(ResnetBottleneckBlock, self).__init__()
        
        self.block_name = block_name
        # get KP_extent from current radius
        current_extent = radius * config.KP_extent / config.conv_radius

        if in_dim != out_dim // 4:
            self.down_scaling = nn.Sequential(
                nn.Linear(in_dim, out_dim // 4, bias=False),
                nn.BatchNorm1d(out_dim // 4, momentum=config.bn_momentum),
                nn.LeakyReLU(0.1)
            )
        else:
            self.down_scaling = nn.Identity()
        
        self.kpconv = KPConv(k=config.num_kernel_points,
                             p_dim=config.p_dim,
                             in_dim=out_dim // 4,
                             out_dim=out_dim // 4,
                             KP_extent=current_extent,
                             radius=radius)
        
        self.bn = nn.BatchNorm1d(out_dim // 4, momentum=config.bn_momentum)
        self.up_scaling = nn.Sequential(
            nn.Linear(out_dim // 4, out_dim, bias=False),
            nn.BatchNorm1d(out_dim, momentum=config.bn_momentum),
            nn.LeakyReLU(0.1)
        )

        if in_dim != out_dim:
            self.res = nn.Sequential(
                nn.Linear(in_dim, out_dim, bias=False),
                nn.BatchNorm1d(out_dim, momentum=config.bn_momentum)
            )
        else:
            self.res = nn.Identity()
        
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, g, feats):
        # conv_g -> homograph, pool_g -> heterograph
        with g.local_scope():
            if 'strided' in self.block_name:
                g.srcdata['feat'] = feats
                g.multi_update_all({
                    'to': (fn.copy_u('feat', 'm'), fn.mean('m', 'feat'))
                }, 'sum')
                shortcut = g.dstdata['feat']
            else:
                shortcut = feats

            x = self.down_scaling(feats)
            x = self.kpconv(g, x)
            x = self.leaky_relu(self.bn(x))
            x = self.up_scaling(x)

            return self.leaky_relu(x + self.res(shortcut))


class GlobalAverageBlock(nn.Module):
    r"""

    Description
    -----------
    Initialize a global average block.
    """
    def __init__(self):
        super(GlobalAverageBlock, self).__init__()
        self.pool = AvgPooling()

    def forward(self, g, feats):
        with g.local_scope():
            feats = self.pool(g, feats)
            return feats
