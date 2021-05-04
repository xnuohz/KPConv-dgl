import torch.nn as nn
from dgl.nn.pytorch.glob import AvgPooling
from modules import FixedRadiusNNGraph, BatchGridSubsampling, KPConv


class SimpleBlock(nn.Module):
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
        
        self.bn = nn.BatchNorm1d(out_dim, momentum=config.bn_momentum)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, g, feats):
        with g.local_scope():
            feats = self.kpconv(g, feats)
            feats = self.leaky_relu(self.bn(feats))
            return feats


class ResnetBlock(nn.Module):
    def __init__(self, in_dim, out_dim, radius, config):
        super(ResnetBlock, self).__init__()
        
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
        with g.local_scope():
            x = self.down_scaling(feats)
            x = self.kpconv(g, x)
            x = self.leaky_relu(self.bn(x))
            x = self.up_scaling(x)
            return self.leaky_relu(x + self.res(feats))


class KPCNN(nn.Module):
    r"""
    
    """
    def __init__(self, config):
        super(KPCNN, self).__init__()
        
        self.fnn = FixedRadiusNNGraph(0.02)
        self.bgs = BatchGridSubsampling(0.2)
        self.block1 = SimpleBlock(3, 64, 0.02, config)
        self.block2 = ResnetBlock(64, 128, 0.04, config)
        self.block3 = ResnetBlock(128, 512, 0.08, config)
        self.block4 = ResnetBlock(512, 1024, 0.16, config)
        self.pool = AvgPooling()
        self.output = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.Linear(1024, 40)
        )

    def forward(self, points, feats, length):
        batch_g = self.fnn(points, feats, length)
        feats = self.block1(batch_g, batch_g.ndata['feat'])
        feats = self.block2(batch_g, feats)
        points1, feats1, length1 = self.bgs(batch_g.ndata['pos'], feats, length)
        
        batch_g1 = self.fnn(points1, feats1, length1)
        feats1 = self.block3(batch_g1, feats1)
        feats1 = self.block4(batch_g1, feats1)
        
        g_feat = self.pool(batch_g1, feats1)
        return self.output(g_feat)
