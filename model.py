import torch.nn as nn
from modules import block_decider


class KPCNN(nn.Module):
    def __init__(self, config):
        super(KPCNN, self).__init__()

        #####################
        # Network opperations
        #####################

        # Current radius of convolution and feature dimension
        layer = 0
        r = config.first_subsampling_dl * config.conv_radius
        in_dim = config.in_features_dim
        out_dim = config.first_features_dim
        self.K = config.num_kernel_points

        # Save all block operations in a list of modules
        self.block_ops = nn.ModuleList()

        # Loop over consecutive blocks
        # only simple, resnetb, resnetb_strided, global_average are valid in our example
        block_in_layer = 0
        for block_i, block in enumerate(config.architecture):
            # Apply the good block function defining torch ops
            self.block_ops.append(block_decider(block,
                                                r,
                                                in_dim,
                                                out_dim,
                                                config))


            # Index of block in this layer
            block_in_layer += 1

            # Update dimension of input from output
            if 'simple' in block:
                in_dim = out_dim // 2
            else:
                in_dim = out_dim

            # Detect change to a subsampled layer
            if 'strided' in block:
                # Update radius and feature dimension for next layer
                layer += 1
                r *= 2
                out_dim *= 2
                block_in_layer = 0

        self.output = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.Linear(1024, config.num_classes)
        )

    def forward(self, batch_conv_gs, batch_pool_gs, batch_feats):
        pass
