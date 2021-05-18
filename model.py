import torch.nn as nn
from modules import block_decider


class KPCNN(nn.Module):
    def __init__(self, config):
        super(KPCNN, self).__init__()

        #####################
        # Network opperations
        #####################

        # Current radius of convolution and feature dimension
        r = config.first_subsampling_dl * config.conv_radius
        in_dim = config.in_features_dim
        out_dim = config.first_features_dim
        self.K = config.num_kernel_points

        # Save all block operations in a list of modules
        self.archs = config.architecture
        self.block_ops = nn.ModuleList()

        # Loop over consecutive blocks
        # only simple, resnetb, resnetb_strided, global_average are valid in our example
        for block_i, block in enumerate(self.archs):
            # Apply the good block function defining torch ops
            self.block_ops.append(block_decider(block,
                                                r,
                                                in_dim,
                                                out_dim,
                                                config))

            # Update dimension of input from output
            in_dim = out_dim

            # Detect change to a subsampled layer
            if 'strided' in block:
                # Update radius and feature dimension for next layer
                r *= 2
                out_dim *= 2
        
        self.output = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.Linear(out_dim, config.num_classes)
        )

    def forward(self, batch_gs, batch_feats):
        layer = 0
        x = batch_feats
        for block_i, block in enumerate(self.archs):
            if 'strided' in block:
                x = self.block_ops[block_i](batch_gs[layer + 1], x)
                layer += 2
            else:
                x = self.block_ops[block_i](batch_gs[layer], x)
        
        return self.output(x)
