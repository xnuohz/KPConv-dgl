import os
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import grid_subsampling, batch_neighbors, batch_grid_subsampling


class ModelNet40Dataset(Dataset):
    def __init__(self, args, root, split='train'):
        assert split in ['train', 'test']

        self.config = args
        self.root = root
        self.split = split
        catfile = os.path.join(root, 'modelnet10_shape_names.txt')
        cat = [l.rstrip() for l in open(catfile)]
        self.label_to_names = {k: v for k, v in enumerate(cat)}
        self.name_to_label = {v: k for k, v in self.label_to_names.items()}
        # load point cloud
        self.points, self.feats, self.lengths, self.labels = self.load_subsampled_clouds()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return 0
    
    def load_subsampled_clouds(self):
        print(f'Loading {self.split} points subsampled at {self.config.first_subsampling_dl:.3f}')
        filename = f'{self.root}/{self.split}_{self.config.first_subsampling_dl}_record.pkl'
        
        if os.path.exists(filename):
            return torch.load(open(filename, 'rb'))
        
        names = np.loadtxt(f'{self.root}/modelnet10_{self.split}.txt', dtype=str)
        
        # Initialize containers
        input_points = []
        input_feats = []

        # Advanced display
        N = len(names)
        progress_n = 30
        fmt_str = '[{:<' + str(progress_n) + '}] {:5.1f}%'

        # Collect point clouds
        for i, cloud_name in enumerate(names):

            # Read points
            class_folder = '_'.join(cloud_name.split('_')[:-1])
            txt_file = os.path.join(self.root, class_folder, cloud_name) + '.txt'
            data = np.loadtxt(txt_file, delimiter=',', dtype=np.float32)
            data = torch.FloatTensor(data)

            # Subsample them
            # Assume self.config.first_subsampling_dl > 0
            points, feats = grid_subsampling(data[:, :3],
                                                self.config.first_subsampling_dl,
                                                data[:, 3:])

            print('', end='\r')
            print(fmt_str.format('#' * ((i * progress_n) // N), 100 * i / N), end='', flush=True)

            # Add to list
            input_points += [points]
            input_feats += [feats]

            if i == 5:
                break

        print('', end='\r')
        print(fmt_str.format('#' * progress_n, 100), end='', flush=True)
        print()

        # Get labels
        label_names = ['_'.join(name.split('_')[:-1]) for name in names]
        input_labels = torch.LongTensor([self.name_to_label[name] for name in label_names])

        # convert to torch.Tensor
        lengths = torch.LongTensor([p.shape[0] for p in input_points])
        input_points = torch.cat(input_points, dim=0)
        input_feats = torch.cat(input_feats, dim=0)
        
        # Save for later use
        torch.save((input_points,
                    input_feats,
                    lengths,
                    input_labels), filename)

        return input_points, input_feats, lengths, input_labels

    def classification_inputs(self,
                              stacked_points,
                              stacked_features,
                              labels,
                              stacked_lengths):
        
        # Starting radius of convolutions
        r_normal = self.config.first_subsampling_dl * self.config.conv_radius

        # Starting layer
        layer_blocks = []

        # Lists of inputs
        input_points = []
        input_neighbors = []
        input_pools = []
        input_stack_lengths = []

        ######################
        # Loop over the blocks
        ######################

        # simple -> resnetb -> resnetb_strided -> resnetb -> resnetb_strided -> resnetb -> global_average
        arch = self.config.architecture

        for block_i, block in enumerate(arch):

            # Get all blocks of the layer
            if not ('strided' in block or 'global' in block):
                layer_blocks += [block]
                continue

            # Stop when meeting a global pooling
            if 'global' in block:
                break

            # Convolution neighbors indices
            # *****************************

            if layer_blocks:
                # Convolutions are done in this layer, compute the neighbors with the good radius
                conv_src, conv_dst = batch_neighbors(stacked_points,
                                                     stacked_points,
                                                     stack_lengths,
                                                     stacked_lengths,
                                                     r_normal)
            else:
                # This layer only perform pooling, no neighbors required
                conv_src = np.zeros((0, 1), dtype=np.int32)
                conv_dst = np.zeros((0, 1), dtype=np.int32)

            # Pooling neighbors indices
            # *************************

            # If end of layer is a pooling operation
            if 'strided' in block:

                # New subsampling length
                dl = 2 * r_normal / self.config.conv_radius

                # Subsampled points
                pool_p, pool_b = batch_grid_subsampling(stacked_points, stack_lengths, dl)

                # Subsample indices
                pool_src, pool_dst = batch_neighbors(pool_p,
                                                     stacked_points,
                                                     pool_b,
                                                     stack_lengths,
                                                     r_normal)

            else:
                # No pooling in the end of this layer, no pooling indices required
                pool_i = np.zeros((0, 1), dtype=np.int32)
                pool_p = np.zeros((0, 1), dtype=np.float32)
                pool_b = np.zeros((0,), dtype=np.int32)

            # Updating input lists
            input_points += [stacked_points]
            input_neighbors += [(conv_src, conv_dst)]
            input_pools += [(pool_src, pool_dst)]
            input_stack_lengths += [stack_lengths]

            # New points for next layer
            stacked_points = pool_p
            stack_lengths = pool_b

            # Update radius and reset blocks
            r_normal *= 2
            layer_blocks = []

        ###############
        # Return inputs
        ###############

        # list of network inputs
        li = input_points + input_neighbors + input_pools + input_stack_lengths
        li += [stacked_features, labels]

        return li


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='KPConv')
    parser.add_argument('--first_subsampling_dl', type=float, default=0.02)
    args = parser.parse_args()

    dataset = ModelNet40Dataset(args, 'data/ModelNet40')
    print(len(dataset))
