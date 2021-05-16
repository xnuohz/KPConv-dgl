import os
import numpy as np
import torch
import dgl
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
        points, self.feats, lengths, labels = self.load_subsampled_clouds()
        lengths = torch.cumsum(torch.cat([torch.LongTensor([0]), lengths]), dim=0)

        # for debug
        points = points[:lengths[3], :]
        labels = points[:lengths[3], :]
        lengths = lengths[:4]

        self.points, self.neighbors_src, self.neighbors_dst, self.pools_src, self.pools_dst, self.stacked_lengths = self.classification_inputs(points, labels, lengths)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feats = self.feats[self.stacked_lengths[0][idx]:self.stacked_lengths[0][idx + 1], :]
        conv_gs, pool_gs = [], []
        
        for i, lengths in enumerate(self.stacked_lengths):
            cg = dgl.graph((self.neighbors_src[i][idx], self.neighbors_dst[i][idx]))
            cg.ndata['pos'] = self.points[i][lengths[idx]:lengths[idx + 1], :]
            conv_gs.append(cg)
            # This is not right, for the range of src node is larger than the range of dst
            # They are not in the same domain
            pg = dgl.graph((self.pools_src[i][idx], self.pools_dst[i][idx]))
            pool_gs.append(pg)

        return conv_gs, pool_gs, feats
    
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
                              labels,
                              stacked_lengths):
        
        print(f'Preprocessing {self.split} points subsampled in classification format')
        filename = f'{self.root}/{self.split}_{self.config.first_subsampling_dl}_classification.pkl'
        
        if os.path.exists(filename):
            return torch.load(open(filename, 'rb'))

        # Starting radius of convolutions
        r_normal = self.config.first_subsampling_dl * self.config.conv_radius

        # Starting layer
        layer_blocks = []

        # Lists of inputs
        input_points = []
        input_neighbors_src = []
        input_neighbors_dst = []
        input_pools_src = []
        input_pools_dst = []
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
            # layer_blocks must not be []
            # Convolutions are done in this layer, compute the neighbors with the good radius
            # conv_src: [torch.Tensor], [[g1_src1, g1_src2, ...], [g2_src1, g2_src2, ...], ...]
            # conv_dst: [torch.Tensor], [[g1_dst1, g1_dst2, ...], [g2_dst1, g2_dst2, ...], ...]
            conv_src, conv_dst = batch_neighbors(stacked_points,
                                                 stacked_points,
                                                 stacked_lengths,
                                                 stacked_lengths,
                                                 r_normal)
            
            # Pooling neighbors indices
            # *************************

            # If come to here, block must be resnetb_strided in our example
            # If end of layer is a pooling operation

            # New subsampling length
            dl = 2 * r_normal / self.config.conv_radius

            # Subsampled points
            pool_p, pool_b = batch_grid_subsampling(stacked_points, stacked_lengths, dl)
            
            # Subsample indices
            # pool_src: [torch.Tensor], [[g1_src1, g1_src2, ...], [g2_src1, g2_src2, ...], ...]
            # pool_dst: [torch.Tensor], [[pool_g1_dst1, pool_g1_dst2, ...], [pool_g2_dst1, pool_g2_dst2, ...], ...]
            pool_src, pool_dst = batch_neighbors(pool_p,
                                                 stacked_points,
                                                 pool_b,
                                                 stacked_lengths,
                                                 r_normal)

            # Updating input lists
            input_points.append(stacked_points)
            input_neighbors_src.append(conv_src)
            input_neighbors_dst.append(conv_dst)
            input_pools_src.append(pool_src)
            input_pools_dst.append(pool_dst)
            input_stack_lengths.append(stacked_lengths)

            # New points for next layer
            stacked_points = pool_p
            stacked_lengths = pool_b

            # Update radius and reset blocks
            r_normal *= 2
            layer_blocks = []

        ###############
        # Return inputs
        ###############

        # Save for later use
        torch.save((input_points,
                    input_neighbors_src,
                    input_neighbors_dst,
                    input_pools_src,
                    input_pools_dst,
                    input_stack_lengths), filename)

        return input_points, input_neighbors_src, input_neighbors_dst, \
            input_pools_src, input_pools_dst, input_stack_lengths


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='KPConv')
    parser.add_argument('--first_subsampling_dl', type=float, default=0.02)
    parser.add_argument('--conv-radius', type=float, default=2.5)
    parser.add_argument('--architecture', type=list, default=['simple',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb',
                    'global_average'])
    args = parser.parse_args()
    print(args)

    dataset = ModelNet40Dataset(args, 'data/ModelNet40')
    print(dataset[0])
