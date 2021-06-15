import os
import numpy as np
import torch
import dgl
import dgl.function as fn
from logzero import logger
from collections import defaultdict
from torch.utils.data import Dataset
from utils import grid_subsampling, batch_neighbors, batch_grid_subsampling


class ModelNet40Dataset(Dataset):
    def __init__(self, args, root, split='train'):
        assert split in ['train', 'test']

        self.config = args
        self.root = root
        self.split = split
        self.type = 10 if args.data_type == 'small' else 40
        catfile = os.path.join(root, f'modelnet{self.type}_shape_names.txt')
        cat = [l.rstrip() for l in open(catfile)]
        self.label_to_names = {k: v for k, v in enumerate(cat)}
        self.name_to_label = {v: k for k, v in self.label_to_names.items()}
        # load point cloud
        points, self.feats, lengths, self.labels = self.load_subsampled_clouds()
        lengths = torch.cumsum(torch.cat([torch.LongTensor([0]), lengths]), dim=0)
        
        self.points, self.stacked_lengths, self.conv_gs, self.pool_gs = self.classification_inputs(points, lengths)

    @property
    def num_classes(self):
        return self.labels.max().item() + 1

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feats = self.feats[self.stacked_lengths[0][idx]:self.stacked_lengths[0][idx + 1], :]
        label = self.labels[idx]
        gs = []
        
        for i, lengths in enumerate(self.stacked_lengths):
            gs.append(self.conv_gs[i][idx])
            if i != len(self.stacked_lengths) - 1:
                pool_g = self.pool_gs[i][idx]
                if 'pos' not in pool_g.srcdata or 'pos' not in pool_g.dstdata:
                    return self.__getitem__(0)
                gs.append(pool_g)

        return gs, feats, label
    
    def load_subsampled_clouds(self):
        logger.info(f'Loading {self.split} points subsampled at {self.config.first_subsampling_dl:.3f}')
        filename = f'{self.root}/{self.split}_{self.config.first_subsampling_dl}_record_{self.type}.pkl'
        
        if os.path.exists(filename):
            return torch.load(open(filename, 'rb'))
        
        names = np.loadtxt(f'{self.root}/modelnet{self.type}_{self.split}.txt', dtype=str)
        
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

    def classification_inputs(self, stacked_points, stacked_lengths):
        filename = f'{self.root}/{self.split}_{self.config.first_subsampling_dl}_classification_{self.type}.pkl'
        
        if not self.config.redo and os.path.exists(filename):
            return torch.load(open(filename, 'rb'))

        logger.info(f'Preprocessing {self.split} points subsampled in classification format')
        # Starting radius of convolutions
        r_normal = self.config.first_subsampling_dl * self.config.conv_radius

        # Starting layer
        layer_blocks = []

        # Lists of inputs
        input_points = []
        # input_conv_gs: [dgl.DGLGraph], [[g1_layer1, ..., gn_layer1], [g1_layer2, ..., gn_layer2], ...]
        input_conv_gs = []
        # input_pool_gs: [dgl.DGLGraph], [[g1_layer1, ..., gn_layer1], [g1_layer2, ..., gn_layer2], ...]
        input_pool_gs = []
        input_stack_lengths = []

        ######################
        # Loop over the blocks
        ######################

        # simple -> resnetb -> resnetb_strided -> resnetb -> resnetb_strided -> resnetb -> global_average
        # conv_g            -> pool_g          -> conv_g  -> pool_g          -> conv_g
        arch = self.config.architecture

        for block_i, block in enumerate(arch):
            # Get all blocks of the layer
            if not ('strided' in block or 'global' in block):
                layer_blocks += [block]
                continue

            # Convolution neighbors indices
            # *****************************
            # layer_blocks must not be []
            # Convolutions are done in this layer, compute the neighbors with the good radius
            conv_gs = batch_neighbors(stacked_points, stacked_points, stacked_lengths, stacked_lengths, r_normal)

            # Updating input lists
            input_points.append(stacked_points)
            input_stack_lengths.append(stacked_lengths)
            input_conv_gs.append(conv_gs)

            # Stop when meeting a global pooling
            if 'global' in block:
                break

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
            pool_gs = batch_neighbors(pool_p, stacked_points, pool_b, stacked_lengths, r_normal)

            # Updating input lists
            input_pool_gs.append(pool_gs)

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
                    input_stack_lengths,
                    input_conv_gs,
                    input_pool_gs), filename)

        return input_points, input_stack_lengths, input_conv_gs, input_pool_gs


def ModelNet40Collate(batch_data):
    _, feats, labels = map(list, zip(*batch_data))
    batch_feats = torch.cat(feats)
    batch_labels = torch.LongTensor(labels).view(-1, 1)

    batch_g_dict = defaultdict(list)

    for gs, _, _ in batch_data:
        for i, g in enumerate(gs):
            batch_g_dict[i].append(g)
    
    batch_gs = []
    
    for _, v in batch_g_dict.items():
        batch_gs.append(dgl.batch(v))
    
    return batch_gs, batch_feats, batch_labels
    