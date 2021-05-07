import os
import numpy as np
import torch
import dgl
from collections import defaultdict
from torch.utils.data import Dataset


def get_bbox(points):
    r"""
    
    Description
    -----------
    Get the bounding box of a point cloud
    
    Parameters
    ----------
    points: torch.Tensor
        [N, 3] matrix of input points
    """
    min_point = torch.min(points, 0)[0].data
    max_point = torch.max(points, 0)[0].data
    
    return min_point, max_point


def grid_subsampling(points, feats, dl):
    r"""

    Description
    -----------
    Grid subsampling is implemented by C++ in the author's PyTorch code
    
    Parameters
    ----------
    points: torch.Tensor
        [N, 3] matrix of input points
    feats: torch.Tensor
        [N, D] matrix of input features
    dl: float
        the size of grid voxels

    Return
    ----------
    subsampled points and features
    """
    n_points = len(points)
    min_corner, max_corner = get_bbox(points)
    # is this nessesary?
    min_corner = torch.floor(torch.div(min_corner, dl)) * dl
    
    sample_nx = int((max_corner[0] - min_corner[0]) / dl) + 1
    sample_ny = int((max_corner[1] - min_corner[1]) / dl) + 1

    data = defaultdict(list)

    for p, f in zip(points, feats):
        idx_x = int((p[0] - min_corner[0]) / dl)
        idx_y = int((p[1] - min_corner[1]) / dl)
        idx_z = int((p[2] - min_corner[2]) / dl)
        idx = idx_x + sample_nx * idx_y + sample_nx * sample_ny * idx_z
        data[idx].append(torch.cat([p, f]).view(1, -1))
    
    subsampled_data = []
    
    for _, v in data.items():
        v = torch.cat(v)
        subsampled_data.append(torch.mean(v, dim=0).view(1, -1))

    subsampled_data = torch.cat(subsampled_data)

    return subsampled_data[:, :3], subsampled_data[:, 3:]


def collate_fn(batch):
    r"""
    points: list, [b, n, 3]
    feats: list, [b, n, d]
    labels: list, [b,]
    len: list, [b,], each value is the number of points in each point cloud
    """
    points, feats, labels, length = map(list, zip(*batch))
    batch_points = torch.FloatTensor(np.concatenate(points))
    batch_feats = torch.FloatTensor(np.concatenate(feats))
    batch_labels = torch.LongTensor(labels).view(-1, 1)
    batch_len = np.concatenate([[0], np.cumsum(length)])
    return batch_points, batch_feats, batch_labels, batch_len


class ModelNet40Dataset(Dataset):
    def __init__(self,
                 root,
                 dl,
                 split='train',
                 orient_correction=True,
                 cache_size=10000):
        assert split in ['train', 'test']
        
        self.root = root
        self.dl = dl
        self.orient_correction = orient_correction
        catfile = os.path.join(root, 'modelnet40_shape_names.txt')
        cat = [l.rstrip() for l in open(catfile)]
        self.label_to_names = {k: v for k, v in enumerate(cat)}
        self.name_to_label = {v: k for k, v in self.label_to_names.items()}
        
        # load point cloud
        self.names = np.loadtxt(os.path.join(root, f'modelnet40_{split}.txt'), dtype=str)
        self.cache_size = cache_size
        self.cache = {}

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        else:
            cloud_name = self.names[idx]
            class_folder = '_'.join(cloud_name.split('_')[:-1])
            txt_file = f'{self.root}/{class_folder}/{cloud_name}.txt'
            # point cloud
            data = np.loadtxt(txt_file, delimiter=',', dtype=np.float32)
            data = torch.FloatTensor(data)
            points, feats = grid_subsampling(points=data[:, :3],
                                             feats=data[:, 3:],
                                             dl=self.dl)

            if self.orient_correction:
                points = points[:, [0, 2, 1]]
                feats = feats[:, [0, 2, 1]]
            # ground truth
            label = self.name_to_label[class_folder]

            if len(self.cache) < self.cache_size:
                self.cache[idx] = (points, feats, label, len(points))

            return points, feats, label, len(points)
