import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ModelNet40Dataset, ModelNet40Collate
from model import KPCNN


def main():
    # check cuda
    device = f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu'

    # load dataset
    train_dataset = ModelNet40Dataset(args, 'data/ModelNet40', split='train')
    test_dataset = ModelNet40Dataset(args, 'data/ModelNet40', split='test')

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,                
                              collate_fn=ModelNet40Collate,
                              shuffle=False)
    
    test_loader = DataLoader(test_dataset,
                              batch_size=args.batch_size,                
                              collate_fn=ModelNet40Collate,
                              shuffle=False)
    
    # load model
    args.num_classes = train_dataset.num_classes
    model = KPCNN(args).to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=args.lr)

    for gs, feats, labels in train_loader:
        batch_gs = [g.to(device) for g in gs]
        batch_feats = feats.to(device)
        labels = labels.to(device)
        logits = model(batch_gs, batch_feats)
        print(logits.size())
        break


if __name__ == '__main__':
    """
    KPConv Hyperparameters
    """
    parser = argparse.ArgumentParser(description='KPConv')
    parser.add_argument('--in-features-dim', type=int, default=3)
    parser.add_argument('--first-features-dim', type=int, default=64)
    parser.add_argument('--first-subsampling-dl', type=float, default=0.02)
    parser.add_argument('--num-kernel-points', type=int, default=15)
    parser.add_argument('--KP-extent', type=float, default=1.2)
    parser.add_argument('--conv-radius', type=float, default=2.5)
    parser.add_argument('--p-dim', type=int, default=3)
    parser.add_argument('--bn-momentum', type=float, default=0.05)
    parser.add_argument('--architecture', type=list, default=['simple',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'global_average'])
    # cuda
    parser.add_argument('--gpu', type=int, default=0)
    # training
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch-size', type=int, default=1)
    
    args = parser.parse_args()
    print(args)

    main()
