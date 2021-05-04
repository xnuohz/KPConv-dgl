import argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ModelNet40Dataset, collate_fn
from model import KPCNN


def main():
    dataset = ModelNet40Dataset('data/ModelNet40', 0.02)
    
    train_data = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

    model = KPCNN(args)

    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=args.lr)

    for points, feats, labels, length in train_data:
        model.train()
        opt.zero_grad()
        logits = model(points, feats, length)
        train_loss = loss_fn(logits, labels.view(-1))
        train_loss.backward()
        opt.step()
        break


if __name__ == '__main__':
    """
    KPConv Hyperparameters
    """
    parser = argparse.ArgumentParser(description='KPConv')

    # Number of kernel points
    parser.add_argument('--num-kernel-points', type=int, default=15)
    # Radius of the area of influence of each kernel point in "number grid cell". (1.0 is the standard value)
    parser.add_argument('--KP-extent', type=float, default=1.2)
    # Radius of convolution in "number grid cell". (2.5 is the standard value)
    parser.add_argument('--conv-radius', type=float, default=2.5)
    # Dimension of input points
    parser.add_argument('--p-dim', type=int, default=3)
    # Batch normalization parameters
    parser.add_argument('--bn-momentum', type=float, default=0.05)
    # learning rate
    parser.add_argument('--lr', type=float, default=0.01)

    args = parser.parse_args()
    print(args)

    main()
