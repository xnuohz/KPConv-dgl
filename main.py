import argparse
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ModelNet40Dataset, collate_fn
from model import KPCNN


def train(model, device, data_loader, opt, loss_fn):
    model.train()

    train_loss = []
    for points, feats, labels, length in data_loader:
        points = points.to(device)
        feats = feats.to(device)
        labels = labels.to(device)

        logits = model(points, feats, length)
        loss = loss_fn(logits, labels.view(-1))
        train_loss.append(loss.item())
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        break
    
    return sum(train_loss) / len(train_loss)


@torch.no_grad()
def test(model, device, data_loader):
    model.eval()

    y_true, y_pred = [], []
    for points, feats, labels, length in data_loader:
        points = points.to(device)
        feats = feats.to(device)
        logits = model(points, feats, length)
        y_true.append(labels.detach().cpu())
        y_pred.append(logits.argmax(1).view(-1, 1).detach().cpu())
        break
    
    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    
    return (y_true == y_pred).sum().item() / len(y_true)


def main():
    # check cuda
    device = f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu'

    # load dataset
    train_dataset = ModelNet40Dataset('data/ModelNet40', dl=args.dl, split='test')
    test_dataset = ModelNet40Dataset('data/ModelNet40', dl=args.dl, split='test')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)

    # load model
    model = KPCNN(args).to(device)

    print(model)

    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=args.lr)

    print('---------- Training ----------')
    for i in range(args.epochs):
        train_loss = train(model, device, train_loader, opt, loss_fn)
        train_acc = test(model, device, train_loader)
        print(f'Epoch {i} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
    
    print('---------- Testing ----------')
    test_acc = test(model, device, test_loader)
    print(f'Test Acc: {test_acc:.4f}')


if __name__ == '__main__':
    """
    KPConv Hyperparameters
    """
    parser = argparse.ArgumentParser(description='KPConv')

    # number of kernel points
    parser.add_argument('--num-kernel-points', type=int, default=15)
    # radius of the area of influence of each kernel point in "number grid cell". (1.0 is the standard value)
    parser.add_argument('--KP-extent', type=float, default=1.2)
    # radius of convolution in "number grid cell". (2.5 is the standard value)
    parser.add_argument('--conv-radius', type=float, default=2.5)
    # dimension of input points
    parser.add_argument('--p-dim', type=int, default=3)
    # batch normalization parameters
    parser.add_argument('--bn-momentum', type=float, default=0.05)
    # size of the first subsampling grid in meter
    parser.add_argument('--dl', type=float, default=0.02)
    # cuda
    parser.add_argument('--gpu', type=int, default=0)
    # model
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch-size', type=int, default=4)

    args = parser.parse_args()
    print(args)

    main()
