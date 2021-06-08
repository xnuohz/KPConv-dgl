import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import time
from logzero import logger
from torch.utils.data import DataLoader
from dataset import ModelNet40Dataset, ModelNet40Collate
from model import KPCNN


def train(model, device, data_loader, opt, loss_fn, i):
    model.train()

    train_loss = []
    for gs, feats, labels in data_loader:
        batch_gs = [g.to(device) for g in gs]
        batch_feats = feats.to(device)
        labels = labels.to(device)
        torch.cuda.empty_cache()
        logits = model(batch_gs, batch_feats)
        loss = loss_fn(logits, labels.view(-1))
        logger.info(f'Epoch {i} | Step Loss: {loss.item():.4f}')
        train_loss.append(loss.item())
        opt.zero_grad()
        torch.cuda.empty_cache()
        loss.backward()
        opt.step()
    
    return sum(train_loss) / len(train_loss)


@torch.no_grad()
def test(model, device, data_loader):
    model.eval()

    y_true, y_pred = [], []
    for gs, feats, labels in data_loader:
        batch_gs = [g.to(device) for g in gs]
        batch_feats = feats.to(device)
        logits = model(batch_gs, batch_feats)
        y_true.append(labels.detach().cpu())
        y_pred.append(logits.argmax(1).view(-1, 1).detach().cpu())

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    
    return (y_true == y_pred).sum().item() / len(y_true)


def main():
    # check cuda
    device = f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu'

    # load dataset
    train_dataset = ModelNet40Dataset(args, 'data/ModelNet40', split='train')
    test_dataset = ModelNet40Dataset(args, 'data/ModelNet40', split='test')
    
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              collate_fn=ModelNet40Collate,
                              shuffle=True)
    
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             collate_fn=ModelNet40Collate,
                             shuffle=False)
    
    # load model
    args.num_classes = train_dataset.num_classes
    model = KPCNN(args).to(device)

    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=args.lr)
    times = []

    logger.info('---------- Training ----------')
    for i in range(args.epochs):
        t1 = time.time()
        train_loss = train(model, device, train_loader, opt, loss_fn, i)
        t2 = time.time()

        if i >= 5:
            times.append(t2 - t1)

        if i % args.interval == 0:
            train_acc = test(model, device, train_loader)
            logger.info(f'Epoch {i} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
        else:
            logger.info(f'Epoch {i} | Train Loss: {train_loss:.4f}')
    
    logger.info('---------- Testing ----------')
    test_acc = test(model, device, test_loader)
    logger.info(f'Test Acc: {test_acc:.4f}')
    if args.epochs >= 5:
        logger.info(f'Times/epoch: {sum(times) / len(times):.4f}')

    model_path = f'models/KPCNN-{args.first_subsampling_dl}-{args.data_type}-arch{len(args.architecture)}.pt'
    torch.save(model.state_dict(), model_path)


if __name__ == '__main__':
    """
    KPConv Hyperparameters
    """
    parser = argparse.ArgumentParser(description='KPConv')
    parser.add_argument('--data-type', type=str, default='small', choices=['small', 'large'])
    parser.add_argument('--redo', action='store_true')
    parser.add_argument('--in-features-dim', type=int, default=3)
    parser.add_argument('--first-features-dim', type=int, default=32)
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
                                                              'resnetb',
                                                              'resnetb_strided',
                                                              'resnetb',
                                                              'resnetb',
                                                              'resnetb_strided',
                                                              'resnetb',
                                                              'resnetb',
                                                              'resnetb_strided',
                                                              'resnetb',
                                                              'resnetb',
                                                              'global_average'])
    # cuda
    parser.add_argument('--gpu', type=int, default=0)
    # training
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch-size', type=int, default=80)
    parser.add_argument('--interval', type=int, default=40)
    
    args = parser.parse_args()
    logger.info(args)

    main()
