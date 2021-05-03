from torch.utils.data import DataLoader
from dataset import ModelNet40Dataset, collate_fn
from modules import FixedRadiusNNGraph, KPConv


def main():
    dataset = ModelNet40Dataset('data/ModelNet40', 0.02)
    
    train_data = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

    fnn = FixedRadiusNNGraph(0.05)
    kpconv = KPConv(15, 3, 3, 64, 1, 0.05)

    for p, feat, labels, length in train_data:
        # print(p)
        # print(feat)
        # print(labels)
        batch_g = fnn(p, feat, length)
        print(batch_g)
        batch_g = kpconv(batch_g)
        print(batch_g)
        break


if __name__ == '__main__':
    main()
