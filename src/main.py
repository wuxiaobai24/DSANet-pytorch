from model import DSANet
import torch
from data import TSDataset
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import os


def init_parser():
    parser = ArgumentParser()
    parser.add_argument('dataset', type=str,
                        help='dataset name, for example electricity')
    parser.add_argument('--dataset_path', type=str, default='../dataset')
    parser.add_argument('--windows', type=int, default=32)
    parser.add_argument('--horizon', type=int, default=6)

    parser.add_argument('--n_global', type=int, default=12)
    parser.add_argument('--n_global_head', type=int, default=3)
    parser.add_argument('--n_global_hidden', type=int, default=64)
    parser.add_argument('--n_global_stack', type=int, default=3)

    parser.add_argument('--n_local', type=int, default=12)
    parser.add_argument('--n_local_filter', type=int, default=6)
    parser.add_argument('--n_local_head', type=int, default=3)
    parser.add_argument('--n_local_hidden', type=int, default=64)
    parser.add_argument('--n_local_stack', type=int, default=3)

    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=64)
    return parser


def main(args):
    data_path = os.path.join(args.dataset_path, args.dataset)
    train_data = TSDataset(data_path + '-train.csv',
                           args.windows, args.horizon)
    torch.save(train_data.scaler, 'scaler.pt')
    val_data = TSDataset(data_path + '-val.csv', args.windows,
                         args.horizon, train_data.scaler)
    # test_data = TSDataset(data_path + '-test.csv', args.windows, args.horizon)
    train_loader = DataLoader(train_data, args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, args.batch_size, shuffle=True)

    
    D = train_data[0][0].shape[-1]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = DSANet(D, args.windows, args.horizon,
                 args.n_global, args.n_local, args.n_local_filter,
                 args.n_global_head, args.n_global_hidden, args.n_global_stack,
                 args.n_local_head, args.n_local_hidden, args.n_local_stack,
                 args.dropout)
    net = net.to(device)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr = args.lr)

    for e in range(1, args.epochs):
        # train one epochs
        train_loss = 0.0
        for index, (X, y) in enumerate(train_loader):
            optimizer.zero_grad()

            yhat = net(X.type(torch.float32).to(device))
            loss = loss_fn(yhat, y.type(torch.float32).to(device))
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        val_loss = 0.0
        with torch.no_grad():
            for (X, y) in val_loader:
                yhat = net(X.type(torch.float32).to(device))
                loss = loss_fn(yhat, y.type(torch.float32).to(device))
                val_loss += loss.item()
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        print('Epoch %d: train loss is %.2f, val loss is %.2f' % (e, train_loss, val_loss))
    
        torch.save(net.state_dict(), 'net-%d-%.2f.pt' % (e, val_loss))


if __name__ == "__main__":
    args = init_parser().parse_args()
    main(args)
