import argparse
import torch
from torch.utils.data import DataLoader
from data import TSDataset
from model import DSANet
import numpy as np
import os

from loss import *

def init_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
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
    
    parser.add_argument('--model', type=str)
    parser.add_argument('--scaler', type=str)

    parser.add_argument('--batch_size', type=int, default=64)

    parser.add_argument('--metrics', type=str, nargs='+')

    return parser

def main(args):
    data_path = os.path.join(args.dataset_path, args.dataset)
    scaler = torch.load(args.scaler)
    test_data = TSDataset(data_path + '-test.csv',
                           args.windows, args.horizon, scaler)
    test_loader = DataLoader(test_data, args.batch_size)
    D = test_data[0][0].shape[-1]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = DSANet(D, args.windows, args.horizon,
                 args.n_global, args.n_local, args.n_local_filter,
                 args.n_global_head, args.n_global_hidden, args.n_global_stack,
                 args.n_local_head, args.n_local_hidden, args.n_local_stack,
                 args.dropout)
    loss_fns = []
    for metric in args.metrics:
        if metric == 'RMSE':
            loss_fns.append(RMSE)
        elif metric == 'MSE':
            loss_fns.append(MSE)
        elif metric == 'MAE':
            loss_fns.append(MAPE)
        elif metric == 'RRSE':
            loss_fns.append(RRSE)
        elif metric == 'MAPE':
            loss_fns.append(MAPE)
        else:
            loss_fns.append(lambda yhat,y: np.nan)
    
    net.load_state_dict(torch.load(args.model))
    net = net.to(device)
    test_losses = [0.0 for i in range(len(loss_fns))]
    
    with torch.no_grad():
        for (X, y) in test_loader:
            yhat = net(X.type(torch.float32).to(device)).to('cpu').numpy()
            y = y.to('cpu').numpy()
            for i, loss_fn in enumerate(loss_fns):
                loss = loss_fn(yhat, y)
                test_losses[i] += loss 
    for metric, loss in zip(args.metrics, test_losses):
        print('%s: %.2f' % (metric, np.mean(loss))) 

if __name__ == "__main__":
    args = init_argparser().parse_args()
    main(args)