from sys import flags
import pandas as pd
import math
import os
import argparse

def main(args):
    d = pd.read_csv(args.input, header=None)
    train_size = math.ceil(len(d) * args.train_rate)
    val_size = math.ceil(len(d) * args.val_rate)

    if args.output_name:
        name = args.output_name
    else:
        _, filename = os.path.split(args.input)
        name, _ = os.path.splitext(filename)
    
    train_path = os.path.join(args.output, name + '-train.csv')
    val_path = os.path.join(args.output, name + '-val.csv')
    test_path = os.path.join(args.output, name + '-test.csv')
    
    d[:train_size].to_csv(train_path, index=0, header=0)
    d[train_size:train_size+val_size].to_csv(val_path, index=0, header=0)
    d[train_size+val_size:].to_csv(test_path, index=0, header=0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split dataset to train/val/test')
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument('--train_rate', default=0.6, type=float)
    parser.add_argument('--val_rate', default=0.2, type=float)
    parser.add_argument('--output_name', type=str)

    args = parser.parse_args()
    main(args)

