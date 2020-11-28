import pandas as pd
from torch.utils.data import Dataset


class TSDataset(Dataset):

    def __init__(self, root, T, k, transforms=[]):
        super(TSDataset, self).__init__()
        self.root = root
        self.T = T
        self.k = k
        self.data = pd.read_csv(root, header=0).values
        for transform in transforms:
            self.data = transform(self.data)
        self.len = len(self.data) - T - k + 1

    def __getitem__(self, index):
        return self.data[index:index + self.T, :], self.data[index + self.T:index + self.T + self.k, :]
    
    def __len__(self):
        return self.len

if __name__ == "__main__":
    dataset = TSDataset('../dataset/electricity-train.csv', 5, 5)
    X, y = dataset[len(dataset) - 1]
    print(X.shape, y.shape)
