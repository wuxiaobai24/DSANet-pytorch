import torch

class ToTensor:

    def __init__(self):
        pass

    def __call__(self, data, *args, **kwds):
        return torch.Tensor(data)

class Scaler:

    def __init__(self, scaler, fit=False):
        self.scaler = scaler
        self.fit = fit

    def __call__(self, data):
        if self.fit:
            self.scaler = self.scaler.fit(data)
        return self.scaler.transform(self.data)
    
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
