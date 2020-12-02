import math
import torch
from torch import nn
from torch.nn import functional as F

from collections import OrderedDict



def attention(q, k, v, scale):
    scores = torch.bmm(q.permute(1, 0, 2), k.permute(1, 2, 0)) / scale
    attn = torch.softmax(scores, 2)
    return torch.bmm(attn, v.permute(1, 0, 2)).permute(1, 0, 2)

class MultiheadAttention(nn.Module):

    def __init__(self, n_feature, n_head, dropout = 0.1):
        """Multihead Attention Module
        MultiHead(Q, K, V) = Concat(head_1, ..., head_n) W^o
        where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
        args:
            n_feature: the number of feature
            num_head: the number of heads
            dropout: the rate of dropout
        """
        super(MultiheadAttention, self).__init__()
        self.n_feature = n_feature
        self.n_head = n_head
        dk = n_feature // n_head
        
        assert dk * self.n_head == self.n_feature

        self.scale = math.sqrt(dk)

        self.dropout = nn.Dropout(dropout)

        self.qfc = nn.Linear(n_feature, n_feature)
        self.kfc = nn.Linear(n_feature, n_feature)
        self.vfc = nn.Linear(n_feature, n_feature)
        
        self.ofc = nn.Linear(n_feature, n_feature)


    def forward(self, x):
        """
        shape:
            query,key,value: T x batch_size x n_feature
        """
        querys = self.qfc(x).chunk(3, -1)
        keys = self.kfc(x).chunk(3, -1)
        values = self.vfc(x).chunk(3, -1)
        
        v_a = torch.cat([attention(q, k, v, self.scale) for q,k,v in zip(querys, keys, values)], -1)
        return self.ofc(v_a)


class ResLayerNorm(nn.Module):

    def __init__(self, sublayer, shapes):
        super(ResLayerNorm, self).__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(shapes)

    def forward(self, x):
        x = x + self.sublayer(x)
        # return self.norm(x.permute(1, 0, 2)).permute(1, 0, 2)
        return self.norm(x)


class PositionWiseFeedForward(nn.Module):

    def __init__(self, n_in, n_hidden, dropout = 0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.w1 = nn.Linear(n_in, n_hidden)
        self.w2 = nn.Linear(n_hidden, n_in)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_t = F.relu(self.w1(x))
        return self.dropout(self.w2(x_t))


class SelfAttention(nn.Module):

    def __init__(self, D, n_feature, n_head, n_hidden, dropout = 0.1):
        super(SelfAttention, self).__init__()
        shape = (D, n_feature)
        
        self.attn = ResLayerNorm(MultiheadAttention(n_feature, n_head, dropout), shape)
        self.ff = ResLayerNorm(PositionWiseFeedForward(n_feature, n_hidden), shape)

    def forward(self, x):
        return self.ff(self.attn(x))

class AR(nn.Module):

    def __init__(self, T):
        super(AR, self).__init__()
        self.linear = nn.Linear(T, 1)

    def forward(self, x):
        return self.linear(x.permute(0, 2, 1)).permute(0, 2, 1)


class DSANet(nn.Module):

    def __init__(self, D, T, n_out, n_global, n_local, L, 
                    n_global_head, n_global_hidden, n_global_attn, 
                    n_local_head, n_local_hidden, n_local_attn,
                    dropout = 0.1):
        """Dual Self-Attention Network for Multivariate Time Series Forecasting

        Args:
            D: the number of univariate time series
            T: the input window size
            n_global: the number of global temporal convolution filter 
            n_local: the number of local temporal convolution filter
            L: the local window size for local temporal convolution filter
        """
        super(DSANet, self).__init__()

        self.global_temporal_conv = nn.Conv2d(1, n_global, (T, 1))
        self.local_tempooral_conv = nn.Conv2d(1, n_local, (L, 1))
        self.pool = nn.MaxPool2d((T-L+1, 1))
        
        self.local_stack = nn.Sequential(OrderedDict(
            [('l_attn_%d' % i, SelfAttention(D, n_local, n_local_head, n_local_hidden, dropout)) for i in range(n_local_attn)]
        ))
        self.global_stack = nn.Sequential(OrderedDict(
            [('g_attn_%d' % i, SelfAttention(D, n_global, n_global_head, n_global_hidden, dropout)) for i in range(n_global_attn)]
        ))
        self.ar = AR(T)
        self.fc = nn.Linear(n_global + n_local, n_out)

    def forward(self, X):
        """
        args:
            X: [n_batch, T, D]
        """
        X_i = X.view(X.size(0), 1, X.size(1), X.size(2))
        X_g = F.relu(self.global_temporal_conv(X_i))[:,:,0,:]
        X_l = self.pool(self.local_tempooral_conv(X_i))[:,:,0,:]
        # X_g: [n_batch, n_feature, T] -> [n_batch, T, n_feature]
        X_g = self.global_stack(X_g.permute(0, 2, 1))
        X_l = self.local_stack(X_l.permute(0, 2, 1))
        X_cat = torch.cat([X_g, X_l], -1)
        fc_out = self.fc(X_cat).permute(0, 2, 1)
        ar = self.ar(X)
        return ar + fc_out


if __name__ == "__main__":
    from data import TSDataset
    d = TSDataset('../dataset/electricity-train.csv', 32, 6)
    dataloader = torch.utils.data.DataLoader(d, batch_size=10, shuffle=True)
    for x, y in dataloader:
        pass
    D = 321
    T = 32
    n_global = 12
    n_local = 12
    L = 10
    n_gloabl_head = 3
    n_local_head = 3
    n_global_hidden = 128
    n_local_hidden = 128
    n_gloabl_attn = 3
    n_local_attn = 2
    dropout = 0.1
    n_out = 6

    net = DSANet(D, T, n_out, n_global, n_local, L,
        n_gloabl_head,n_global_hidden, n_gloabl_attn,
        n_local_head,n_local_hidden, n_local_attn, dropout)
    yhat = net(x.type(torch.float32))
    print(yhat.size(), y.size())