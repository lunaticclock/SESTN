import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter


class GNN(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, dropout=0.5):
        super(GNN, self).__init__()
        self.dropout = dropout
        self.hgc1 = GNN_Layer(in_ch, n_hid)
        self.hgc2 = GNN_Layer(n_hid, n_class)

    def forward(self, x, G):
        x1 = F.relu(self.hgc1(x, G))
        x1d = F.dropout(x1, self.dropout)
        x2 = self.hgc2(x1d, G)
        return x2  # torch.cat((x, x1, x2), dim=-1)


class GNN_Layer(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(GNN_Layer, self).__init__()
        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        # x = G.unsqueeze(1).unsqueeze(1).repeat(1, x.shape[1], 1, 1).matmul(x)
        # x = G.unsqueeze(1).repeat(1, x.shape[1], 1, 1, 1).matmul(x)
        return x
