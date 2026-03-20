import torch
from torch import nn
from hgnn.layers import HGNN_conv
import torch.nn.functional as F


class HGNN(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, dropout=0.5, seqlen=3):
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid, seqlen=seqlen)
        self.hgc2 = HGNN_conv(n_hid, n_class, seqlen=seqlen)

    def forward(self, x, G):
        x1 = F.relu(self.hgc1(x, G))
        x1d = F.dropout(x1, self.dropout)
        x2 = self.hgc2(x1d, G)
        return torch.cat((x, x1, x2), dim=-1)
