import torch
import torch.nn as nn
from einops import reduce
from torch.autograd import Function

from HGCNMNoEmbed.HGNN import HGNN
from HGCNMNoEmbed.GNN import GNN
from HGCNMNoEmbed.mamba_lmForEEGPara import MambaLMPara, MambaLMConfig
from utils import HMBuilder


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class MLPForHM5(nn.Module):
    def __init__(self, args, inputdim, featuredim, Gdim):
        super().__init__()
        self.args = args
        self.nclass = args.n_class
        self.dropout = args.dropout
        self.l_relu = args.lr
        self.channel = inputdim[1]
        self.seq_len = inputdim[0]
        self.feature_dim = inputdim[2]
        self.Gdim = Gdim

        self.mlp0 = nn.Linear(featuredim, 30)
        self.mlp1 = nn.Linear(self.Gdim * 30, 256)
        self.mlp2 = nn.Linear(256, 16)
        self.mlp5 = nn.Linear(self.feature_dim * 16, 128)
        self.conv1 = nn.Conv1d(1, 16, kernel_size=1)
        self.conv2 = nn.Conv1d(16, 1, kernel_size=1)
        self.conv3 = nn.Conv1d(1, 1, kernel_size=4)
        self.mlp3 = nn.Linear(125, 128)
        self.bn1 = nn.BatchNorm1d(self.feature_dim)
        self.bn2 = nn.BatchNorm1d(1)
        self.bn3 = nn.BatchNorm1d(128)
        self.lrelu = nn.LeakyReLU(self.l_relu)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, input):
        # (b, s, c, f)
        x = self.lrelu(self.mlp0(input))
        x = self.dropout(x).view(x.size(0), x.size(1), -1)
        # x = self.bn(x)
        x = self.lrelu(self.mlp1(x))
        x = self.bn1(x)
        # x = self.dropout(x)
        x = self.lrelu(self.mlp2(x))
        x = self.lrelu(self.mlp5(x.view(x.size(0), -1))).unsqueeze(1)
        x = self.conv2(self.conv1(x))
        x = self.conv3(self.bn2(x)).squeeze(1).squeeze(1)
        x = self.lrelu(self.mlp3(x))
        x = self.bn3(x)
        return x


class STFusion(nn.Module):
    def __init__(self):
        super(STFusion, self).__init__()
        self.gcn = GNN(256, 128, 64)

    def forward(self, x, G):
        return self.gcn(x, G)


class SpatialExt(nn.Module):
    def __init__(self, inputdim=(3, 62, 5)):
        super(SpatialExt, self).__init__()
        self.channel = inputdim[1]
        self.seq_len = inputdim[0]
        self.feature_dim = inputdim[2]
        self.hmBuilder = HMBuilder(inputdim=inputdim)

        self.EAdj = nn.Parameter(torch.Tensor(self.seq_len, self.hmBuilder.helen))
        # self.SAdj = nn.Parameter(torch.Tensor(self.feature_dim, self.channel, self.channel))
        torch.nn.init.uniform_(self.EAdj, a=0, b=1)
        # nn.init.kaiming_uniform_(self.SAdj)

        self.ln = nn.LayerNorm(self.channel)
        # self.hgnn = Chebynet(self.EmbedDim, 20, self.EmbedDim, seqlen=self.seq_len)
        self.hgnn = HGNN(self.feature_dim, 8, 6)
        self.SpatialEmb = nn.Linear(57, 256)

    def forward(self, x):
        # b, s, c, f
        # x = self.continuous_embedding(x.unsqueeze(-1))
        # 根据c、e维度进行矩阵计算，得到余弦相似度矩阵，通过索引筛选脑区相关通道的邻接矩阵，并与batch维共同计算平均距离
        # 得到超图G(s, c, c)
        G = self.getHyperLap(x, self.EAdj)
        # G = self.ln(self.SAdj)
        # G = self.EAdj @ G
        # fx = x.permute(2, 0, 1, 3, 4)  # (f, b, s, c, e)
        output = self.hgnn(x, G).permute(0, 2, 1, 3)
        output = output.reshape(output.shape[0], output.shape[1], -1) # (b, s, c, f1)
        output = self.SpatialEmb(output)
        return output

    def getHyperLap(self, features, edgeWeight, eps=1e-8):
        # norm
        a_n = features.norm(p=2, dim=-1, keepdim=True)
        a_n = torch.clamp(a_n, min=eps)
        a_norm = features / a_n
        # cosine
        sim_matrix = torch.einsum('bscf,bsfd->bscd', a_norm, a_norm.transpose(-1, -2))
        # similar，余弦相似性矩阵元素分布在-1~1之间，加一保证为正，除二正规化到0~1的范围
        sim_matrix = (sim_matrix + 1) / 2
        # 构造根据余弦相似性矩阵计算的超图矩阵并计算LapMat
        H = self.hmBuilder.GetLapByAdjNoEmbed(sim_matrix, edgeWeight)
        return H


class TemporalExt(nn.Module):
    def __init__(self, args, inputdim):
        super(TemporalExt, self).__init__()
        self.args = args
        self.channel = inputdim[1]
        self.seq_len = inputdim[0]
        self.feature_dim = inputdim[2]
        self.EmbedDim = 5
        self.vocab_size = 128
        self.mamba = self.getMamba(self.channel * self.feature_dim, self.vocab_size)
        self.bn = nn.BatchNorm1d(self.seq_len)
        self.TemporalEmb = nn.Linear(128, 256)

    def getMamba(self, d_model, vocab_size):
        config = MambaLMConfig(d_model=d_model, n_layers=2, vocab_size=vocab_size)
        return MambaLMPara(config, self.args)

    def forward(self, x):
        mambaOut = self.mamba(x)
        mambaOut = self.bn(mambaOut)
        mambaOut = self.TemporalEmb(mambaOut)
        return mambaOut


class SpatialTemporalFushion(nn.Module):
    def __init__(self, dropout, inputdim, Gdim):
        super(SpatialTemporalFushion, self).__init__()
        self.channel = inputdim[1]
        self.seq_len = inputdim[0]
        self.feature_dim = inputdim[2]
        self.dropout = dropout
        self.Gdim = Gdim
        self.STAdj = nn.Parameter(torch.Tensor(self.Gdim, self.Gdim))
        nn.init.kaiming_uniform_(self.STAdj)
        # nn.init.xavier_normal_(self.STAdj)
        # torch.nn.init.uniform_(self.STAdj, a=0, b=1)
        self.fre = STFusion()
        self.dropout = nn.Dropout(self.dropout)
        self.Fresqueeze = nn.Sequential(nn.Linear(128, 40),
                                        nn.Flatten(),
                                        nn.Linear(self.Gdim * 40, 128),
                                        )

    def forward(self, x):
        # STG = self.getLap(catput.detach())
        catput = self.fre(x, self.STAdj)

        # output = self.squeeze(catput)
        catput = self.dropout(catput)
        catput = self.Fresqueeze(catput)
        return catput

    def getLap(self, features, eps=1e-8):
        # norm
        a_n = features.norm(p=2, dim=-1, keepdim=True)
        a_n = torch.clamp(a_n, min=eps)
        a_norm = features / a_n
        # cosine
        sim_matrix = torch.einsum('bfce,bfed->bfcd', a_norm, a_norm.transpose(-1, -2))
        # similar，余弦相似性矩阵元素分布在-1~1之间，加一保证为正，除二正规化到0~1的范围
        sim_matrix = (sim_matrix + 1) / 2
        sim_matrix = reduce(sim_matrix.permute(1, 0, 2, 3), "f b c e -> f c e", "mean")
        # 构造根据余弦相似性矩阵计算的超图矩阵并计算LapMat
        LapMat = []
        for HM in sim_matrix:
            LapMat.append(self.GetLapMatByMatrix(HM))
        LapMat = torch.stack(LapMat)
        return LapMat

    def GetLapMatByMatrix(self, Matrix, eps=1e-2):
        D = torch.diag(torch.clamp(torch.sum(Matrix, dim=1), min=eps)).to(Matrix.device)
        I = torch.eye(D.shape[0]).to(Matrix.device)
        try:
            DI = torch.sqrt(torch.inverse(D))
        except:
            print(Matrix)
        else:
            LapMat = I - DI @ Matrix @ DI
            return LapMat


class HGCNMPara(nn.Module):
    def __init__(self, args, inputdim=(3, 62, 5), logger=None):
        super(HGCNMPara, self).__init__()
        self.args = args
        self.nclass = self.args.n_class
        self.ndomain = self.args.n_domain
        self.dropout = self.args.dropout
        self.channel = inputdim[1]
        self.seq_len = inputdim[0]
        self.feature_dim = inputdim[2]
        self.EmbedDim = 5
        self.vocab_size = 128
        self.Gdim = 0
        self.Fusiondim = 0
        # self.continuous_embedding = nn.Parameter(torch.Tensor(3, 5, 1, self.EmbedDim))
        # # stdv = 1. / math.sqrt(self.continuous_embedding.size(1))
        # # self.continuous_embedding.data.uniform_(-stdv, stdv)
        # nn.init.xavier_uniform_(self.continuous_embedding)
        # self.continuous_embedding = MultiActivationEmbeddingLayer(1, 1)

        self.spaExt = SpatialExt(inputdim)
        self.temExt = TemporalExt(self.args, inputdim)

        if hasattr(self, 'spaExt'):
            self.Gdim += self.channel
        if hasattr(self, 'temExt'):
            self.Gdim += self.seq_len

        self.STFusion = SpatialTemporalFushion(self.dropout, inputdim, self.Gdim)

        # self.mlp = MLPForHM5(self.args, inputdim, 256, self.Gdim)
        if hasattr(self, 'STFusion'):
            self.Fusiondim += 128
        if hasattr(self, 'mlp'):
            self.Fusiondim += 128
        self.fc = nn.Linear(self.Fusiondim, self.nclass)
        self.domain = nn.Linear(self.Fusiondim, self.ndomain)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, alpha=0):
        return self.forwardnew(x, alpha)

    def forwardnew(self, x, alpha=0):
        # (b, s, c, f)
        B, S, C, F = x.shape
        catput = None
        if hasattr(self, 'spaExt') and hasattr(self, 'temExt'):
            sout = self.spaExt(x)
            tout = self.temExt(x)
            catput = torch.cat((sout, tout), dim=-2)
        elif hasattr(self, 'temExt'):
            catput = self.temExt(x)
        elif hasattr(self, 'spaExt'):
            catput = self.spaExt(x)

        catout = self.STFusion(catput)
        # catput = self.mlp(catput)
        # out = self.softmax(self.fc(output))
        out = self.fc(catout)
        weight = 0  #self.STAdj.data.clone().norm(p=2, dim=-1).sum() # + self.Fresqueeze[2].weight.data.clone().norm(p=2, dim=-1).sum()
        domain = ReverseLayerF.apply(catout, alpha)
        domain = self.softmax(self.domain(domain))
        # domain = self.domain(domain)
        return out, domain, weight
