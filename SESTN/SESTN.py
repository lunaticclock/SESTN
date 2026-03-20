import torch
import torch.nn as nn
from einops import reduce
from torch.autograd import Function

from SESTN.mamba_lmForEEGPara import MambaLMPara, MambaLMConfig
from hgnn import HGNN
from hgnn.GNN import GNN
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


class FrequenceSqueeze(nn.Module):
    def __init__(self):
        super(FrequenceSqueeze, self).__init__()
        self.gcn = GNN(256, 128, 64)

    def forward(self, x, G):
        return self.gcn(x, G)


class SpatialExt(nn.Module):
    def __init__(self, inputdim=(3, 62, 5), EmbedDim=5):
        super(SpatialExt, self).__init__()
        self.channel = inputdim[1]
        self.seq_len = inputdim[0]
        self.feature_dim = inputdim[2]
        self.EmbedDim = EmbedDim
        self.hmBuilder = HMBuilder(inputdim=inputdim)

        self.EAdj = nn.Parameter(torch.Tensor(self.feature_dim, self.seq_len * self.hmBuilder.helen))
        torch.nn.init.uniform_(self.EAdj, a=0, b=1)

        self.ln = nn.LayerNorm(self.channel)

        self.n_hid = 6
        self.n_class = 8
        self.hgnn = HGNN(self.EmbedDim, self.n_class, self.n_hid, seqlen=self.seq_len)
        self.SpatialEmb = nn.Linear((self.EmbedDim + self.n_class + self.n_hid) * self.seq_len, 256)

    def forward(self, x, F, B, C):
        # 根据c、e维度进行矩阵计算，得到余弦相似度矩阵，通过索引筛选脑区相关通道的邻接矩阵，并与batch维共同计算平均距离
        # 得到超图G(f, c, c)
        G = self.getHyperLap(x, self.EAdj)
        fx = x.permute(2, 0, 1, 3, 4)  # (f, b, s, c, e)
        output = self.hgnn(fx, G)
        output = output.permute(0, 1, 3, 2, 4).reshape(F, B, C, -1)
        output = self.SpatialEmb(output).permute(1, 0, 2, 3)
        return output, G.detach()

    def getHyperLap(self, features, edgeWeight, eps=1e-8):
        # norm
        a_n = features.norm(p=2, dim=-1, keepdim=True)
        a_n = torch.clamp(a_n, min=eps)
        a_norm = features / a_n
        # cosine
        sim_matrix = torch.einsum('bsfce,bsfed->bsfcd', a_norm, a_norm.transpose(-1, -2))
        # similar，余弦相似性矩阵元素分布在-1~1之间，加一保证为正，除二正规化到0~1的范围
        sim_matrix = (sim_matrix + 1) / 2
        # 构造根据余弦相似性矩阵计算的超图矩阵并计算LapMat
        H = self.hmBuilder.GetLapByAdjEmbed(sim_matrix, edgeWeight)
        return H


class GCNExt(nn.Module):
    def __init__(self, inputdim=(3, 62, 5), EmbedDim=5):
        super(GCNExt, self).__init__()
        self.channel = inputdim[1]
        self.seq_len = inputdim[0]
        self.feature_dim = inputdim[2]
        self.EmbedDim = EmbedDim
        self.hmBuilder = HMBuilder(inputdim=inputdim)

        self.SAdj = nn.Parameter(torch.Tensor(self.feature_dim, self.channel, self.channel))
        nn.init.kaiming_uniform_(self.SAdj)

        self.ln = nn.LayerNorm(self.channel)

        self.n_hid = 6
        self.n_class = 8
        self.hgnn = HGNN(self.EmbedDim, self.n_class, self.n_hid, seqlen=self.seq_len)
        self.SpatialEmb = nn.Linear((self.EmbedDim + self.n_class + self.n_hid) * self.seq_len, 256)

    def forward(self, x, F, B, C):
        G = self.ln(self.SAdj)
        fx = x.permute(2, 0, 1, 3, 4)  # (f, b, s, c, e)
        output = self.hgnn(fx, G)
        output = output.permute(0, 1, 3, 2, 4).reshape(F, B, C, -1)
        output = self.SpatialEmb(output).permute(1, 0, 2, 3)
        return output


class TemporalExt(nn.Module):
    def __init__(self, args, inputdim=(3, 62, 5), EmbedDim=5):
        super(TemporalExt, self).__init__()
        self.args = args
        self.channel = inputdim[1]
        self.seq_len = inputdim[0]
        self.feature_dim = inputdim[2]
        self.EmbedDim = EmbedDim
        self.vocab_size = 128
        self.mambaList = nn.ModuleList(
            [self.getMamba(self.channel * self.EmbedDim, self.vocab_size) for _ in range(self.feature_dim)])
        # self.mambaList = self.getMamba(self.channel * self.EmbedDim * self.feature_dim, self.vocab_size)
        self.bn = nn.BatchNorm2d(self.feature_dim)
        self.TemporalEmb = nn.Linear(128, 256)

    def getMamba(self, d_model, vocab_size):
        config = MambaLMConfig(d_model=d_model, n_layers=2, vocab_size=vocab_size)
        return MambaLMPara(config, self.args)

    def forward(self, x):
        fx = x.permute(2, 0, 1, 3, 4)  # (f, b, s, c, e) # 单mamba模块 x.view(x.size(0), x.size(1), x.size(2), -1)
        mambaOut = []
        for i, mamba in enumerate(self.mambaList):
            mambaOut.append(mamba(fx[i]))
        mambaOut = torch.stack(mambaOut, dim=0).permute(1, 0, 2, 3)
        mambaOut = self.bn(mambaOut)
        mambaOut = self.TemporalEmb(mambaOut)
        return mambaOut


class SingleTemporalExt(nn.Module):
    def __init__(self, args, inputdim=(3, 62, 5), EmbedDim=5):
        super(SingleTemporalExt, self).__init__()
        self.args = args
        self.channel = inputdim[1]
        self.seq_len = inputdim[0]
        self.feature_dim = inputdim[2]
        self.EmbedDim = EmbedDim
        self.vocab_size = 128
        self.mamba = self.getMamba(self.channel * self.EmbedDim * self.feature_dim, self.vocab_size)
        self.bn = nn.BatchNorm2d(self.feature_dim)
        self.TemporalEmb = nn.Linear(128, 256)

    def getMamba(self, d_model, vocab_size):
        config = MambaLMConfig(d_model=d_model, n_layers=2, vocab_size=vocab_size)
        return MambaLMPara(config, self.args)

    def forward(self, x):
        b, s, f, c, e = x.shape
        fx = x.permute(0, 1, 3, 2, 4).reshape(b, s, c, f * e)  # (f, b, s, c, e) (b, s, f*c, e)性能好
        mambaOut = self.mamba(fx).unsqueeze(1).repeat(1, f, 1, 1)
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
        self.STAdj = nn.Parameter(torch.Tensor(self.feature_dim, self.Gdim, self.Gdim))
        nn.init.kaiming_uniform_(self.STAdj)
        self.fre = FrequenceSqueeze()
        self.dropout = nn.Dropout(self.dropout)
        self.Fresqueeze = nn.Sequential(nn.Linear(128, 40),
                                        nn.Flatten(),
                                        nn.Linear(5 * self.Gdim * 40, 128),
                                        )

    def forward(self, x):
        catput = self.fre(x, self.STAdj)
        catput = self.dropout(catput)
        outbeforesqueeze = catput.detach()
        catput = self.Fresqueeze(catput)
        return catput, self.STAdj.detach(), outbeforesqueeze

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


class SESTN(nn.Module):
    def __init__(self, args, inputdim=(3, 62, 5), logger=None):
        super(SESTN, self).__init__()
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
        self.continuous_embedding = nn.Parameter(torch.Tensor(3, 5, 1, self.EmbedDim))
        nn.init.xavier_uniform_(self.continuous_embedding)

        self.spaExt = SpatialExt(inputdim, self.EmbedDim)
        self.temExt = TemporalExt(self.args, inputdim, self.EmbedDim)

        if hasattr(self, 'spaExt'):
            self.Gdim += self.channel
        if hasattr(self, 'temExt'):
            self.Gdim += self.seq_len
        self.STFusion = SpatialTemporalFushion(self.dropout, inputdim, self.Gdim)

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
        x = x.permute(0, 1, 3, 2)  # (b, s, f, c)
        x = x.unsqueeze(-1) @ self.continuous_embedding  # (b, s, f, c, e)

        catput = None
        if hasattr(self, 'spaExt') and hasattr(self, 'temExt'):
            sout, G = self.spaExt(x, F, B, C)
            tout = self.temExt(x)
            catput = torch.cat((sout, tout), dim=-2)
        elif hasattr(self, 'temExt'):
            catput = self.temExt(x)
        elif hasattr(self, 'spaExt'):
            catput, G = self.spaExt(x, F, B, C)

        catout, STAdj, freOut = self.STFusion(catput)
        out = self.fc(catout)
        weight = 0  #self.STAdj.data.clone().norm(p=2, dim=-1).sum() # + self.Fresqueeze[2].weight.data.clone().norm(p=2, dim=-1).sum()
        domain = ReverseLayerF.apply(catout, alpha)
        domain = self.softmax(self.domain(domain))
        return out, domain, weight, STAdj, G, freOut
