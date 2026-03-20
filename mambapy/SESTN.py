import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from torch.autograd import Function

from hgnn import HGNN
from hgnn.GNN import GNN
from mambapy.mamba_lmForEEGPara import MambaLMPara, MambaLMConfig
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


class MultiActivationEmbeddingLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MultiActivationEmbeddingLayer, self).__init__()
        # 定义线性层
        # self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # 通过线性层
        # linear_output = self.linear(x)
        linear_output = x
        # 使用不同的激活函数提取特征
        relu_output = F.relu(linear_output)  # ReLU
        sigmoid_output = torch.sigmoid(linear_output)  # Sigmoid
        tanh_output = torch.tanh(linear_output)  # Tanh
        softplus_output = F.softplus(linear_output)  # Softplus
        leaky_relu_output = F.leaky_relu(linear_output, 0.2)  # Leaky ReLU

        # 将所有输出连接在一起
        features = torch.cat((relu_output,
                              sigmoid_output,
                              tanh_output,
                              softplus_output,
                              leaky_relu_output), dim=-1)

        return features


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

        # mlp
        self.mlp0 = nn.Linear(featuredim, 30)
        self.mlp1 = nn.Linear(self.Gdim * 30, 256)
        self.mlp2 = nn.Linear(256, 16)
        self.mlp5 = nn.Linear(self.feature_dim * 16, 128)
        self.conv1 = nn.Conv1d(1, 16, kernel_size=1)
        self.conv2 = nn.Conv1d(16, 1, kernel_size=1)
        self.conv3 = nn.Conv1d(1, 1, kernel_size=4)
        self.mlp3 = nn.Linear(125, 128)
        # self.mlp4 = nn.Linear(64, self.nclass)
        # self.mlp6 = nn.Linear(64, 2)

        # common
        # self.layer_norm = nn.LayerNorm([30])
        self.bn1 = nn.BatchNorm1d(self.feature_dim)
        self.bn2 = nn.BatchNorm1d(1)
        self.bn3 = nn.BatchNorm1d(128)
        self.lrelu = nn.LeakyReLU(self.l_relu)
        self.dropout = nn.Dropout(self.dropout)
        # self.att_dropout = nn.Dropout(0.9)

        # self.lm_head = nn.Linear(self.config.d_model, self.lm_config.vocab_size, bias=False)
        # self.lm_head.weight = self.embedding.weight
        # self.mlp = nn.Linear(self.lm_config.vocab_size * 3, 4, bias=False)
        self.softmax = nn.Softmax(dim=-1)

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
        # logits = self.mlp4(x)
        # x = ReverseLayerF.apply(x, 0.1)
        # domain = self.softmax(self.mlp6(x))
        # logits = self.softmax(logits)
        return x


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
        # self.hgnn = Chebynet(self.EmbedDim, 20, self.EmbedDim, seqlen=self.seq_len)

        self.n_hid = 6
        self.n_class = 8
        self.hgnn = HGNN(self.EmbedDim, self.n_class, self.n_hid, seqlen=self.seq_len)
        self.SpatialEmb = nn.Linear((self.EmbedDim + self.n_class + self.n_hid) * self.seq_len, 256)

    def forward(self, x, F, B, C):
        # x = self.continuous_embedding(x.unsqueeze(-1))
        # 根据c、e维度进行矩阵计算，得到余弦相似度矩阵，通过索引筛选脑区相关通道的邻接矩阵，并与batch维共同计算平均距离
        # 得到超图G(f, c, c)
        G = self.getHyperLap(x, self.EAdj)
        # G = self.EAdj @ G
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
        # mambaOut = self.mambaList(fx).view(-1, 5, 3, 128)
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
        fx = x.permute(0, 1, 3, 2, 4).reshape(b, s, c, f*e)  # (f, b, s, c, e) (b, s, f*c, e)性能好
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
        # nn.init.xavier_normal_(self.STAdj)
        # torch.nn.init.uniform_(self.STAdj, a=0, b=1)
        self.fre = FrequenceSqueeze()
        self.dropout = nn.Dropout(self.dropout)
        self.Fresqueeze = nn.Sequential(nn.Linear(128, 40),
                                        nn.Flatten(),
                                        nn.Linear(5 * self.Gdim * 40, 128),
                                        )

    def forward(self, x):
        # STG = self.getLap(catput.detach())
        catput = self.fre(x, self.STAdj)

        # output = self.squeeze(catput)
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


class HGCNMEmbedPara(nn.Module):
    def __init__(self, args, inputdim=(3, 62, 5), logger=None):
        super(HGCNMEmbedPara, self).__init__()
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
        # self.continuous_embedding = nn.Linear(1, self.EmbedDim, bias=False)
        # self.continuous_embedding = nn.Parameter(torch.ones(3, 5, 1, self.EmbedDim))
        self.continuous_embedding = nn.Parameter(torch.Tensor(3, 5, 1, self.EmbedDim))
        # self.continuous_embedding.data[0, 0, 0, 0] = -6.6141e-27
        # self.continuous_embedding.data[0, 0, 0, 1] = 1.1084e-42
        # if logger is not None:
        #     logger.info("continuous_embedding: {}".format(self.continuous_embedding.data))
        # stdv = 1. / math.sqrt(self.continuous_embedding.size(1))
        # self.continuous_embedding.data.uniform_(-stdv, stdv)
        nn.init.xavier_uniform_(self.continuous_embedding)
        # self.continuous_embedding = MultiActivationEmbeddingLayer(1, 1)

        self.spaExt = SpatialExt(inputdim, self.EmbedDim)
        # self.spaExt = GCNExt(inputdim, self.EmbedDim)
        self.temExt = TemporalExt(self.args, inputdim, self.EmbedDim)

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
        x = x.permute(0, 1, 3, 2)  # (b, s, f, c)
        x = x.unsqueeze(-1) @ self.continuous_embedding  # (b, s, f, c, e)

        # sout = self.spaExt(x, F, B, C)
        #
        # tout = self.temExt(x)
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
        # catout = self.mlp(catput)
        # STAdj = 0
        # out = self.softmax(self.fc(output))
        out = self.fc(catout)
        weight = 0  #self.STAdj.data.clone().norm(p=2, dim=-1).sum() # + self.Fresqueeze[2].weight.data.clone().norm(p=2, dim=-1).sum()
        domain = ReverseLayerF.apply(catout, alpha)
        domain = self.softmax(self.domain(domain))
        # domain = self.domain(domain)
        return out, domain, weight, STAdj, G, freOut
