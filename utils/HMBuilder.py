import torch
from einops import reduce


class HMBuilder:
    def __init__(self, DisAdj=None, inputdim=(3, 62, 5)):
        self.channel = inputdim[1]
        self.seq_len = inputdim[0]
        self.feature_dim = inputdim[2]
        if self.channel == 62:
            # 根据脑区划分设置的超边，采用两种脑区划分，一种7脑区，一种2脑区（半球）
            self.heset = [
                [0, 1, 2, 3, 4],
                [5, 6, 7, 14, 15, 16, 23, 24, 25],
                [23, 24, 25, 32, 33, 34, 41, 42, 43],
                [11, 12, 13, 20, 21, 22, 29, 30, 31],
                [29, 30, 31, 38, 39, 40, 47, 48, 49],
                [7, 8, 9, 10, 11, 16, 17, 18, 19, 20, 25, 26, 27, 28, 29, 34, 35, 36, 37, 38, 43, 44, 45, 46, 47],
                [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61],
                [0, 3, 5, 6, 7, 8, 14, 15, 16, 17, 23, 24, 25, 26, 32, 33, 34, 35, 41, 42, 43, 44, 50, 51, 52, 57, 58],
                [2, 4, 10, 11, 12, 13, 19, 20, 21, 22, 28, 29, 30, 31, 37, 38, 39, 40, 46, 47, 48, 49, 54, 55, 56, 60,
                 61]
            ]
        elif self.channel == 32:
            # ts 32 channel
            self.heset = self.Get32heset()

        self.helen = len(self.heset)

        HyperM = torch.zeros(self.helen, self.channel)

        # 遍历 heset，设置 HyperM 中的相应位置为 1
        for index, indices in enumerate(self.heset):
            HyperM[index, indices] = 1
        self.LapMat = self.GetLapMatByHyperM(HyperM)
        if DisAdj is not None:
            self.DisLap = self.GetLapByDistance(DisAdj)

    def GetLapByAdj(self, Adj):
        HyperM = torch.zeros(self.seq_len, self.helen, self.channel).to(Adj.device)
        # 遍历 heset，设置 HyperM 中的相应位置为 1
        for index, indices in enumerate(self.heset):
            # 构造对应节点的单位矩阵
            I = torch.eye(len(indices)).to(Adj.device)
            # 求对应节点到超边其他节点的距离
            simVec = reduce((Adj[:, :, indices, :][:, :, :, indices] - I), 'b f c d -> b f c', 'sum') / (
                        len(indices) - 1)
            simVec = simVec.permute(1, 0, 2)
            simVec = torch.mean(simVec, dim=1)
            HyperM[:, index, indices] = simVec
        HyperM = HyperM.reshape(-1, self.channel)
        LapMat = self.GetLapMatByHyperM(HyperM)
        return LapMat

    def GetLapByDistance(self, Distance):
        HyperM = torch.zeros(self.helen, self.channel).to(Distance.device)
        for index, indices in enumerate(self.heset):
            # 构造对应节点的单位矩阵
            I = torch.eye(len(indices)).to(Distance.device)
            # 求对应节点到超边其他节点的距离
            simVec = reduce((Distance[indices, :][:, indices] - I), 'c d -> c', 'sum') / (len(indices) - 1)
            HyperM[index, indices] = simVec
        LapMat = self.GetLapMatByHyperM(HyperM)
        return LapMat

    def GetLapByAdjEmbed(self, Adj, edgeWeight):
        HyperM = torch.zeros(self.seq_len, self.feature_dim, self.helen, self.channel).to(Adj.device)
        # 遍历 heset，设置 HyperM 中的相应位置为 1
        for index, indices in enumerate(self.heset):
            # 构造对应节点的单位矩阵
            I = torch.eye(len(indices)).to(Adj.device)
            # 求对应节点到超边其他节点的距离
            simVec = reduce((Adj[:, :, :, indices, :][:, :, :, :, indices] - I), 'b s f c d -> b s f c', 'sum') / (
                        len(indices) - 1)
            simVec = simVec.permute(1, 2, 0, 3)
            simVec = torch.mean(simVec, dim=2)
            HyperM[:, :, index, indices] = simVec
        HyperM = HyperM.permute(1, 0, 2, 3).reshape(self.feature_dim, -1, self.channel)
        # HyperM = HyperM.reshape(-1, 9, 62)
        LapMat = []
        for HM, w in zip(HyperM, edgeWeight):
            LapMat.append(self.GetLapMatByHyperM((w * HM.T).T))
        LapMat = torch.stack(LapMat)
        return LapMat

    def GetLapByAdjNoEmbed(self, Adj, edgeWeight):
        HyperM = torch.zeros(self.seq_len, self.helen, self.channel).to(Adj.device)
        # 遍历 heset，设置 HyperM 中的相应位置为 1
        for index, indices in enumerate(self.heset):
            # 构造对应节点的单位矩阵
            I = torch.eye(len(indices)).to(Adj.device)
            # 求对应节点到超边其他节点的距离
            simVec = reduce((Adj[:, :, indices, :][:, :, :, indices] - I), 'b s c f -> b s c', 'sum') / (
                        len(indices) - 1)
            simVec = torch.mean(simVec, dim=0)
            HyperM[:, index, indices] = simVec
        # HyperM = HyperM.reshape(-1, self.channel)
        # HyperM = HyperM.reshape(-1, 9, 62)
        # LapMat = self.GetLapMatByHyperM((edgeWeight * HyperM.T).T)
        LapMat = []
        for HM, w in zip(HyperM, edgeWeight):
            LapMat.append(self.GetLapMatByHyperM((w * HM.T).T))
        LapMat = torch.stack(LapMat)
        return LapMat

    def GetLapMatByHyperM(self, HyperM):
        DEMat = torch.diag(torch.sum(HyperM.detach(), dim=1)).to(HyperM.device)
        DVMat = torch.diag(torch.sum(HyperM.detach(), dim=0)).to(HyperM.device)
        DVI = torch.sqrt(torch.inverse(DVMat))
        I = torch.eye(DVMat.shape[0]).to(HyperM.device)
        inverse_DEMat = torch.inverse(DEMat)
        AMat = HyperM.T @ inverse_DEMat @ HyperM
        LapMat = I - DVI @ AMat @ DVI
        return LapMat

    def Get32heset(self):
        channel = [0, 2, 3, 4, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49,
                   52, 54, 58, 59, 60]
        # 原始分组结构（基于62通道的索引）
        heset = [
            [0, 2, 3, 4],
            [5, 7, 15, 23, 25],
            [23, 25, 33, 41, 43],
            [11, 13, 21, 29, 31],
            [29, 31, 39, 47, 49],
            [7, 9, 11, 17, 19, 25, 27, 29, 35, 37, 43, 45, 47],
            [52, 54, 58, 59, 60],
            [0, 3, 5, 7, 15, 17, 23, 25, 33, 35, 41, 43, 52, 58],
            [2, 4, 11, 13, 19, 21, 29, 31, 37, 39, 47, 49, 54, 60]
        ]

        # 创建反向映射字典（原通道号 -> 新索引）
        channel_mapping = {orig: new_idx for new_idx, orig in enumerate(channel)}

        # 重新映射分组结构
        new_heset = []
        for group in heset:
            new_group = [channel_mapping[orig] for orig in group]
            new_heset.append(new_group)
        return new_heset
