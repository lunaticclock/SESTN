import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from einops import reduce
from mne.channels import make_dig_montage

import evaluate_SEED
import evaluate_SEEDIV

matplotlib.use("TKAgg")
plt.rcParams.update({'font.size': 14})

mode = "multi"  # 可切换为 "multi" 启用多子图模式


def drawBrain(data, datasetName="SEED"):
    config = {
        "single": {
            "figsize": (4, 4),
            "ncols": 1,
            "titles": [datasetName],  # 单图使用数据集名称
            "suptitle": "Hypergraph Adj",
            "data_indices": [0]  # 仅使用 data0
        },
        "multi": {
            "figsize": (18, 4),
            "ncols": 5,
            "titles": ["Delta", "Theta", "Alpha", "Beta", "Gamma"],  # 多图波段标题
            "suptitle": "Sub Band Adj",
            "data_indices": range(5)  # 使用 data0~data5
        }
    }
    for i, l in enumerate(data):
        features_min = np.min(l)
        features_max = np.max(l)
        data[i] = (l - features_min) / (features_max - features_min)

    df = pd.read_excel('./1020.xlsx', header=0)
    # 提取电极名称和坐标
    electrodes = df['Unnamed: 0'].tolist()
    positions = df[[0, 1, 2]].values
    custom_ch_pos = {e: np.array(p) for e, p in zip(electrodes, positions)}
    montage = make_dig_montage(ch_pos=custom_ch_pos, coord_frame='head')
    ch_names = ['Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5',
                'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8',
                'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4',
                'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'Oz', 'O2',
                'CB2']  # 示例通道名（需与实际数据一致）
    info = mne.create_info(ch_names, sfreq=250, ch_types='eeg')  # 创建模拟的EEG信息
    info.set_montage(montage)  # 绑定电极位置

    # 初始化画布
    fig = plt.figure(figsize=config[mode]["figsize"])
    grid = gridspec.GridSpec(
        ncols=config[mode]["ncols"],
        nrows=1,
        figure=fig,
        wspace=0.05,
        hspace=0.005
    )

    # 创建子图并绘制
    axs = []
    for i, data_idx in enumerate(config[mode]["data_indices"]):
        ax = fig.add_subplot(grid[0, i])
        im, _ = mne.viz.plot_topomap(
            data=data[data_idx],
            pos=info,
            show=False,
            cmap='RdBu_r',
            axes=ax,
            outlines='head'
        )
        ax.set_title(config[mode]["titles"][i], y=-0.2)
        axs.append(ax)

    # 统一颜色条和主标题
    fig.colorbar(im, ax=axs, fraction=0.01)
    fig.suptitle(config[mode]["suptitle"], fontsize=16, y=0.0, ha='center')
    plt.show()


def drawBrain1(data):
    features_min = np.min(data)
    features_max = np.max(data)
    data = (data - features_min) / (features_max - features_min)

    df = pd.read_excel('./1020.xlsx', header=0)
    # 提取电极名称和坐标
    electrodes = df['Unnamed: 0'].tolist()
    positions = df[[0, 1, 2]].values
    custom_ch_pos = {e: np.array(p) for e, p in zip(electrodes, positions)}
    montage = make_dig_montage(ch_pos=custom_ch_pos, coord_frame='head')
    ch_names = ['Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5',
                'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8',
                'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4',
                'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'Oz', 'O2',
                'CB2']  # 示例通道名（需与实际数据一致）
    info = mne.create_info(ch_names, sfreq=250, ch_types='eeg')  # 创建模拟的EEG信息
    info.set_montage(montage)  # 绑定电极位置

    fig = plt.figure(figsize=(18, 4))
    gridlayout = gridspec.GridSpec(ncols=1, nrows=1, figure=fig, wspace=0.05, hspace=0.005)
    axs0 = fig.add_subplot(gridlayout[0, 0])
    im, cn = mne.viz.plot_topomap(data=data, pos=info, show=False, cmap='RdBu_r', axes=axs0, outlines='head')
    axs0.set_title('all', y=-0.2)
    fig.colorbar(im, ax=axs0)  # 添加颜色条
    fig.suptitle("Sub Band Adj", fontsize=16, y=0.0, ha='center')
    plt.show()


if __name__ == "__main__":
    # 时空学习的两个模型参数
    a, ag = evaluate_SEED.main(modelpath='2025_03_09_00_15_20')
    b, bg = evaluate_SEEDIV.main(modelpath='2025_03_08_19_09_40')
    a = a[:, :-3, :-3]
    b = b[:, :-3, :-3]
    a = np.diagonal(a, axis1=1, axis2=2)
    a.flags.writeable = True
    b = np.diagonal(b, axis1=1, axis2=2)
    b.flags.writeable = True
    drawBrain(a, "SEED")
    drawBrain(b, "SEED-IV")
