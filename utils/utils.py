import datetime
import logging
import os
import random
import re

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from scipy.signal import welch
from scipy.stats import wasserstein_distance
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from torch.nn import functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader

from utils import Timer


class CE_Label_Smooth_Loss(nn.Module):
    def __init__(self, epsilon=0.14):
        super(CE_Label_Smooth_Loss, self).__init__()
        self.epsilon = epsilon

    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * self.epsilon / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.epsilon))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss


def set_logging_config(logdir):
    """
    logging configuration
    :param logdir:
    :return:
    """

    def beijing(sec, what):
        beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
        return beijing_time.timetuple()

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    logging.Formatter.converter = beijing

    logging.basicConfig(format="[%(asctime)s] [%(name)s] %(message)s",
                        level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(logdir, ("log.txt"))),
                                  logging.StreamHandler(os.sys.stdout)])


def save_checkpoint(state, is_best, dir, subject_name):
    """
    save the checkpoint during training stage
    :param state: content to be saved
    :param is_best: if model's performance is the best at current step
    :param exp_name: experiment name
    :return: None
    """
    torch.save(state, os.path.join(f'{dir}', f'{subject_name}_model_best.pth.tar'))
    # if is_best:
    #     shutil.copyfile(os.path.join(f'{dir}', f'{subject_name}_checkpoint.pth.tar'),
    #                     os.path.join(f'{dir}', f'{subject_name}_model_best.pth.tar'))


def normalize_adj(adj):
    D = torch.diag(torch.sum(adj, dim=1))
    D_ = torch.diag(torch.diag(1 / torch.sqrt(D)))  # D^(-1/2)
    lap_matrix = torch.matmul(D_, torch.matmul(adj, D_))

    return lap_matrix


# def de_train_test_split_3fold(data, label, index1, index2, config):

#     x_train = np.array([])
#     x_test = np.array([])
#     y_train = np.array([])
#     y_test = np.array([])

#     # if str(config["dataset_name"]) == "SEED5":
#     #     new_data = split_eye_data(data, config["sup_node_num"])

#     new_data = data[:,:310]
#     new_data = einops.rearrange(new_data, "w (h c) -> w h c", c=5)  # (235,62,5)


#     x1 = new_data[:index1]
#     x2 = new_data[index1:index2]
#     x3 = new_data[index2:]

#     y1 = label[:index1]
#     y2 = label[index1:index2]
#     y3 = label[index2:]

#     if config["cfold"] == 1:
#         x_train = np.append(x2, x3, axis=0)
#         x_test = x1
#         y_train = np.append(y2, y3, axis=0)
#         y_test = y1

#     elif config["cfold"] == 2:
#         x_train = np.append(x1, x3, axis=0)
#         x_test = x2
#         y_train = np.append(y1, y3, axis=0)
#         y_test = y2

#     else:
#         x_train = np.append(x1, x2, axis=0)
#         x_test = x3
#         y_train = np.append(y1, y2, axis=0)
#         y_test = y3


#     data_and_label = {"x_train": x_train,
#                       "x_test": x_test,
#                       "y_train": y_train,
#                       "y_test": y_test}

#     return data_and_label


# class Data_split(object):

#     def __init__(self, args):
#         super(Data_split, self).__init__()
#         self.args = args
#         self.subject_index = self.args.subject_index
#         self.sample_list = self.args.dataloader.sampleList
#         self.label_list = self.args.dataloader.labelList
#         self.split_index = self.args.dataloader.split_index[self.subject_index] # for dependent only


#         self.data_and_label = None
#         if self.args.mode.upper() == "DEPENDENT":
#             self.data_and_label = self.de_train_test_split()

#         if self.args.mode.upper() == "INDEPENDENT":
#             self.data_and_label = self.inde_train_test_split()


#     ## for dependent experiment
#     def de_train_test_split(self):

#         x_train = self.sample_list[self.subject_index][:self.split_index]
#         x_test = self.sample_list[self.subject_index][self.split_index:]
#         y_train = self.label_list[self.subject_index][:self.split_index]
#         y_test = self.label_list[self.subject_index][self.split_index:]

#         data_and_label = {"x_train": x_train,
#                           "x_test": x_test,
#                           "y_train": y_train,
#                           "y_test": y_test}

#         return data_and_label

#     ## for independent experiment
#     def inde_train_test_split(self):
#         x_train, x_test, y_train, y_test = np.array([]), np.array([]), \
#                                            np.array([]), np.array([])

#         for j in range(len(self.sample_list)):
#             if j == self.subject_index:
#                 x_test = self.sample_list[j]
#                 y_test = self.label_list[j]
#             else:
#                 if x_train.shape[0] == 0:
#                     x_train = self.sample_list[j]
#                     y_train = self.label_list[j]
#                 else:
#                     x_train = np.append(x_train, self.sample_list[j], axis=0)
#                     y_train = np.append(y_train, self.label_list[j], axis=0)

#         data_and_label = {"x_train": x_train,
#                           "x_test": x_test,
#                           "y_train": y_train,
#                           "y_test": y_test}

#         return data_and_label


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=20, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_max = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, subject_name, val_acc, model, epoch):

        score = val_acc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(subject_name, val_acc, model, epoch)
        elif score <= self.best_score + self.delta:
            pass
            # self.counter += 1
            # self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            # if self.counter >= self.patience:
            #     self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(subject_name, val_acc, model, epoch)
            self.counter = 0

    def save_checkpoint(self, subject_name, val_acc, model, epoch):
        '''Saves model when validation acc increase.'''
        if self.verbose:
            self.trace_func(
                f'Validation acc increased ({self.val_acc_max:.6f} --> {val_acc:.6f}) in epoch ({epoch}).  Saving model ...')
        model_save_path = self.path + subject_name + ".pt"
        torch.save(model.state_dict(), model_save_path)
        self.val_acc_max = val_acc


def initialize_weights(module):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, (nn.Module, nn.Sequential)):
        for sub_module in module.children():
            initialize_weights(sub_module)


def PrintScore(true, pred, savePath=None, average='macro'):
    # savePath=None -> console, else to Result.txt
    if savePath == None:
        saveFile = None
    else:
        saveFile = open(savePath + "Result.txt", 'a+')
    # Main scores
    F1 = metrics.f1_score(true, pred, average=None)
    print("Main scores:")
    print('Acc\tF1S\tKappa\tSad\tNatural\tHappy', file=saveFile)
    print('%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f' %
          (metrics.accuracy_score(true, pred),
           metrics.f1_score(true, pred, average=average),
           metrics.cohen_kappa_score(true, pred),
           F1[0], F1[1], F1[2]),
          file=saveFile)
    # Classification report
    print("\nClassification report:", file=saveFile)
    print(metrics.classification_report(true, pred,
                                        target_names=['Sad', 'Natural', 'Happy'],
                                        digits=4), file=saveFile)
    # Confusion matrix
    print('Confusion matrix:', file=saveFile)
    print(metrics.confusion_matrix(true, pred), file=saveFile)
    # Overall scores
    print('\n    Accuracy\t', metrics.accuracy_score(true, pred), file=saveFile)
    print(' Cohen Kappa\t', metrics.cohen_kappa_score(true, pred), file=saveFile)
    print('    F1-Score\t', metrics.f1_score(true, pred, average=average), '\tAverage =', average, file=saveFile)
    print('   Precision\t', metrics.precision_score(true, pred, average=average), '\tAverage =', average, file=saveFile)
    print('      Recall\t', metrics.recall_score(true, pred, average=average), '\tAverage =', average, file=saveFile)
    if savePath != None:
        saveFile.close()
    return


def ConfusionMatrix(y_true, y_pred, classes, savePath, title=None, cmap=plt.cm.Blues):
    if not title:
        title = 'Confusion matrix'
    # Compute confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    cm_n = cm
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j] * 100, '.2f') + '%\n' + format(cm_n[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(savePath + title + ".pdf")
    plt.show()
    return ax


def VariationCurve(fit, val, yLabel, savePath, figsize=(9, 6)):
    plt.figure(figsize=figsize)
    fitx = range(1, len(fit) + 1)
    valx = range(1, len(val) + 1)
    plt.plot(fitx, fit, label='Train')
    plt.plot(valx, val, label='Val')
    last_x = fitx[-1]
    last_y = fit[-1]
    plt.text(last_x, last_y, f'{last_y}', fontsize=12)
    val_x = valx[-1]
    val_y = val[-1]
    plt.text(val_x, val_y, f'{val_y}', fontsize=12)
    plt.title('Model ' + yLabel)
    plt.xlabel('Epochs')
    plt.ylabel(yLabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(savePath + 'Model_' + yLabel + '.png')
    plt.show()
    return


# 定义一个函数递归查找某个网络层
def find_layer(model, layer_name):
    for name, layer in model.named_children():
        if name == layer_name:
            return layer
        elif isinstance(layer, nn.Sequential) or isinstance(layer, nn.Module):
            # 如果当前层是Sequential，递归查找
            found_layer = find_layer(layer, layer_name)
            if found_layer is not None:
                return found_layer


def get_layer_sizes(model):
    global layer_sizes
    for name, layer in model.named_children():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            layer_sizes.append((name, layer.weight.data.size()))
        elif isinstance(layer, (nn.Sequential, nn.Module)):
            # 如果当前层是Sequential，递归查找
            get_layer_sizes(layer)


def find_files_by_subject(directory, prefix):
    """
    在指定目录中找到具有特定前缀的文件。
    :param directory: 要搜索的目录路径。
    :param prefix: 文件名前缀。
    :return: 文件路径列表。
    """
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith(prefix):
                file_list.append(os.path.join(root, file))
    return file_list


def getDomain(idx_fold, fold_long):
    onehotencoder = OneHotEncoder()
    domain = torch.zeros(15, fold_long)
    for i in range(15):
        domain[i] = i
    domain = torch.tensor(onehotencoder.fit_transform(
        domain.reshape(-1, 1)).toarray(), dtype=torch.float32).reshape(15, -1, 15)
    domain_train = domain[~(np.arange(15) == idx_fold)].reshape(-1, 15)
    domain_test = domain[idx_fold].reshape(-1, 15)
    return domain_train, domain_test


def GetAllDistanceInCos(x, fs):
    allmatrix = torch.stack(
        [torch.stack(
            [GetWassersteinDistanceInCos(step)
             for step in per.unfold(1, size=fs, step=fs).permute(1, 0, 2)])
            for per in x])

    return allmatrix


def GetWassersteinDistanceInCos(eeg_data):
    # 获取通道数
    num_channels, _ = eeg_data.size()
    dot_product = torch.matmul(eeg_data, eeg_data.T)

    # 计算每个向量的模
    norm = torch.norm(eeg_data, p=2, dim=1, keepdim=True)

    # 计算余弦相似性矩阵
    cosine_similarity = dot_product / (norm * norm.T)
    return cosine_similarity


def GetAllDistance(x, fs):
    allmatrix = torch.stack(
        [torch.stack(
            [GetWassersteinDistance(step)
             for step in per.unfold(1, size=fs, step=fs).permute(1, 0, 2)])
            for per in x])

    return allmatrix


def GetWassersteinDistance(eeg_data):
    # 获取通道数
    num_channels, _ = eeg_data.size()
    connectivity_matrix = np.array([[wasserstein_distance(eeg_data[i], eeg_data[j])
                                     for j in range(num_channels)]
                                    for i in range(num_channels)])
    connectivity_matrix = connectivity_matrix
    return torch.reciprocal(torch.from_numpy(connectivity_matrix) + torch.eye(num_channels, num_channels))


def findu(curve, window_size=50):
    # 定义滑动窗口的大小
    # window_size = 50  # 可以根据你的需求调整

    # 对于每一个可能的窗口位置，计算窗口内的数据的二次拟合系数
    u_trends = []
    i = 0
    while i < len(curve) - window_size + 1:
        window = curve[i:i + window_size]
        x = np.arange(window_size)
        coef = np.polyfit(x, window, 2)

        # 如果二次项的系数小于0，那么这个窗口就可能是一个倒U型趋势
        if coef[0] < 0:
            # 检查左侧是否是上升趋势，右侧是否是下降趋势
            left_trend = np.mean(np.diff(window[:window_size // 2])) > 0
            right_trend = np.mean(np.diff(window[window_size // 2:])) < 0
            if left_trend and right_trend:
                u_trends.append((i, i + window_size))
                i += window_size  # 跳过这个窗口内的其他点
            else:
                i += 1
        else:
            i += 1
    if (len(u_trends) != 0):
        print(len(u_trends))
        # # 绘制原始曲线
        # plt.figure(figsize=(10, 6))
        # plt.plot(curve, label='Original Curve')
        #
        # # 标记所有的倒U型趋势
        # for start, end in u_trends:
        #     plt.plot(range(start, end), curve[start:end], label='U Trend from {} to {}'.format(start, end))
        # plt.title('Inverted U trend Curve')
        # plt.xlabel('Time(Sec)')
        # plt.ylabel('Similarity')
        # plt.legend()
        # plt.grid(True)
        # plt.show()
    else:
        # 绘制原始曲线
        plt.figure(figsize=(10, 6))
        plt.plot(curve, label='Original Curve')
        plt.title('Inverted U trend Curve')
        plt.xlabel('Time(Sec)')
        plt.ylabel('Similarity')
        plt.legend()
        plt.grid(True)
        plt.show()
    return u_trends


def differential_entropy(data, fs=200, nperseg=2000):
    """
        计算EEG数据的差分熵特征
        :param data: 原始EEG数据，形状为 (samples, channels)
        :param fs: 采样率，单位为Hz
        :param nperseg: 每段数据的长度
        :return: 差分熵特征，形状为 (segments, channels, frequency_bands)
        """
    # 定义频带
    frequency_bands = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 50)}
    band_names = list(frequency_bands.keys())

    samples, channels = data.shape
    num_segments = samples // nperseg
    de_features = []

    for segment in range(num_segments):
        segment_features = []
        start = segment * nperseg
        end = start + nperseg
        for channel in range(channels):
            segment_data = data[start:end, channel]
            f, Pxx = welch(segment_data, fs=fs, nperseg=nperseg)
            band_features = []
            for band in band_names:
                fmin, fmax = frequency_bands[band]
                band_power = np.mean(Pxx[(f >= fmin) & (f < fmax)])
                de = 0.5 * np.log2(2 * np.pi * np.e * band_power)
                band_features.append(de)
            segment_features.append(band_features)
        de_features.append(segment_features)

    de_features = np.array(de_features)
    return de_features


def local_weighted_smoothing(features, kernel_size=20, sigma=2):
    """
    对提取的特征进行LDS平滑
    :param features: 提取的特征，形状为 (features, channels)
    :param kernel_size: 核大小
    :param sigma: 高斯核标准差
    :return: 平滑后的特征
    """
    feature_len, channels = features.shape
    smoothed_features = np.zeros((feature_len, channels))

    # 生成高斯核
    kernel = np.exp(-0.5 * (np.arange(kernel_size) - kernel_size // 2) ** 2 / sigma ** 2)
    kernel = kernel / kernel.sum()

    for channel in range(channels):
        # 将卷积核和数据进行卷积
        smoothed_channel = np.convolve(features[:, channel], kernel, mode='same')
        smoothed_features[:, channel] = smoothed_channel

    return smoothed_features


def LDSSingleChannel(features, kernel_size=20, sigma=2):
    """
    对提取的特征进行LDS平滑
    :param features: 提取的特征，形状为 (features, channels)
    :param kernel_size: 核大小
    :param sigma: 高斯核标准差
    :return: 平滑后的特征
    """
    # 生成高斯核
    kernel = np.exp(-0.5 * (np.arange(kernel_size) - kernel_size // 2) ** 2 / sigma ** 2)
    kernel = kernel / kernel.sum()

    # 将卷积核和数据进行卷积
    smoothed_channel = np.convolve(features, kernel, mode='same')
    smoothed_features = smoothed_channel

    return smoothed_features


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def inter3aug(timg, label):
    # tmig(也就是整个dataloader): batchnum * seqlength * channel * feature
    # label: batchnum * 1
    aug_data = []
    aug_label = []

    batch_size, seqlength, channel, feature = timg.shape

    for cls3aug in range(3):  #因为是我的任务是3分类，所以我打算每个类别都做一个增强，即每个类别增强 batchsize / 3个
        cls_idx = torch.where(label == cls3aug)
        tmp_data = timg[cls_idx]  # 从整个dataloader中取到了298个同一个类别的
        tmp_label = label[cls_idx]

        tmp_aug_data = torch.zeros((tmp_data.shape[0], seqlength, channel, feature))  # 24
        for ri in range(tmp_data.shape[0]):  # 24
            for rj in range(feature):
                # rand_idx = np.random.randint(0, tmp_data.shape[0], 5)
                # 使用np.random.randint函数生成5个随机整数作为索引（对应特征的维数5），范围从0到tmp_data数组的长度   0 - 298中选5个随机数
                rand_idx = torch.randint(0, tmp_data.shape[0], (feature,))
                tmp_aug_data[ri, :, :, rj:(rj + 1)] = tmp_data[rand_idx[rj], :, :, rj:(rj + 1)]

        aug_data.append(tmp_aug_data)  # 有值的[24，6，62，5] * 4
        aug_label.append(tmp_label[:tmp_data.shape[0]])  # [24] * 4
    aug_data = torch.cat(aug_data)  # [72，6，62，5]
    aug_label = torch.cat(aug_label)  # [72]
    aug_shuffle = torch.randperm(len(aug_data))
    aug_data = aug_data[aug_shuffle, :, :]
    aug_label = aug_label[aug_shuffle]

    # aug_data = torch.from_numpy(aug_data).cuda()
    # aug_data = aug_data.float()
    # aug_label = torch.from_numpy(aug_label).cuda()
    # aug_label = aug_label.long()
    # 输出 [72,6,62,5],[72,1]
    return aug_data, aug_label


def inter4aug(timg, label):
    # tmig(也就是整个dataloader): batchnum * seqlength * channel * feature
    # label: batchnum * 1
    aug_data = []
    aug_label = []

    batch_size, seqlength, channel, feature = timg.shape

    for cls3aug in range(4):  #因为是我的任务是3分类，所以我打算每个类别都做一个增强，即每个类别增强 batchsize / 3个
        cls_idx = torch.where(label == cls3aug)
        tmp_data = timg[cls_idx]  # 从整个dataloader中取到了298个同一个类别的
        tmp_label = label[cls_idx]

        tmp_aug_data = torch.zeros((tmp_data.shape[0], seqlength, channel, feature))  # 24
        for ri in range(tmp_data.shape[0]):  # 24
            for rj in range(feature):
                # rand_idx = np.random.randint(0, tmp_data.shape[0], 5)
                # 使用np.random.randint函数生成5个随机整数作为索引（对应特征的维数5），范围从0到tmp_data数组的长度   0 - 298中选5个随机数
                rand_idx = torch.randint(0, tmp_data.shape[0], (feature,))
                tmp_aug_data[ri, :, :, rj:(rj + 1)] = tmp_data[rand_idx[rj], :, :, rj:(rj + 1)]

        aug_data.append(tmp_aug_data)  # 有值的[24，6，62，5] * 4
        aug_label.append(tmp_label[:tmp_data.shape[0]])  # [24] * 4
    aug_data = torch.cat(aug_data)  # [72，6，62，5]
    aug_label = torch.cat(aug_label)  # [72]
    aug_shuffle = torch.randperm(len(aug_data))
    aug_data = aug_data[aug_shuffle, :, :]
    aug_label = aug_label[aug_shuffle]

    # aug_data = torch.from_numpy(aug_data).cuda()
    # aug_data = aug_data.float()
    # aug_label = torch.from_numpy(aug_label).cuda()
    # aug_label = aug_label.long()
    # 输出 [72,6,62,5],[72,1]
    return aug_data, aug_label


def inter5aug(timg, label):
    # tmig(也就是整个dataloader): batchnum * seqlength * channel * feature
    # label: batchnum * 1
    aug_data = []
    aug_label = []

    batch_size, seqlength, channel, feature = timg.shape

    for cls5aug in range(5):  #因为是我的任务是3分类，所以我打算每个类别都做一个增强，即每个类别增强 batchsize / 3个
        cls_idx = torch.where(label == cls5aug)
        tmp_data = timg[cls_idx]  # 从整个dataloader中取到了298个同一个类别的
        tmp_label = label[cls_idx]

        tmp_aug_data = torch.zeros((tmp_data.shape[0], seqlength, channel, feature))  # 24
        for ri in range(tmp_data.shape[0]):  # 24
            for rj in range(feature):
                # rand_idx = np.random.randint(0, tmp_data.shape[0], 5)
                # 使用np.random.randint函数生成5个随机整数作为索引（对应特征的维数5），范围从0到tmp_data数组的长度   0 - 298中选5个随机数
                rand_idx = torch.randint(0, tmp_data.shape[0], (feature,))
                tmp_aug_data[ri, :, :, rj:(rj + 1)] = tmp_data[rand_idx[rj], :, :, rj:(rj + 1)]

        aug_data.append(tmp_aug_data)  # 有值的[24，6，62，5] * 4
        aug_label.append(tmp_label[:tmp_data.shape[0]])  # [24] * 4
    aug_data = torch.cat(aug_data)  # [72，6，62，5]
    aug_label = torch.cat(aug_label)  # [72]
    aug_shuffle = torch.randperm(len(aug_data))
    aug_data = aug_data[aug_shuffle, :, :]
    aug_label = aug_label[aug_shuffle]

    # aug_data = torch.from_numpy(aug_data).cuda()
    # aug_data = aug_data.float()
    # aug_label = torch.from_numpy(aug_label).cuda()
    # aug_label = aug_label.long()
    # 输出 [72,6,62,5],[72,1]
    return aug_data, aug_label


class DatasetWithDomain(Dataset):
    def __init__(self, x_tensor, y_tensor, d_tensor=None):
        self.x = x_tensor
        self.d = d_tensor
        self.y = y_tensor

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], (1 if self.d is None else self.d[idx])


def generate_cheby_adj(A, K, device):
    fsupport = []
    for fnum in range(5):
        support = []
        for i in range(K):
            if i == 0:
                # support.append(torch.eye(A.shape[1]).cuda())  #torch.eye生成单位矩阵
                temp = torch.eye(A.shape[1])
                temp = temp.to(device)
                support.append(temp)
            elif i == 1:
                support.append(A[fnum])
            else:
                temp = torch.matmul(support[-1], A[fnum])
                support.append(temp)
        fsupport.append(torch.stack(support, dim=0))
    return fsupport


def normalize_A(A, symmetry=False):
    A = F.relu(A)
    if symmetry:
        A = A + torch.transpose(A, 0, 1)  #A+ A的转置
        d = torch.sum(A, 1)  #对A的第1维度求和
        d = 1 / torch.sqrt(d + 1e-10)  #d的-1/2次方
        D = torch.diag_embed(d)
        L = torch.matmul(torch.matmul(D, A), D)
    else:
        d = torch.sum(A, 1)
        d = 1 / torch.sqrt(d + 1e-10)
        D = torch.diag_embed(d)
        L = torch.matmul(torch.matmul(D, A), D)
    return L


def generate_cheby_adj(A, K, device):
    support = []
    for i in range(K):
        if i == 0:
            # support.append(torch.eye(A.shape[1]).cuda())  #torch.eye生成单位矩阵
            temp = torch.eye(A.shape[1])
            temp = temp.to(device)
            support.append(temp)
        elif i == 1:
            support.append(A)
        else:
            temp = torch.matmul(support[-1], A)
            support.append(temp)
    return support


class eegDataset(Dataset):
    # x_tensor: (sample, channel, datapoint(feature)) type = torch.tensor
    # y_tensor: (sample,) type = torch.tensor

    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.y)


class eegDatasetWithDomain(Dataset):
    # x_tensor: (sample, channel, datapoint(feature)) type = torch.tensor
    # y_tensor: (sample,) type = torch.tensor

    def __init__(self, x_tensor, y_tensor, d_tensor):
        self.x = x_tensor
        self.y = y_tensor
        self.d = d_tensor

    def __getitem__(self, index):
        return self.x[index], self.y[index], (1 if self.d is None else self.d[index])

    def __len__(self):
        return len(self.y)


#文件根据后缀数字排序
def extract_number(filename):
    match = re.match(r'(\d+)_', filename)
    if match:
        return int(match.group(1))
    return 0


class Eval(object):

    def __init__(self, args, subject_name, device):
        self.args = args
        self.subject_name = subject_name
        self.device = device
        self.softmax = nn.Softmax(dim=-1)

    def compute_band_contribution_from_stfusion_output(self, X):
        """
        X shape: [B, F, Node, Feature]
        """

        # spatial aggregation
        Z = X.mean(dim=2)  # [B, F, Feature]

        # representation magnitude
        band_energy = torch.norm(Z, p=2, dim=2)  # [B, F]

        # dataset-level average
        contribution = band_energy.mean(dim=0)

        # normalization for visualization
        contribution = contribution / contribution.sum()

        return contribution.cpu().numpy()

    def eval(self, data_and_label, Model):
        # laplacian_array = []  # 存放该subject优化后的laplacian matrix 列表
        test_data = (data_and_label["x_ts"]).type(torch.FloatTensor)
        test_label = (data_and_label["y_ts"]).type(torch.FloatTensor)
        val_set = TensorDataset(test_data, test_label)

        val_loader = DataLoader(val_set, batch_size=self.args.batch_size, shuffle=True, drop_last=False)

        #####################################################################################
        # 2.define model
        #####################################################################################
        trainnum, seqlength, channel, feature = test_data.shape
        model = Model(self.args, inputdim=(seqlength, channel, feature)).to(self.device)
        _loss = CE_Label_Smooth_Loss(epsilon=self.args.epsilon).to(self.args.device)
        model.load_state_dict(
            torch.load(os.path.join(f'{self.args.log_dir}', f'{self.subject_name}_model_best.pth.tar'),
                       weights_only=False)['enc_module_state_dict'])

        #############################################################################
        # 3.start train
        #############################################################################

        best_val_acc = 0

        t0 = Timer()
        t0.start()
        # train_acc = train_acc / train_set.__len__()
        # train_loss = train_loss / train_set.__len__()
        model.eval()
        val_acc, val_loss = 0, 0
        energy_list = []
        AllPred_temp, AllTrue_temp, AllAdj_temp, AllG_temp = None, None, None, None
        with torch.no_grad():
            for j, (a, b) in enumerate(val_loader):
                a, b = a.to(self.args.device), b.to(device=self.args.device, dtype=torch.int64)
                output, _, _, Adj, G, freOut = model(a)
                energy_list.append(freOut)
                softout = self.softmax(output)
                val_acc += np.sum(np.argmax(output.cpu().data.numpy(), axis=1) == b.cpu().data.numpy())
                if AllPred_temp is None:
                    AllPred_temp = torch.argmax(output, dim=1).cpu().numpy()
                    AllTrue_temp = b.cpu().data.numpy()
                    AllAdj_temp = Adj.cpu().data.numpy()#Adj[:, :, :-3].cpu().data.numpy()
                    AllG_temp = G.cpu().data.numpy()
                else:
                    AllPred_temp = np.concatenate((AllPred_temp, torch.argmax(output, dim=1).cpu().numpy()))
                    AllTrue_temp = np.concatenate((AllTrue_temp, b.cpu().data.numpy()))
                    AllAdj_temp = AllAdj_temp + Adj.cpu().data.numpy()#Adj[:, :, :-3].cpu().data.numpy()
                    AllG_temp = AllG_temp + G.cpu().data.numpy()
                # AllPred_temp += torch.argmax(output, dim=1).cpu().numpy()
                # AllTrue_temp += torch.argmax(b, dim=1).cpu().numpy()
        energy_all = torch.cat(energy_list, dim=0)
        self.compute_band_contribution_from_stfusion_output(energy_all)
        val_acc = round(float(val_acc / val_set.__len__()), 4)

        t0.stop()
        print(f"\n> Finished eval in: {t0.stop()}")
        # self.writer.close()
        return val_acc, AllPred_temp, AllTrue_temp, AllAdj_temp, AllG_temp
