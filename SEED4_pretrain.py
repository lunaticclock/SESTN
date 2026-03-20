# -------------------------------------
# SEED4数据的预处理代码 将数据打包到npy文件中
# -------------------------------------


import os

import einops
import numpy as np
import torch
from scipy.io import loadmat


def extend_normal(sample):
    for i in range(len(sample)):
        features_min = torch.min(sample[i])
        features_max = torch.max(sample[i])
        sample[i] = (sample[i] - features_min) / (features_max - features_min)
    return sample


def eeg_data(label_list, trial_list, raw_path, path):
    if not os.path.exists(path):
        os.makedirs(path)
    channel = [0, 2, 3, 4, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 52,
               54, 58, 59, 60]  # 使用TSception中设定的32个通道
    # channel = [7, 11, 25, 27, 29, 31, 43, 47, 58, 60]  # 使用ISRUC中设定的10个通道
    # channel = [14, 22]  # 左右耳
    allSubSamples = []
    for subject in os.listdir(raw_path):
        subject_name = str(subject).strip('.mat')
        allTrialSamples = 0
        data = None  # 存储每一个subject的数组
        label = None
        clip = None
        frequency = "de_LDS"
        count = 0

        for i in trial_list:
            dataKey = frequency + str(i + 1)
            metaData = np.array((loadmat(os.path.join(raw_path, subject_name), verify_compressed_data_integrity=False)[
                dataKey])).astype('float')  # 读取到原始的三维元数据
            # trMetaData = einops.rearrange(metaData[channel, :, :], 'w h c -> h w c')  # (42,channel,5)
            trMetaData = einops.rearrange(metaData, 'w h c -> h w c')  # [:, channel, :]  # (42,62,5)
            allTrialSamples += trMetaData.shape[0]
            count += 1
            x = torch.from_numpy(np.array(trMetaData)).type(torch.float32).unfold(0, size=3, step=3).permute(0, 3, 1, 2)
            y = torch.from_numpy(np.array([label_list[i], ] * x.shape[0])).type(torch.float32)
            # n1x, n1y = add_gaussian_noise(x, y)
            if data is None:
                data = x
                label = y
            else:
                data = torch.cat((data, x), dim=0)
                label = torch.cat((label, y), dim=0)

            if count == 16:
                clip = data.shape[0]
        allSubSamples.append(allTrialSamples)
        print("subject:", subject, "Samples:", allTrialSamples)
        # 对于data进行min-max归一化
        data = extend_normal(data)
        subt_data = data[:clip]
        subv_data = data[clip:]
        subt_label = label[:clip]
        subv_label = label[clip:]
        # 增强数据
        # ndata, nlabel = inter4aug(subt_data, subt_label)
        # subt_data = torch.cat((subt_data, ndata), dim=0)
        # subt_label = torch.cat((subt_label, nlabel), dim=0)
        clip = subt_data.shape[0]
        sub_data = torch.cat((subt_data, subv_data), dim=0)
        sub_label = torch.cat((subt_label, subv_label), dim=0)
        dict = {'sample': sub_data, 'label': sub_label, 'clip': clip}
        np.save(os.path.join(path, (str(subject).strip('.mat') + ".npy")), dict)


def add_gaussian_noise(eeg_signal, label, noise_std=0.01):
    noise = torch.randn_like(eeg_signal) * noise_std
    return eeg_signal + noise, label.clone()


if __name__ == "__main__":
    """Resave all data to .npy"""
    raw_data_path = "SEED_IV/eeg_feature_smooth/"
    data_path = "SEED_IV/final/"  # 存储合并的两种数据
    # 对于seed4数据集，由于类别的排布不均匀，所以对于每个session的trial需要进行调整
    session = 1
    label_list = [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3]
    # 选取每种类型的后2个trial用于测试
    trial_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 18, 19, 12, 13, 16, 17, 20, 21, 22, 23]
    print(f"Load session {session}...")
    eeg_data(label_list, trial_list, os.path.join(raw_data_path, str(session)), os.path.join(data_path, str(session)))

    session = 2
    label_list = [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1]
    # 选取每种类型的后2个trial用于测试
    trial_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 17, 12, 16, 18, 19, 20, 21, 22, 23]
    print(f"Load session {session}...")
    eeg_data(label_list, trial_list, os.path.join(raw_data_path, str(session)), os.path.join(data_path, str(session)))

    session = 3
    label_list = [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0]
    # 选取每种类型的后2个trial用于测试
    trial_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 18, 19, 10, 14, 16, 17, 20, 21, 22, 23]
    print(f"Load session {session}...")
    eeg_data(label_list, trial_list, os.path.join(raw_data_path, str(session)), os.path.join(data_path, str(session)))

    print(f"Resave data succeed.")
