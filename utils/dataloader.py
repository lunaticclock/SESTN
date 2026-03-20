import os
import re

import numpy as np
import torch


def load_data_de(path, subject):

    dict_load = np.load(os.path.join(path, (str(subject))), allow_pickle=True)
    data = dict_load[()]['sample']
    label = dict_load[()]['label']
    split_index = dict_load[()]["clip"]

    x_tr = data[:split_index]
    x_ts = data[split_index:]
    y_tr = label[:split_index]
    y_ts = label[split_index:]

    data_and_label = {
        "x_tr": x_tr,
        "x_ts": x_ts,
        "y_tr": y_tr,
        "y_ts": y_ts
    }
    if 'ndata' in dict_load[()].keys():
        data_and_label['ndata'] = dict_load[()]['ndata']
        data_and_label['nlabel'] = dict_load[()]['nlabel']

    return data_and_label


def load_seed5data_de(path, subject):

    dict_load = np.load(os.path.join(path, (str(subject))), allow_pickle=True)
    data = dict_load[()]['sample']
    label = dict_load[()]['label']
    split_index = dict_load[()]["clip"]
    fold_data, fold_label = [], []
    fold_data.append(data[:split_index[0]])
    fold_data.append(data[split_index[0]:split_index[1]])
    fold_data.append(data[split_index[1]:])
    fold_label.append(label[:split_index[0]])
    fold_label.append(label[split_index[0]:split_index[1]])
    fold_label.append(label[split_index[1]:])

    data_and_label = {
        "fold_data": fold_data,
        "fold_label": fold_label,
    }

    return data_and_label


def load_data_inde(path, subject):
    x_tr = None
    y_tr = None
    x_ts = None
    y_ts = None
    domain_tr = None
    domain_ts = None

    count = 0
    for i_subject in os.listdir(path):
        dict_load = np.load(os.path.join(path, (str(i_subject))), allow_pickle=True)
        data = dict_load[()]['sample']
        label = dict_load[()]['label']
        if i_subject == subject:
            x_ts = data
            y_ts = label
            domain_ts = None
        else:
            if x_tr == None:
                x_tr = data
                y_tr = label
                domain_tr = torch.ones(label.shape[0], dtype=torch.int64) * count
                count += 1
            else:
                x_tr = torch.cat((x_tr, data), dim=0)
                y_tr = torch.cat((y_tr, label), dim=0)
                domain_tr = torch.cat((domain_tr, (torch.ones(label.shape[0], dtype=torch.int64) * count)), dim=0)
                count += 1

    data_and_label = {
        "x_tr": x_tr,
        "x_ts": x_ts,
        "y_tr": y_tr,
        "y_ts": y_ts,
        "domain_tr": domain_tr,
        "domain_ts": domain_ts,
    }
    return data_and_label

