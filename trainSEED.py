import datetime
import math
import os
import re

import pytorch_warmup as warmup
from torch.nn import Parameter
from torch.utils.data import TensorDataset, DataLoader

from args import parse_args
from mambapy.SESTN import HGCNMEmbedPara
from utils import Timer, convert_dis_m, get_ini_dis_m, return_coordinates, seed_everything, save_checkpoint
from utils import load_data_de, load_data_inde
from utils import set_logging_config, CE_Label_Smooth_Loss

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging
import torch

import numpy as np

# Automated device selection based on available backends
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available() and False
    else "cpu"
)

print(f"> Using {device} device")


def listdir_nohidden(path):
    files = []
    for f in os.listdir(path):
        if not f.startswith('.'):
            files.append(f"{path}/{f}")
    return files


def load_checkpoint(filepath, model, scheduler, optimizer):
    print(f"> Loading model from: {filepath}")
    try:
        loaded_checkpoint = torch.load(filepath, map_location=device)

        loaded_epoch = loaded_checkpoint['epoch']
        loaded_model = model
        loaded_scheduler = scheduler
        loaded_optimizer = optimizer

        loaded_model.load_state_dict(loaded_checkpoint['model_state'])
        if scheduler is not None:
            loaded_scheduler.load_state_dict(loaded_checkpoint['scheduler_state'])
        if optimizer is not None:
            loaded_optimizer.load_state_dict(loaded_checkpoint['optimizer_state'])

        print("> Loaded model")
        return True, loaded_epoch, loaded_model, loaded_scheduler, loaded_optimizer
    except Exception as e:
        print("> Cannot load model")
        return False, 0, model, scheduler, optimizer


class Trainer(object):

    def __init__(self, args, subject_name):
        self.args = args
        self.subject_name = subject_name

    def train(self, data_and_label):
        model_path = f'saves/model.pth'
        backup_path = f"saves/model-b.pth"
        train_epoch = self.args.epochs
        logger = logging.getLogger("train")
        # laplacian_array = []  # 存放该subject优化后的laplacian matrix 列表
        train_data = (data_and_label["x_tr"]).type(torch.FloatTensor)
        train_label = (data_and_label["y_tr"]).type(torch.FloatTensor)
        ndata = (data_and_label["ndata"]) if 'ndata' in data_and_label.keys() else None
        # ndata = (data_and_label["ndata"]).type(torch.FloatTensor)
        nlabel = (data_and_label["nlabel"]) if 'nlabel' in data_and_label.keys() else None
        # nlabel = (data_and_label["nlabel"]).type(torch.FloatTensor)
        test_data = (data_and_label["x_ts"]).type(torch.FloatTensor)
        test_label = (data_and_label["y_ts"]).type(torch.FloatTensor)
        # train_data = torch.cat((train_data, ndata), dim=0)
        # train_label = torch.cat((train_label, nlabel), dim=0)
        train_set = TensorDataset(train_data, train_label)
        val_set = TensorDataset(test_data, test_label)

        train_loader = DataLoader(train_set, batch_size=self.args.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=self.args.batch_size, shuffle=True, drop_last=False)

        # 局部视野的邻接矩阵
        adj_matrix = Parameter(torch.FloatTensor(convert_dis_m(get_ini_dis_m(), 9)), requires_grad=False).to(self.args.device)

        # 返回节点的绝对坐标
        coordinate_matrix = torch.FloatTensor(return_coordinates()).to(self.args.device)

        #####################################################################################
        # 2.define model
        #####################################################################################
        trainnum, seqlength, channel, feature = train_data.shape
        model = HGCNMEmbedPara(self.args, inputdim=(seqlength, channel, feature)).to(device)
        Spatial_params, Temporal_params, weight_params = [], [], []
        for pname, p in model.named_parameters():
            #     print(pname)
            if "temExt" in str(pname):
                Temporal_params += [p]
            elif "spaExt" in str(pname):
                Spatial_params += [p]
            else:
                weight_params += [p]

        optim = torch.optim.AdamW([
            {'params': Spatial_params, 'lr': self.args.tlr},
            {'params': Temporal_params, 'lr': self.args.lr},
            {'params': weight_params, 'lr': self.args.tlr},
        ], betas=(0.9, 0.999), weight_decay=self.args.weight_decay)
        # optim = torch.optim.Adam(model.parameters(), weight_decay=self.args.weight_decay)

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optim,
                                                            milestones=[train_epoch // 3],
                                                            gamma=0.1)
        warmup_scheduler = warmup.UntunedLinearWarmup(optim)
        warmup_scheduler.last_step = -1
        _loss = CE_Label_Smooth_Loss(epsilon=self.args.epsilon).to(self.args.device)
        # Load previously trained weights
        ''' 
        If the model is the same it will load previous weights located in specified path
        If the model differs or the path is empty it'll skip loading and train from scratch
        '''
        # _, epoch, model, scheduler, optim = load_checkpoint(model_path, model, lr_scheduler, optim)

        #############################################################################
        # 3.start train
        #############################################################################

        best_val_acc = 0

        t0, t1 = Timer(), Timer()
        t0.start()
        t1.start()
        for epoch in range(train_epoch):
            alpha = 2.0 / (1 + math.exp(-10 * (float(epoch) / train_epoch))) - 1
            train_acc = 0
            train_loss = 0
            val_loss = 0
            val_acc = 0
            avg_loss = 0
            model.train()
            for i, (x, y) in enumerate(train_loader):
                model.zero_grad()  # 清空上一步残余更新参数值

                x, y = x.to(self.args.device), y.to(device=self.args.device, dtype=torch.int64)
                # alpha = 2.0 / (1 + math.exp(-10 * (float(i) / len(train_loader)))) - 1
                output, domainx, weight, _, _ = model(x, alpha)
                loss = _loss(output, y) + weight
                lossall = loss
                if self.args.mode == 'independent':
                    domainy = torch.ones(y.shape[0], device=y.device, dtype=torch.int64)
                    lossd = _loss(domainx, domainy)
                    lossall = lossall + lossd
                lossall.backward()  # 误差反向传播，计算参数更新值
                avg_loss += loss.item()
                optim.step()  # 将参数更新值施加到net的parmeters上
                if i < len(train_loader) - 1:
                    with warmup_scheduler.dampening():
                        pass
                train_acc += np.sum(np.argmax(output.cpu().data.numpy(), axis=1) == y.cpu().data.numpy())
                train_loss += loss.item() * y.size(0)
            with warmup_scheduler.dampening():
                lr_scheduler.step()
            # train_acc = train_acc / train_set.__len__()
            # train_loss = train_loss / train_set.__len__()
            model.eval()
            with torch.no_grad():
                for j, (a, b) in enumerate(val_loader):
                    a, b = a.to(self.args.device), b.to(device=self.args.device, dtype=torch.int64)
                    output = model(a)[0]

                    val_acc += np.sum(np.argmax(output.cpu().data.numpy(), axis=1) == b.cpu().data.numpy())
                    batch_loss = _loss(output, b)
                    val_loss += batch_loss.item() * b.size(0)

            val_acc = round(float(val_acc / val_set.__len__()), 4)
            val_loss = round(float(val_loss / val_set.__len__()), 4)

            is_best_acc = 0
            if best_val_acc < val_acc:
                best_val_acc = val_acc
                is_best_acc = 1
                save_checkpoint({
                    'iteration': epoch,
                    'enc_module_state_dict': model.state_dict(),
                    'test_acc': val_acc,
                    'optimizer': optim.state_dict(),
                }, is_best_acc, self.args.log_dir, self.subject_name)

            if epoch == 0:
                logger.info(self.args)

            if epoch % 5 == 0:
                logger.info(
                    "time for five epoch: {}, val acc and loss on epoch_{} are: {} and {}".format(t1.stop(), epoch,
                                                                                                  val_acc, val_loss))
                t1.start()

            if best_val_acc == 1:
                break

        t0.stop()
        print(f"\n> Finished training in: {t0.stop()}")
        # self.writer.close()
        return best_val_acc  # , laplacian_array


def extract_number(filename):
    match = re.match(r'(\d+)_', filename)
    if match:
        return int(match.group(1))
    return 0


def main(models=None, lrs=None, datapath=None, mode=None):
    args = parse_args("SEED")
    if models is not None:
        args.model = models
    if lrs is not None:
        args.lr = lrs
    if datapath is not None:
        args.datapath = datapath
    if mode is not None:
        args.mode = mode
    print("")
    print(f"Current device is {args.device}.")
    seed_everything(args.seed)
    # 将当前的实验结果存储到按秒组织的文件夹中
    datatime_path = '/' + (datetime.datetime.now() + datetime.timedelta(hours=8)).strftime('%Y_%m_%d_%H_%M_%S')
    args.log_dir = args.log_dir + args.dataset + datatime_path
    set_logging_config(args.log_dir)
    logger = logging.getLogger("main")
    logger.info("Logs and checkpoints will be saved to：{}".format(args.log_dir))

    # summary(model, input_size=(62, 5))

    acc_list = []
    acc_dic = {}
    count = 0
    sessions = range(0, 3)
    # true_path = os.path.join(args.datapath, str(args.session))
    for session in sessions:
        true_path = os.path.join(args.datapath, f'session{session}')
        args.session = session
        for subject in sorted(os.listdir(true_path), key=extract_number):
            count += 1
            # if count == 1:
            #     continue
            # load data of every single subject, 对于de和inde的设定，所读取的数据有所区别
            data_and_label = None
            subject_name = str(subject).strip('.npy')
            if args.mode == "dependent":
                logger.info(f"Dependent experiment on {count}th subject : {subject_name}")
                data_and_label = load_data_de(true_path, subject)
            elif args.mode == "independent":
                logger.info(f"Independent experiment on {count}th subject : {subject_name}")
                data_and_label = load_data_inde(true_path, subject)
            else:
                raise ValueError("Wrong mode selected.")

            trainer = Trainer(args, subject_name)
            # valAcc, lap_array = trainer.train(data_and_label)
            valAcc = trainer.train(data_and_label)
            acc_list.append(valAcc)
            # lap_array = np.array(lap_array)
            acc_dic[subject_name] = valAcc
            logger.info("Current best acc is : {}".format(acc_dic))
            logger.info("Current average acc is : {}, std is : {}".format(np.mean(acc_list), np.std(acc_list, ddof=1)))
        fullacc = torch.sum(torch.from_numpy(np.asarray(acc_list)).unfold(0, size=15, step=15), dim=0) / len(sessions)
        logger.info("Full best acc is : {}".format(fullacc))
        accdict = {'acc_dic': fullacc, 'average': torch.mean(fullacc), 'std': torch.std(fullacc)}
        np.save(os.path.join('./SEED/', str(args.model) + "ret.npy"), accdict)
        logger.info("Training finished")


def batchTrainKV():
    modelsNlrs = {'HGCNMEmbedPara': 1e-4}
    for model, lr in modelsNlrs.items():
        main(model, lr)


if __name__ == "__main__":
    # batchTrainKV()
    main()
