
import math
import os
import sys
import torch
import numpy as np
import time
import re
import json
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import pandas as pd
from sklearn.model_selection import train_test_split

import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import transforms, utils
from tensorboardX import SummaryWriter

from model2 import LipNet
from dataset import MyDataset
from preprocess import Rescale, RandomCrop, ToTensor,ColorNormalize

if(__name__ == '__main__'):
    opt = __import__('options')
    writer = SummaryWriter()
# 导入数据


def dataset2dataloader(dataset, num_workers=4, shuffle=True):
    return DataLoader(dataset,
                      batch_size=opt.batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      drop_last=False)
# 划分数据集


def split_dataSet(txt_path, test_size=0.2, random_seed=0):
    f = open(txt_path, 'r')
    data = f.readlines()
    train_data, test_data = train_test_split(
        data, test_size=test_size, random_state=random_seed)
    train_file = open('data/hackLip_train.txt', 'w')
    val_file = open('data/hackLip_val.txt', 'w')
    train_file.writelines(train_data)
    val_file.writelines(test_data)
# 绘制帧长分布图


def plt_framelen_distribution():
    len_dict = MyDataset.frame_length_statistics(
        'F:/Video/hack_data/1_train/hack_lip_train')  # opt.video_path
    fig = plt.figure(figsize=(8, 6))
    plt.rcParams['font.sans-serif'] = 'SimHei'  # 设置中文显示
    plt.rcParams['axes.unicode_minus'] = False
    plt.bar(len_dict.keys(), len_dict.values())
    plt.ylabel('数目')
    plt.xlabel('帧长')
    plt.title('帧长分布直方图')
    y_major_locator = MultipleLocator(50)  # 把y轴的刻度间隔设置为50，并存在变量里
    ax = plt.gca()  # ax为两条坐标轴的实例
    ax.yaxis.set_major_locator(y_major_locator)  # 把y轴的主刻度设置为50的倍数
    plt.show()
# 返回学习率的平均值


def show_lr(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return np.array(lr).mean()
# 调用ctc方法？


def ctc_decode(y):
    y = y.argmax(-1)
    return [MyDataset.ctc_arr2txt(y[_]) for _ in range(y.size(0))]
# 测试模型


def test(model, net):
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    composed = transforms.Compose([Rescale(114), RandomCrop((112, 224)),ToTensor(),ColorNormalize()])
    with torch.no_grad():
        # 引入测试集数据
        dataset = MyDataset(opt.video_path, opt.val_list
                                    ,max_frame_len=opt.max_frame_len, transform=composed)
        # 输出测试集大小
        print('num_test_data:{}'.format(len(dataset.data)))
        # ？
        model.eval()
        # 初始化数据加载器
        loader = dataset2dataloader(dataset, shuffle=False)
        loss_list = []
        cer = []
        crit = nn.CTCLoss(blank=0, reduction='mean')
        tic = time.time()
        for (i_iter, input) in enumerate(loader):
            # 存储图片id集、标签集
            # 记录图片数和标签数
            video = input.get('video').to(device)
            txt = input.get('txt').to(device)
            vid_len = input.get('vid_len').to(device)
            txt_len = input.get('txt_len').to(device)

            y = net(video)

            loss = crit(y.transpose(0, 1).log_softmax(-1), txt,
                        vid_len.view(-1), txt_len.view(-1)).detach().cpu().numpy()
            loss_list.append(loss)
            pred_txt = ctc_decode(y)

            truth_txt = [MyDataset.arr2txt(txt[_])
                         for _ in range(txt.size(0))]
            cer.extend(MyDataset.cer(pred_txt, truth_txt))
            if(i_iter % opt.display == 0):
                # 剩余时间
                v = 1.0*(time.time()-tic)/(i_iter+1)
                eta = v * (len(loader)-i_iter) / 3600.0

                print(''.join(101*'-'))
                print('{:<50}|{:>50}'.format('predict', 'truth'))
                print(''.join(101*'-'))
                for (predict, truth) in list(zip(pred_txt, truth_txt))[:10]:
                    print('{:<50}|{:>50}'.format(predict, truth))
                print(''.join(101 * '-'))
                print('test_iter={},eta={},cer={}'.format(
                    i_iter, eta, np.array(cer).mean()))
                print(''.join(101 * '-'))

        return (np.array(loss_list).mean(),  np.array(cer).mean())


def train(model, net):
    composed = transforms.Compose([Rescale(114), RandomCrop((112, 224)),ToTensor(),ColorNormalize()])
    dataset = MyDataset(opt.video_path, opt.train_list 
                                   ,max_frame_len=opt.max_frame_len, transform=composed)

    loader = dataset2dataloader(dataset)
    optimizer = optim.Adam(model.parameters(), 
                           lr=opt.base_lr,
                           weight_decay=0.,
                           amsgrad=True)

    print('num_train_data:{}'.format(len(dataset.data)))
    crit = nn.CTCLoss(blank=0, reduction='mean')
    tic = time.time()

    train_cer = []
    for epoch in range(opt.max_epoch):
        for (i_iter, input) in enumerate(loader):
            model.train()
            video = input.get('video').to(device)
            txt = input.get('txt').to(device)
            vid_len = input.get('vid_len').to(device)
            txt_len = input.get('txt_len').to(device)

            optimizer.zero_grad()
            y = net(video)
            y_trans_log_soft=y.transpose(0, 1).log_softmax(-1)
            loss = crit(y_trans_log_soft, txt,
                        vid_len.view(-1), txt_len.view(-1))
            loss.backward()
            if(opt.is_optimize):
                optimizer.step()

            tot_iter = i_iter + epoch*len(loader)

            pred_txt = ctc_decode(y)

            truth_txt = [MyDataset.arr2txt(txt[_])
                         for _ in range(txt.size(0))]
            train_cer.extend(MyDataset.cer(pred_txt, truth_txt))

            if(tot_iter % opt.display == 0):
                v = 1.0*(time.time()-tic)/(tot_iter+1)
                eta = (len(loader)-i_iter)*v/3600.0

                writer.add_scalar('train loss', loss, tot_iter)
                writer.add_scalar('train cer', np.array(
                    train_cer).mean(), tot_iter)
                print(''.join(101*'-'))
                print('{:<50}|{:>50}'.format('predict', 'truth'))
                print(''.join(101*'-'))

                for (predict, truth) in list(zip(pred_txt, truth_txt))[:3]:
                    print('{:<50}|{:>50}'.format(predict, truth))
                print(''.join(101*'-'))
                print('epoch={},tot_iter={},eta={},loss={},train_cer={}'.format(
                    epoch, tot_iter, eta, loss, np.array(train_cer).mean()))
                print(''.join(101*'-'))

            if(tot_iter % opt.test_step == 0):
                (loss, cer) = test(model, net)
                print('i_iter={},lr={},loss={},cer={}'
                      .format(tot_iter, show_lr(optimizer), loss, cer))
                writer.add_scalar('val loss', loss, tot_iter)
                writer.add_scalar('cer', cer, tot_iter)
                writer.add_graph(model,video)
                savename = '{}_loss_{}_cer_{}.pt'.format(
                    opt.save_prefix, loss, cer)
                (path, name) = os.path.split(savename)
                if(not os.path.exists(path)):
                    os.makedirs(path)
                torch.save(model.state_dict(), savename)
                if(not opt.is_optimize):
                    exit()


if(__name__ == '__main__'):
    print("Loading options...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = LipNet().to(device)
    net = nn.DataParallel(model).to(device)

    '''
    if(hasattr(opt, 'weights')):
        pretrained_dict = torch.load(opt.weights)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}
        missed_params = [k for k, v in model_dict.items() if not k in pretrained_dict.keys()]
        print('loaded params/tot params:{}/{}'.format(len(pretrained_dict),len(model_dict)))
        print('miss matched params:{}'.format(missed_params))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    '''
    torch.manual_seed(opt.random_seed)
    torch.cuda.manual_seed(opt.random_seed)
    torch.cuda.manual_seed_all(opt.random_seed)
    train(model, net)
