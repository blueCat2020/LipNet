# encoding: utf-8
import numpy as np
import glob
import time
import cv2
import os
from torch.utils.data import Dataset
import torch
import glob
import re
import copy
import json
import random
import editdistance
import glob

# 自定义数据集


class MyDataset(Dataset):
    nums = ['-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    def __init__(self, video_path, txt_path, max_frame_len=14, transform=None):
        '''
        video_path:视频路径
        txt_path:文本路径
        transform:数据集类型：一个样本上的可用的可选变换
        '''
        self.video_path = video_path
        self.txt_path = txt_path
        self.max_frame_len = max_frame_len
        self.transform = transform
        self.data = self._load_data(self.txt_path)

    def __getitem__(self, idx):
        # 加载视频对应文件夹名称及对应文本，如：vid=4d068456e623fc7337bf197499fac6a4	txt=262
        (vid, txt) = self.data[idx]
        # 加载视频数据，将文本转化为对应的数字
        vid = self._load_video(vid)
        txt = MyDataset.txt2arr(txt)
        vid_len = vid.shape[0]
        txt_len = txt.shape[0]
        #[C,T,H,W]
        sample = {'video': torch.FloatTensor(vid.transpose(1, 0,2,3)),
                  'txt': torch.LongTensor(txt),
                  'txt_len': txt_len,
                  'vid_len': vid_len}
        return sample

    def __len__(self):
        return len(self.data)

    def _load_data(self, txt_path, max_len=18, min_len=5):
        data = []
        with open(txt_path, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                video = line.split()
                video_path = os.path.join(self.video_path, video[0], '*.png')
                path_file_number = glob.glob(video_path)
                len_video = len(path_file_number)
                # 进行筛选
                if len_video <= max_len and len_video >= min_len:
                    data.append(video)
        return data

    def _load_video(self, vid):
        vid_path = os.path.join(self.video_path, vid)
        files = os.listdir(vid_path)
        files = list(filter(lambda file: file.find('.png') != -1, files))
        files = sorted(files, key=lambda file: int(os.path.splitext(file)[0]))
        array = [cv2.imread(os.path.join(vid_path, file))
                 for file in files]  # 加载图片
        array = list(filter(lambda im: not im is None, array))  # 过滤空图片
        # 应用变换
        if self.transform:
            for i in range(len(array)):
                array[i] = self.transform(array[i])

        array = self._deal_video(array, keep_len=self.max_frame_len)
        array = np.stack(array, axis=0).astype(np.float32)
        return array
    # 处理唇读图片序列，使其长度一致

    def _deal_video(self, img_array, keep_len):
        img_array = list(img_array)
        arr_len = len(img_array)
        if arr_len > keep_len:
            for i in range(arr_len-keep_len):
                del_index = np.random.randint(0, arr_len-i)
                img_array.pop(del_index)
        elif arr_len < keep_len:
            for i in range(keep_len-arr_len):
                add_index = np.random.randint(0, arr_len+i)
                img_array.insert(add_index, img_array[add_index])
        return img_array
    # 对指定文件夹内各子文件夹内图片数量。root_path：文件根目录

    @staticmethod
    def frame_length_statistics(root_path):
        len_dict = {}
        video_files = os.listdir(root_path)
        for file in list(video_files):
            video_path = os.path.join(root_path, file, '*.png')
            path_file_number = glob.glob(video_path)
            len_video = len(path_file_number)
            len_dict.setdefault(len_video, 0)
            len_dict[len_video] += 1
        return len_dict

    @staticmethod
    def txt2arr(txt):
        arr = []
        for c in list(txt):
            arr.append(MyDataset.nums.index(c))
        return np.array(arr)

    @staticmethod
    def arr2txt(arr):
        txt = []
        for n in arr:
            txt.append(MyDataset.nums[n])
        return ''.join(txt).strip()

    @staticmethod
    def ctc_arr2txt(arr):
        pre = -1
        txt = []
        for n in arr:
            if(pre != n):
                if(len(txt) > 0 and txt[-1] == '-' and MyDataset.nums[n] == '-'):
                    pass
                else:
                    txt.append(MyDataset.nums[n])
            pre = n
        # result=''.join(txt).strip()
        return ''.join(txt).replace('-', '')
        #return ''.join(txt).strip()
    @staticmethod
    def wer(predict, truth):
        # 词评价标准
        word_pairs = [(p[0].split(' '), p[1].split(' '))
                      for p in zip(predict, truth)]
        wer = [1.0*editdistance.eval(p[0], p[1])/len(p[1]) for p in word_pairs]
        return wer

    @staticmethod
    def cer(predict, truth):
        # 句评价标准
        cer = [1.0*editdistance.eval(p[0], p[1])/len(p[1])
               for p in zip(predict, truth)]
        return cer
