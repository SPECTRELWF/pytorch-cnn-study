# !/usr/bin/python3
# -*- coding:utf-8 -*-
# Author:WeiFeng Liu
# @Time: 2021/11/9 下午4:40

import torch
import torchvision
import torch.utils.data as data
import os
import PIL.Image as Image


root_path = r'dataset'
class COVID(data.Dataset):
    def __init__(self,transformer,train):
        super(COVID, self).__init__()
        self.transformer = transformer
        if train:
            self.datapath = root_path + '/train'
            self.images = os.listdir(self.datapath)
        else:
            self.datapath = root_path + '/test'
            self.images = os.listdir(self.datapath)

    def __len__(self):
        return len(self.images)

    def __getitem__(self,index):
        image_index = self.images[index]
        img = Image.open(os.path.join(self.datapath,image_index))
        img = img.convert('RGB')
        img = self.transformer(img)
        # img = torch.tensor(img)
        label = image_index.split('_')[0]
        label = torch.tensor(int(label))

        return img, label



