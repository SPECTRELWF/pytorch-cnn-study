# !/usr/bin/python3
# -*- coding:utf-8 -*-
# Author:WeiFeng Liu
# @Time: 2021/11/5 下午4:11

import torch
import torchvision
from torchvision import transforms as transforms
from PIL import Image
import os
from torch.utils import data
class CAR_DATASET(data.Dataset):
    def __init__(self,root,transform):
        self.path = root
        self.transform = transform
        self.images = os.listdir(self.path)
    def __len__(self):
        return len(self.images)
    def __getitem__(self,index):
        image_index = self.images[index]
        img_path = os.path.join(self.path,image_index)
        img = Image.open(img_path).convert('RGB')
        # print(img)
        label = image_index.split('_')[0]

        img = self.transform(img)
        label = torch.tensor(int(label))

        return img, label