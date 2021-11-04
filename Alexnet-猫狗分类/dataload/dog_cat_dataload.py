# !/usr/bin/python3
# -*- coding:utf-8 -*-
# Author:WeiFeng Liu
# @Time: 2021/11/4 下午12:42

import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import os
from PIL import Image

class Dog_Cat_Dataset(data.Dataset):
    def __init__(self , root_path , transform):
        self.path = root_path
        self.transform = transform
        self.images = os.listdir(self.path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self,index):
        image_index = self.images[index]
        img_path = os.path.join(self.path, image_index)
        img = Image.open(img_path).convert("RGB")
        if image_index[0] == 'c':
            label = 0
        else:
            label = 1
        img = self.transform(img)
        label = torch.tensor(label)
        return img, label
