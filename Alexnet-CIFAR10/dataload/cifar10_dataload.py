# !/usr/bin/python3
# -*- coding:utf-8 -*-
# Author:WeiFeng Liu
# @Time: 2021/11/5 下午12:49

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils import data
from PIL import Image
import os

class CIFAR10_dataset(data.Dataset):
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
        img = self.transform(img)
        label = int(image_index.split('_')[0])
        label = torch.tensor(label)

        return img, label