# !/usr/bin/python3
# -*- coding:utf-8 -*-
# Author:WeiFeng Liu
# @Time: 2021/11/9 下午4:57

import torchvision
import torch.nn as nn



class my_densenet(nn.Module):
    def __init__(self):
        super(my_densenet, self).__init__()
        self.backbone = torchvision.models.densenet121(pretrained=False)
        self.fc2 = nn.Linear(1000,512)
        self.fc3 = nn.Linear(512,2)

    def forward(self,x):
        x = self.backbone(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x