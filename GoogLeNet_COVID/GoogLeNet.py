# !/usr/bin/python3
# -*- coding:utf-8 -*-
# Author:WeiFeng Liu
# @Time: 2021/11/9 下午4:57
import torch
import torchvision
import torch.nn as nn



class my_Googlenet(nn.Module):
    def __init__(self):
        super(my_Googlenet, self).__init__()
        self.backbone = torchvision.models.inception_v3(pretrained=True)
        self.fc2 = nn.Linear(1000,512)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512,2)

    def forward(self,x):
        feature = self.backbone(x)
        x = feature[0]
        # print(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

