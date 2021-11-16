# !/usr/bin/python3
# -*- coding:utf-8 -*-
# Author:WeiFeng Liu
# @Time: 2021/11/16 上午10:40

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_channels,out_channels,stride=(1,1),downsample=None):
        super(BasicBlock, self).__init__()
        self.Block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,kernel_size=3,stride=stride,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.downsample = downsample

    def forward(self,x):
        out = self.Block(x)
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x
        out += residual
        return out

class ResNet34(nn.Module):
    def __init__(self,num_classes=1000):
        super(ResNet34, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )
        self.layer1 = self.make_layer(64,64,3)
        self.layer2 = self.make_layer(64,128,4,strides=2)
        self.layer3 = self.make_layer(128,256,6,strides=2)
        self.layer4 = self.make_layer(256,512,3,strides=2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(512,256)
        self.fc2 = nn.Linear(256,num_classes)

    def make_layer(self,in_channels,out_channels,block_num,strides=1):

        downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1,stride=strides,bias=False),
            nn.BatchNorm2d(out_channels),
        )
        layer = []
        layer.append(BasicBlock(in_channels,out_channels,strides,downsample))
        for i in range(1,block_num):
            layer.append(BasicBlock(out_channels,out_channels))
        return nn.Sequential(* layer)

    def forward(self,x):
        x = self.head(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
