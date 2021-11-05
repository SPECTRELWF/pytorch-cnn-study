# !/usr/bin/python3
# -*- coding:utf-8 -*-
# Author:WeiFeng Liu
# @Time: 2021/11/4 下午12:59


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from alexnet import AlexNet
from utils import plot_curve
from dataload.cifar10_dataload import CIFAR10_dataset
# 定义使用GPU
from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置超参数
epochs = 50
batch_size = 256
lr = 0.01

transform = transforms.Compose([
    transforms.Resize([32,32]),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5]),
    ])

train_dataset = CIFAR10_dataset(r'/home/lwf/code/pytorch学习/alexnet-CIFAR10/dataset/train',transform=transform)
# print(train_dataset[0])
train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle = True,)
net = AlexNet().cuda(device)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr=lr,momentum=0.9)

train_loss = []
for epoch in range(epochs):
    sum_loss = 0
    for batch_idx,(x,y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)
        pred = net(x)

        optimizer.zero_grad()
        loss = loss_func(pred,y)
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()
        train_loss.append(loss.item())
        print(["epoch:%d , batch:%d , loss:%.3f" %(epoch,batch_idx,loss.item())])
    torch.save(net.state_dict(), '/home/lwf/code/pytorch学习/alexnet-CIFAR10/model/cat_dog_model_epoch' + str(epoch) + '.pth')


plot_curve(train_loss)