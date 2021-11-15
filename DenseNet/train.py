# !/usr/bin/python3
# -*- coding:utf-8 -*-
# Author:WeiFeng Liu
# @Time: 2021/11/9 下午4:48

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.utils.data as data
from torch.utils.data import DataLoader
from dataload.COVID_Dataload import COVID
from densenet import my_densenet
from torch import nn,optim

transforms = transforms.Compose([
    transforms.Resize([224,224]),
    transforms.RandomHorizontalFlip(),
    # transforms.RandomCrop(224),
    transforms.ToTensor(),

])

batch_size = 32
train_set = COVID(transformer=transforms,train=True)
train_loader = DataLoader(train_set,
                          batch_size = batch_size,
                          shuffle = True,
                          )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#设置超参数
epochs = 200
lr = 1e-4

net = my_densenet().cuda(device)
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
        loss = loss_func(pred, y)
        loss.backward()
        optimizer.step()

        sum_loss += loss.item()
        train_loss.append(loss.item())

        print(["epoch:%d , batch:%d , loss:%.3f" % (epoch, batch_idx,loss.item())])
    torch.save(net.state_dict(),'model/no_pretrain/epoch' + str(epoch+1) + '.pth')
from utils import plot_curve
plot_curve(train_loss)
