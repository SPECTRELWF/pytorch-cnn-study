# !/usr/bin/python3
# -*- coding:utf-8 -*-
# Author:WeiFeng Liu
# @Time: 2021/11/4 下午12:59


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from vgg16 import VGG16
from utils import plot_curve
from dataload.dog_cat_dataload import Dog_Cat_Dataset
# 定义使用GPU
from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置超参数
epochs = 200
batch_size = 32
lr = 0.01

transform = transforms.Compose([
    transforms.RandomCrop(224,224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5]),
    ])

train_dataset = Dog_Cat_Dataset(r'/home/lwf/code/pytorch学习/vgg/vgg-cat-dag/ImageNet/train',transform=transform)

train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle = True,)
net = VGG16().cuda(device)
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
        print(["epoch:%d , batch:%d , loss:%.3f" %(epoch,batch_idx,1.0*sum_loss/(batch_idx+1))])


    torch.save(net.state_dict(), '/home/lwf/code/pytorch学习/vgg/vgg-cat-dag/model/vgg_cat_dog_model'+str(epoch)+'.pth')
plot_curve(train_loss)