# !/usr/bin/python3
# -*- coding:utf-8 -*-
# Author:WeiFeng Liu
# @Time: 2021/11/2 下午3:38

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from alexnet import AlexNet
from utils import plot_curve

# 定义使用GPU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置超参数
epochs = 30
batch_size = 256
lr = 0.01

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   # 数据归一化
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=256, shuffle=False
)

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 定义网络
net = AlexNet().to(device)

# 定义优化器
optimzer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

# train
train_loss = []
for epoch in range(epochs):
    sum_loss = 0.0
    for batch_idx, (x, y) in enumerate(train_loader):
        print(x.shape)
        x = x.to(device)
        y = y.to(device)

        # 梯度清零
        optimzer.zero_grad()

        pred = net(x)
        loss = criterion(pred, y)
        loss.backward()
        optimzer.step()
        train_loss.append(loss.item())

        sum_loss += loss.item()

        if batch_idx % 100 == 99:
            print('[%d, %d] loss: %.03f'
                  % (epoch + 1, batch_idx + 1, sum_loss / 100))
            sum_loss = 0.0
torch.save(net.state_dict(), '/home/lwf/code/pytorch学习/alexnet图像分类/model/model.pth')
plot_curve(train_loss)
