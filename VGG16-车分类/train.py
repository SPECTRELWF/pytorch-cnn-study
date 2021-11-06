# !/usr/bin/python3
# -*- coding:utf-8 -*-
# Author:WeiFeng Liu
# @Time: 2021/11/5 下午4:37

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms as transforms
import torch.optim as optim
from dataload.car_dataload import CAR_DATASET
from vgg16 import VGG16
import torch.nn as nn
from utils import plot_curve

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = 200
batch_size = 32
lr = 0.01
transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])
train_dataset = CAR_DATASET(r'dataset/train', transform=transform)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
)


model = VGG16().to(device)
opt = optim.SGD(model.parameters(),lr=lr,momentum=0.9)
cri = nn.CrossEntropyLoss()

train_loss = []
for epoch in range(epochs):
    sum_loss = 0
    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)
        pred = model(x)

        opt.zero_grad()

        loss = cri(pred, y)
        loss.backward()
        opt.step()
        train_loss.append(loss.item())

        print('[epoch : %d  ,batch : %d  ,loss : %.3f]' %(epoch,batch_idx,loss.item()))
    torch.save(model.state_dict(), 'model/new/epoch'+str(epoch)+'.pth')
plot_curve(train_loss)


