# !/usr/bin/python3
# -*- coding:utf-8 -*-
# Author:WeiFeng Liu
# @Time: 2021/11/4 下午1:29

import torch
import torchvision
from dataload.dog_cat_dataload import Dog_Cat_Dataset
# 定义使用GPU
from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torchvision.transforms as transforms
from alexnet import AlexNet
transform = transforms.Compose([
    transforms.Resize([224,224]),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5]),
    ])

test_dataset = Dog_Cat_Dataset('/home/lwf/code/pytorch学习/alexnet-cat-dag/ImageNet/test',transform=transform)
test_loader = DataLoader(test_dataset,
                         batch_size = 32,
                         shuffle = False,
                         )




def predict():
    net = AlexNet().to(device)
    net.load_state_dict(torch.load('/home/lwf/code/pytorch学习/alexnet-cat-dag/model/cat_dog_model.pth'))
    print(net)
    total_correct = 0
    for batch_idx, (x, y) in enumerate(test_loader):
        # x = x.view(x.size(0),28*28)
        # x = x.view(256,28,28)
        x = x.to(device)
        print(x.shape)
        y = y.to(device)
        print('y',y)
        out = net(x)
        # print(out)
        pred = out.argmax(dim=1)
        print('pred',pred)
        correct = pred.eq(y).sum().float().item()
        total_correct += correct
    total_num = len(test_loader.dataset)

    acc = total_correct / total_num
    print("test acc:", acc)


predict()
