# !/usr/bin/python3
# -*- coding:utf-8 -*-
# Author:WeiFeng Liu
# @Time: 2021/11/5 下午3:55
import os
import shutil
train_path = r'/home/lwf/code/pytorch学习/DATASET/CAR/car/val'
save_path = r'/home/lwf/code/pytorch学习/vgg/vgg16-car/dataset/test'
calss = os.listdir(train_path)
for i in range(len(calss)):
    images = os.listdir(os.path.join(train_path,calss[i]))
    for j in range(len(images)):
        src_path = os.path.join(os.path.join(train_path,calss[i]),images[j])
        drc_path = os.path.join(save_path,str(i)+'_'+str(j) +'.jpg')
        shutil.copyfile(src_path,drc_path)
        print(drc_path)