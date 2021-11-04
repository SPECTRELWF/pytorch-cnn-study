# 使用Alexnet实现一个经典的猫狗分类问题。
## 博客：http://liuweifeng.top:8090/

alexnet的结构：
![图片](https://user-images.githubusercontent.com/51198441/140272131-66bbc83e-a56e-49e3-9c8e-cf4ae07df7ec.png)

针对此二分类问题模型在第一层和全连接层和原文有部分修改，详细信息见train.py

训练：将数据集下载后直接解压至Imagenet文件夹下，对应修改一下train里面的文件路径即可运行

评估：在测试集上面做了一个简单的正确率的评估，允许test.py文件即可得到结果

单张图片的测试：
运行predict.py文件，输入要预测的文件路径，得到结果，如下：
![图片](https://user-images.githubusercontent.com/51198441/140273364-07f58ac3-d594-43b7-89ef-8538b86c5320.png)
