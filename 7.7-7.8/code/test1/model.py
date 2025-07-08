import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.conv1=nn.Conv2d(3,16,5)#输入的是一张彩色图片，所以第一个深度是3；采用了16个卷积核，每个卷积核尺度是5*5
        self.pool1=nn.MaxPool2d(2,2)#卷积核大小，尺寸
        self.conv2=nn.Conv2d(16,32,5)#通过第一个卷积变成了16，采用32个卷积核，尺寸5*5
        self.pool2=nn.MaxPool2d(2,2)
        self.fc1=nn.Linear(32*5*5,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)#输出需要根据训练集来进行修改，这个数据集分10类

    def forward(self,x):
        x=F.relu(self.conv1(x))  #input(3,32,32),output(16,28,28)
        x=self.pool1(x)          #input(16,28,28),output(16,14,14)
        x=F.relu(self.conv2(x))  #input(16,14,14),output(32,10,10)
        x=self.pool2(x)          #input(32,10,10),output(32,5,5)
        x=x.view(-1,32*5*5)
        x=F.relu(self.fc1(x))    #input(32*5*5),output(120)
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x

import torch
input1=torch.rand([32,3,32,32])#分别是batch,channel,height,width
model=LeNet()
print(model)
output=model(input1)