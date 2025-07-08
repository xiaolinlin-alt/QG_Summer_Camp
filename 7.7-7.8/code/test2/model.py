import torch.nn as nn
import torch

class AlexNet(nn.Module):
    def __init__(self,num_classes=1000,init_weights=False):
        super(AlexNet,self).__init__()
        self.features=nn.Sequential(
            nn.Conv2d(3,96,11,4,padding=2),
            nn.ReLU(inplace=True),#这里inplace=True表示直接在张量上进行操作但是不创建新的张量
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(96,256,5,1,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(256,384,3,1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384,384,3,1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384,256,3,1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2)
        )
        self.classifier=nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256*6*6,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096,num_classes)
        )
        if init_weights:
            self._initialize_weights()#权重初始化(类内部的方法)
    def forward(self,x):
        x=self.features(x)
        x=torch.flatten(x,start_dim=1)
        x=self.classifier(x)
        return x
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
                elif isinstance(m,nn.Linear):
                    nn.init.normal_(m.weight,0,0.01)
                    nn.init.constant_(m.bias,0)
