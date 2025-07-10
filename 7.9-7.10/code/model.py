import torch
import torch.nn as nn

class ConvBNReLU(nn.Sequential):
    def __init__(self,in_channel,out_channel,kernel_size=3,stride=1,groups=1):#groups=1的话就是普通的卷积，等于输入特征矩阵的深度，就是DW的卷积
        padding=(kernel_size-1)//2
        super(ConvBNReLU,self).init__(
            nn.Coonv2d(in_channel,out_channel,kernel_size,stride,padding,groups=groups,bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )

class InvertedResidual(nn.Module):#倒残差模块
    def __init__(self,in_channel,out_channel,stride,expand_ratio):
        super(InvertedResidual).__init__()
        hidden_channel=in_channel*expand_ratio#hidden_channel是中间的通道数，拓展因子啊
        self.use_shortcut=stride=1 and in_channel==out_channel

        layers=[]
        if expand_ratio!=1:
            layers.append(ConvBNReLU(in_channel,hidden_channel,kernel_size=1))
        layers.extend([#一次性批量插入
            ConvBNReLU(hidden_channel,hidden_channel,stride=stride,groups=hidden_channel),
            nn.Conv2d(hidden_channel,out_channel,kernel_size=1,bias=False),
            nn.BatchNorm2d(out_channel)
        ])
        self.conv=nn.Sequential(*layers)
    def forward(self,x):
        if self.use_shortcut:
            return x+self.conv(x)
        else:
            return self.conv(x)

def _make_divisible(ch,divisor=8,min_ch=None):#更好地调用硬件设备
    if min_ch is None:
        min_ch=divisor
    new_ch=max(min_ch,int(ch+divisor/2)//divisor*divisor)
    if new_ch<0.9*ch:
        new_ch+=divisor
    return new_ch

class MobileNetV2(nn.Module):
    def __init__(self,num_classes=1000,alpha=1.0,round_nearest=8):
        super(MobileNetV2, self).__init__()
        block=InvertedResidual
        input_channel=_make_divisible(32*alpha,round_nearest)
        last_channel=_make_divisible(1280*alpha,round_nearest)
        inverted_residual_setting=[
            [1,16,1,1],
            [6,24,2,2],
            [6,32,3,2],
            [6,64,4,2],
            [6,96,3,1],
            [6,160,3,2],
            [6,320,1,1],
        ]
        features=[]
        features.append(ConvBNReLU(3,input_channel,stride=2))

        for t,c,n,s in inverted_residual_setting:
            output_channel=_make_divisible(c*alpha,round_nearest)
            for i in range(n):
                stride=2 if s==1 and i==0 else 1
                features.append(block(input_channel,output_channel,stride=stride,expand_ratio=t))
                input_channel=output_channel
            features.append(ConvBNReLU(input_channel,last_channel,kernel_size=1))
            self.feature=nn.Sequential(*features)
            self.avgpool=nn.AdaptiveAvgPool2d((1,1))
            self.classifier=nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(last_channel,num_classes)
            )
            for m in self.modules():
                if isinstance(m,nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight,mode='fan_out')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m,nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m,nn.Linear):
                    nn.init.normal_(m.weight,0,0.01)
                    nn.init.zeros_(m.bias)
    def forward(self,x):
        x=self.features(x)
        x=self.avgpool(x)
        x=torch.flatten(x,1)
        x=self.classifier(x)
        return x