import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torchvision.transforms as transforms
import torch.optim as optim

transform=transforms.Compose([#数据预处理
    transforms.ToTensor(),#变tensor
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))#标准化mean,std
    ])

trainset=torchvision.datasets.CIFAR10(root='./data',train=True,
                                        download=False,transform=transform)#这个transform就是前面对图像进行预处理的函数
trainloader=torch.utils.data.DataLoader(trainset,batch_size=4,
                                 shuffle=True,num_workers=0)

testset=torchvision.datasets.CIFAR10(root='./data',train=False,
                                     download=False,transform=transform)
testloader=torch.utils.data.DataLoader(testset,batch_size=4,
                                     shuffle=False,num_workers=0)#shuffle=False表示不洗牌，按顺序取
testdata_iter=iter(testloader)
test_image,test_label=next(testdata_iter)

classes=('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

def imshow(img):
    img=img/2+0.5#反标准化
    npimg=img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

imshow(torchvision.utils.make_grid(test_image))
print(' '.join('%5s' % classes[test_label[j]] for j in range(4)))

net=LeNet()
loss_function=nn.CrossEntropyLoss()
optimizer=optim.Adam(net.parameters(),lr=0.001)

for epoch in range(5):
    running_loss=0.0
    for step,data in enumerate(trainloader,start=0):
        inputs,labels=data
        optimizer.zero_grad()
        outputs=net(inputs)
        loss=loss_function(outputs,labels)
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()
        if step%500==499:
            with torch.no_grad():
                outputs=net(test_image)
                predict_y=torch.max(outputs,dim=1)[1]
                accuracy=(predict_y==test_label).sum().item()/test_label.size(0)
                print('[%d,%5d] loss:%.3f accuracy:%.3f' %(epoch+1,step+1,running_loss/500,accuracy))
                running_loss=0.0
    print('Finished Training')
    save_path='./lenet5.pth'
    torch.save(net.state_dict(),save_path)