import os
import sys
import json
import torch
import torch.nn as nn
from torchvision import transforms,datasets,utils
import torch.optim as optim
from model import AlexNet
import time

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

data_transform={
    "train":transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]),
    "val":transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
}

image_path=os.path.join("flower_data")
train_dataset=datasets.ImageFolder(root=os.path.join(image_path,"train"),
                                   transform=data_transform["train"])
train_num=len(train_dataset)
#{'daisy=0', 'dandelion=1', 'roses=2', 'sunflowers=3', 'tulips=4'}
flower_list=train_dataset.class_to_idx
cla_dict=dict((val,key) for key,val in flower_list.items())
json_str=json.dumps(cla_dict,indent=4)
with open("class_indices.json","w") as json_file:
    json_file.write(json_str)

batch_size=32
train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,
                                         shuffle=False,num_workers=0)
val_dataset=datasets.ImageFolder(root=os.path.join(image_path,"val"),
                                 transform=data_transform["val"])
val_num=len(val_dataset)
val_loader=torch.utils.data.DataLoader(val_dataset,batch_size=batch_size,
                                       shuffle=False,num_workers=0)

net=AlexNet(num_classes=5,init_weights=True)
net.to(device)
loss_function=nn.CrossEntropyLoss()#多类别损失交叉熵函数
optimizer=optim.Adam(net.parameters(),lr=0.0001)#优化
best_acc=0.0
save_path="./AlexNet.pth"
for epoch in range(10):
    net.train()
    running_loss=0.0
    t1=time.perf_counter()
    for step,data in enumerate(train_loader,start=0):
        images,labels=data
        optimizer.zero_grad()
        outputs=net(images.to(device))
        loss=loss_function(outputs,labels.to(device))
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
        rate=(step+1)/len(train_loader)
        a="*"*int(rate*50)
        b="."*int((1-rate)*50)
        print("\rtrain loss:{:^3.0f}%[{}->{}]{:.3f}".format(int(rate*100),a,b,loss,end=""))
    print()
    print("Time:{:.2f}s".format(time.perf_counter()-t1))
    net.eval()
    acc=0.0
    with torch.no_grad(): #这一步啊，一定不能忘记
        for data_test in val_loader:
            images_test,labels_test=data_test
            outputs=net(images_test.to(device))
            predict_y=torch.max(outputs,dim=1)[1]
            acc+=torch.eq(predict_y,labels_test.to(device)).sum().item()
        accurate_test=acc/val_num
        if accurate_test>best_acc:
            best_acc=accurate_test
            torch.save(net.state_dict(),save_path)
        print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' % (epoch+1,running_loss/step,acc/val_num))

print("Finished Training")