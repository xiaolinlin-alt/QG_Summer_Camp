import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
import os
import pandas as pd
from PIL import Image
from torchvision import datasets,transforms

#设置随机种子
torch.manual_seed(12046)

class CassavaDataset(Dataset):
    def __init__(self,root_dir,csv_file,transform=None):
        self.root_dir=root_dir
        self.annotations=pd.read_csv(csv_file)
        self.transform=transform

    def __len__(self):
        return len(self.annotations)
    def __getitem__(self,index):
        img_name=os.path.join(self.root_dir,self.annotations.iloc[index,0])
        image=Image.open(img_name).convert("RGB")
        label=self.annotations.iloc[index,1]
        if self.transform:
            image=self.transform(image)
        return image,torch.tensor(label,dtype=torch.long)

base_dir="cassava-leaf-disease-classification"
train_img_dir=os.path.join(base_dir,"train_images")
train_csv=os.path.join(base_dir,"train.csv")

#数据转换
data_transform={
    "train":transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1,contrast=0.1,saturation=0.1,hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]),
    "val":transforms.Compose([
        transforms.Resize((384,384)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]),
}

full_dataset=CassavaDataset(root_dir=train_img_dir,csv_file=train_csv,transform=data_transform["train"])
train_size=int(0.8*len(full_dataset))
val_size=len(full_dataset)-train_size
train_set,val_set=torch.utils.data.random_split(full_dataset,[train_size,val_size])
val_set.dataset.transform=data_transform["val"]

# 残差块
class ResidualBlock(nn.Module):
    def __init__(self,in_channel,out_channel,stride=1):
        super().__init__()
        self.conv1=nn.Conv2d(in_channel,out_channel,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(out_channel)
        self.conv2=nn.Conv2d(out_channel,out_channel,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(out_channel)
        #下采样模块
        self.downsample=nn.Sequential()
        if stride!=1 or in_channel!=out_channel:
            self.downsample=nn.Sequential(
                nn.Conv2d(in_channel,out_channel,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        identity=self.downsample(x)
        out=F.relu(self.bn1(self.conv1(x)))
        out=self.bn2(self.conv2(out))
        out+=identity
        out=F.relu(out)
        return out

class Cassava_ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        #初始层
        self.conv1=nn.Conv2d(3, 64, kernel_size=3,stride=1,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(64)

        # 残差块组4个阶段
        self.layer1=self._make_layer(64,64,stride=1)
        self.layer2=self._make_layer(64,128,stride=2)
        self.layer3=self._make_layer(128,256,stride=2)
        self.layer4=self._make_layer(256,512,stride=2)

        # 分类头
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(512,5)

    def _make_layer(self,in_channels,out_channels,stride):
        return nn.Sequential(
            ResidualBlock(in_channels,out_channels,stride),
            ResidualBlock(out_channels,out_channels,stride=1)
        )

    def forward(self,x):
        x=F.relu(self.bn1(self.conv1(x)))  #[B,64,32,32]
        x=self.layer1(x)  #[B,64,32,32]
        x=self.layer2(x)  #[B,128,16,16]
        x=self.layer3(x)  # [B,256,8,8]
        x=self.layer4(x)  # [B,512,4,4]
        x =self.avgpool(x)  #[B,512,1,1]
        x=torch.flatten(x,1)  #[B,512]
        x=self.fc(x)  #[B,5]
        return x
# 训练工具函数
def estimate_loss(model,data_loader,eval_iters=10):
    model.eval()
    losses=[]
    accuracies=[]
    for i,(inputs,labels) in enumerate(data_loader):
        if i>=eval_iters: break
        inputs,labels=inputs.to(device),labels.to(device)
        with torch.no_grad():
            logits=model(inputs)
            loss=F.cross_entropy(logits,labels)
            _,predicted=torch.max(logits,1)
            acc=(predicted==labels).float().mean()
        losses.append(loss.item())
        accuracies.append(acc.item())
    model.train()
    return {
        'loss': sum(losses)/len(losses),
        'accuracy': sum(accuracies)/len(accuracies)
    }

def train(model,optimizer,epochs=10):
    for epoch in range(epochs):
        model.train()
        for inputs,labels in train_loader:
            inputs,labels=inputs.to(device),labels.to(device)
            optimizer.zero_grad()
            logits=model(inputs)
            loss=F.cross_entropy(logits,labels)
            loss.backward()
            optimizer.step()
        # 评估
        train_stats=estimate_loss(model,train_loader)
        val_stats =estimate_loss(model,val_loader)
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_stats['loss']:.4f} | Val Loss: {val_stats['loss']:.4f}")
        print(f"Train Acc: {train_stats['accuracy']:.4f} | Val Acc: {val_stats['accuracy']:.4f}")
        print("-"*50)


if __name__=="__main__":
    #训练模型
    batch_size=8
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model=Cassava_ResNet().to(device)
    optimizer=optim.Adam(model.parameters(),lr=0.001,weight_decay=1e-4)

    train_loader=DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=0,pin_memory=True)
    val_loader=DataLoader(val_set,batch_size=batch_size,shuffle=False,num_workers=0,pin_memory=True)
    # 开始训练
    train(model,optimizer,epochs=20)
    torch.save(model.state_dict(),"cassava_resnet.pth")