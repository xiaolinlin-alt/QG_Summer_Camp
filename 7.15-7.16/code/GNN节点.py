#节点分类任务
import torch
import torch.nn as nn
import torch.nn.functional as F
import os.path as osp
from torch_geometric.nn import GCNConv

data_dir = osp.join(osp.dirname(osp.abspath(__file__)), 'data', 'cora')
processed_file = osp.join(data_dir, 'processed', 'data.pt')
if osp.exists(processed_file):
    data= torch.load(processed_file,weights_only=False)
    print("数据集已存在，直接加载")
else:
    raise FileNotFoundError(f"数据集文件不存在: {processed_file}")

"""
先尝试直接下载，下载不了的就选择上面那个方法（你怎么知道就是我下载不了~）
dataset=Planetoid(root="./data",name='Cora')
data=dataset[0]"""
#数据集2708个节点，1433维特征，边数是5429，共有七个标签，7分类问题

class GCN(nn.Module):
    def __init__(self,num_features,num_classes):
        super(GCN,self).__init__()
        self.conv1=GCNConv(num_features,16)
        self.conv2=GCNConv(16,num_classes)

    def forward(self,x,edge_index):
        x=self.conv1(x,edge_index)
        x=F.relu(x)
        x=self.conv2(x,edge_index)
        return F.log_softmax(x,dim=1)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
num_features=data.num_features
num_classes=int(data.y.max().item())+1
model=GCN(num_features,num_classes).to(device)
data=data.to(device)
optimizer=torch.optim.Adam(model.parameters(),lr=0.01,weight_decay=5e-4)
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out=model(data.x,data.edge_index)
    loss=F.nll_loss(out[data.train_mask],data.y[data.train_mask])
    loss.backward()
    optimizer.step()
model.eval()
test_predict=model(data.x,data.edge_index)[data.test_mask]
max_index=torch.argmax(test_predict,dim=1)
test_true=data.y[data.test_mask]
correct=0
for i in range(len(max_index)):
    if max_index[i]==test_true[i]:
        correct+=1
print("准确率：{}".format(correct/len(max_index)))