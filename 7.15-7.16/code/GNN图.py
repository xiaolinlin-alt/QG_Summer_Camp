#六分类任务，预测蛋白酶属于哪一个类别
import torch
import torch.nn.functional as F
import os.path as osp
from torch_geometric.nn import GCNConv,global_mean_pool
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.loader import DataLoader

data_dir=osp.join(osp.dirname(__file__), 'data', 'EMZYMES')
processed_file=osp.join(data_dir,"processed","data.pt")
if osp.exists(processed_file):
    dataset=torch.load(processed_file,weights_only=False)
    print("数据集已存在，直接加载")
else:
    raise FileNotFoundError("数据集不存在")

train_dataset=dataset[:540]
test_dataset=dataset[540:]
train_loader=DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader=DataLoader(test_dataset, batch_size=64, shuffle=False)

class GCN(torch.nn.Module):
    def __init__(self,hidden_channels,num_classes,num_node_features):
        super(GCN, self).__init__()
        self.num_node_features=num_node_features
        self.num_classes=num_classes
        self.conv1=GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2=GCNConv(hidden_channels,hidden_channels)
        self.conv3=GCNConv(hidden_channels,hidden_channels)
        self.lin=torch.nn.Linear(hidden_channels, dataset.num_classes)
    def forward(self,x, edge_index,batch):
        x=self.conv1(x, edge_index)
        x=F.relu(x)
        x=self.conv2(x, edge_index)
        x=F.relu(x)
        x=self.conv3(x, edge_index)
        x=global_mean_pool(x,batch)
        x=F.dropout(x,p=0.5,training=self.training)
        x=self.lin(x)
        return x
model=GCN(hidden_channels=64,num_classes=6,num_node_features=dataset.num_node_features)
optimizer=torch.optim.Adam(model.parameters(), lr=0.01)
criterion=torch.nn.CrossEntropyLoss()

def train():
    model.train()
    for data in train_loader:
        optimizer.zero_grad()
        out=model(data.x,data.edge_index,data.batch)
        loss=criterion(out,data.y)
        loss.backward()
        optimizer.step()

def test():
    model.eval()
    correct=0
    for data in test_loader:
        out=model(data.x,data.edge_index,data.batch)
        pred=out.max(dim=1)[1]
        correct+=pred.eq(data.y).sum().item()
        return correct/len(test_dataset)

for epoch in range(1,1001):
    train()
    test_acc=test()
    print("Epoch: {:03d}, Test: {:.4f}".format(epoch, test_acc))