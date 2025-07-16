#不再关注节点特征，而是边特征，手动创建标签的正例与负例，二分类问题
import torch
import torch.nn.functional as F
import os.path as osp
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling

data_dir = osp.join(osp.dirname(osp.abspath(__file__)), 'data', 'cora')
processed_file = osp.join(data_dir, 'processed', 'data.pt')
if osp.exists(processed_file):
    data= torch.load(processed_file,weights_only=False)
    print("数据集已存在，直接加载")
else:
    raise FileNotFoundError(f"数据集文件不存在: {processed_file}")

class EdgeClassifier(torch.nn.Module):
    def __init__(self,in_channels,out_channels):
        super(EdgeClassifier,self).__init__()
        # 定义GCN卷积层
        self.conv1=GCNConv(in_channels,out_channels)
        # 定义线性分类器
        self.classifier=torch.nn.Linear(2*out_channels,2)
    def forward(self,x,edge_index):
        # 使用GCN卷积层进行卷积操作
        x=F.relu(self.conv1(x,edge_index))
        # 获取正边索引
        pos_edge_index=edge_index
        # 获取负边索引
        totla_edge_index=torch.cat([pos_edge_index,negative_sampling(pos_edge_index,num_neg_samples=pos_edge_index.size(1))],dim=1)
        # 获取边特征
        edge_features=torch.cat([x[totla_edge_index[0]],x[totla_edge_index[1]]],dim=1)
        # 返回分类器输出
        return self.classifier(edge_features)
# 创建训练和测试掩码
edges=data.edge_index.t().cpu().numpy()
num_edges=edges.shape[0]
train_mask=torch.zeros(num_edges,dtype=torch.bool)
test_mask=torch.zeros(num_edges,dtype=torch.bool)
train_size=int(num_edges*0.8)
train_indices=torch.randperm(num_edges)[:train_size]
train_mask[train_indices]=True
test_mask[~train_mask]=True

model=EdgeClassifier(data.num_features,64)
optimizer=torch.optim.Adam(model.parameters(),lr=0.01,weight_decay=5e-4)
def train():
    model.train()
    optimizer.zero_grad()
    logits=model(data.x,data.edge_index)
    pos_edge_index=data.edge_index
    pos_labels=torch.ones(pos_edge_index.size(1),dtype=torch.long)
    neg_edge_index=negative_sampling(pos_edge_index,num_neg_samples=pos_edge_index.size(1))
    neg_labels=torch.zeros(neg_edge_index.size(1),dtype=torch.long)
    labels=torch.cat([pos_labels,neg_labels],dim=0).to(logits.device)
    new_train_mask=torch.cat([train_mask,train_mask],dim=0)
    loss=F.cross_entropy(logits[new_train_mask],labels[new_train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def test():
    model.eval()
    with torch.no_grad():
        logits=model(data.x,data.edge_index)
        pos_edge_index=data.edge_index
        pos_labels=torch.ones(pos_edge_index.size(1),dtype=torch.float)
        neg_edge_index=negative_sampling(pos_edge_index,num_neg_samples=pos_edge_index.size(1))
        neg_labels=torch.zeros(neg_edge_index.size(1),dtype=torch.float)
        labels=torch.cat([pos_labels,neg_labels],dim=0).to(logits.device)
        new_test_mask=torch.cat([test_mask,test_mask],dim=0)
        predictions=logits[new_test_mask].max(1)[1]
        correct=predictions.eq(labels[new_test_mask]).sum().item()
    return correct/len(predictions)

for epoch in range(1,1001):
    loss=train()
    acc=test()
    print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Acc: {acc:.4f}")