import os
import os.path as osp
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


def process_cora(data_dir):
    # 1. 读取内容文件 (节点特征和标签)
    content_path = osp.join(data_dir, 'raw', 'cora.content')
    content = pd.read_csv(content_path, sep='\t', header=None)

    # 提取节点ID、特征和标签
    node_ids = content[0].values
    features = content.iloc[:, 1:-1].values.astype(np.float32)
    labels = content.iloc[:, -1].values

    # 创建节点ID到索引的映射
    node_map = {node_id: idx for idx, node_id in enumerate(node_ids)}

    # 2. 读取引用文件 (边信息)
    cites_path = osp.join(data_dir, 'raw', 'cora.cites')
    cites = pd.read_csv(cites_path, sep='\t', header=None)

    # 创建边列表
    edges = []
    for _, row in cites.iterrows():
        src = node_map.get(row[0])
        dst = node_map.get(row[1])
        if src is not None and dst is not None:
            edges.append([src, dst])

    # 转换为PyG需要的格式 [2, num_edges]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # 3. 创建标签映射 (字符串标签 -> 整数)
    unique_labels = sorted(set(labels))
    label_map = {label: i for i, label in enumerate(unique_labels)}
    y = torch.tensor([label_map[label] for label in labels], dtype=torch.long)

    # 4. 创建特征张量
    x = torch.tensor(features, dtype=torch.float)

    # 5. 创建训练/测试掩码 (标准划分)
    num_nodes = len(node_ids)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    # 标准Cora划分: 140训练节点, 500验证节点, 1000测试节点
    train_indices = list(range(140))
    val_indices = list(range(140, 640))
    test_indices = list(range(640, 1640))

    train_mask[train_indices] = True
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    # 6. 创建PyG Data对象
    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )

    # 7. 保存处理后的数据
    processed_dir = osp.join(data_dir, 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    torch.save(data, osp.join(processed_dir, 'data.pt'))

    print(f"Cora数据集处理完成! 保存到: {processed_dir}")
    return data


if __name__ == "__main__":
    # 设置你的数据目录路径
    data_dir = osp.join(osp.dirname(osp.abspath(__file__)), 'data', 'cora')
    data = process_cora(data_dir)

    # 打印数据集信息
    print("\n数据集信息:")
    print(f"节点数: {data.num_nodes}")
    print(f"特征维度: {data.num_features}")
    print(f"边数: {data.num_edges}")
    print(f"类别数: {int(data.y.max().item()) + 1}")
    print(f"训练集节点数: {data.train_mask.sum().item()}")
    print(f"验证集节点数: {data.val_mask.sum().item()}")
    print(f"测试集节点数: {data.test_mask.sum().item()}")