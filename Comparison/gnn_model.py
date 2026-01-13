import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, TopKPooling, global_mean_pool
from torch_geometric.utils import add_self_loops

class Network(nn.Module):
    def __init__(self, indim=90, ratio=0.8, gnn_hid=128, nclass=1):
        super(Network, self).__init__()

        # 第一层边网络（用于生成边权变换）
        nn1 = nn.Sequential(
            nn.Linear(1, 64),  # 输入边的特征维度（这里为2：例如 abs(weight), maybe coord diff）
            nn.ReLU(),
            nn.Linear(64, indim * gnn_hid)
        )
        self.conv1 = NNConv(indim, gnn_hid, nn1, aggr='mean')
        self.pool1 = TopKPooling(gnn_hid, ratio=ratio)

        # 第二层边网络
        nn2 = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, gnn_hid * gnn_hid)
        )
        self.conv2 = NNConv(gnn_hid, gnn_hid, nn2, aggr='mean')
        self.pool2 = TopKPooling(gnn_hid, ratio=ratio)

        # 全连接层做回归
        self.fc1 = nn.Linear(gnn_hid * 2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, nclass)  # 输出脑龄（标量）

    def forward(self, x, edge_index, batch, edge_attr, pos):
        # 添加自环边
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr=edge_attr, fill_value=1.0)

        # -------- Layer 1 --------
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch, _, _ = self.pool1(x, edge_index, edge_attr, batch=batch)

        x1 = global_mean_pool(x, batch)  # [batch_size, gnn_hid]

        # -------- Layer 2 --------
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch, _, _ = self.pool2(x, edge_index, edge_attr, batch=batch)

        x2 = global_mean_pool(x, batch)  # [batch_size, gnn_hid]

        # -------- Concatenate and Regress --------
        x = torch.cat([x1, x2], dim=1)  # [batch_size, gnn_hid * 2]

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x).squeeze(1)  # [batch_size], 回归预测年龄
        return x
