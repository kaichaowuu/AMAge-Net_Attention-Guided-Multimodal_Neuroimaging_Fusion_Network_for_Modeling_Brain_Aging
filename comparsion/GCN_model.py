import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        layers = []
        if num_layers == 1:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class GraphCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_gc_layers=2, mlp_layers=2):
        super(GraphCNN, self).__init__()
        assert num_gc_layers >= 1, "num_gc_layers should be >= 1"

        # GCN卷积层列表
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_gc_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # MLP回归层
        self.mlp = MLP(mlp_layers, hidden_dim, hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        # 多层GCN卷积+ReLU
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)

        # 全图池化，把所有节点的特征池化成图的整体特征向量
        x = global_mean_pool(x, batch)

        # MLP回归预测
        out = self.mlp(x)
        return out.view(-1)  # 返回(batch_size,)的一维张量
