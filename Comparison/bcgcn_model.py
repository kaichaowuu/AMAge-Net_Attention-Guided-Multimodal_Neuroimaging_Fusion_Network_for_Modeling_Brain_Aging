import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class GATModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4, dropout_rate=0.1):
        super(GATModel, self).__init__()

        self.gat1 = GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout_rate)
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout_rate, concat=True)
        self.gat_out = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, concat=False, dropout=dropout_rate)

        # 计算节点重要性的注意力池化层
        self.att_pool = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.regressor = nn.Linear(hidden_dim, output_dim)

        # 用于保存节点的注意力权重
        self.node_attention = None

    def forward(self, x, edge_index, batch):
        # GAT层前向传播
        x = F.elu(self.gat1(x, edge_index))  # [num_nodes, hidden_dim * heads]
        x = F.elu(self.gat2(x, edge_index))  # [num_nodes, hidden_dim * heads]
        x = F.elu(self.gat_out(x, edge_index))  # [num_nodes, hidden_dim]

        # 保存节点嵌入
        self.node_embeds = x.detach()  # 不参与梯度传播

        # 计算节点注意力分数
        attn_score = self.att_pool(x)  # [num_nodes, 1]
        self.node_attention = attn_score.squeeze()  # 保存为 [num_nodes]

        # 用注意力分数加权节点特征
        x = x * attn_score  # [num_nodes, hidden_dim]

        # 通过全局平均池化得到图级别特征
        x = global_mean_pool(x, batch)  # [batch_size, hidden_dim]

        # 预测输出
        out = self.regressor(x)  # [batch_size, output_dim]
        return out

    def get_node_importance(self):
        """
        返回当前 batch 中所有节点的注意力权重，代表脑区重要性。
        必须在 forward() 之后调用。
        """
        if self.node_attention is None:
            raise ValueError("node_attention is None. Run forward() before calling get_node_importance().")
        return self.node_attention


# class GATModel(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_heads=4, dropout_rate=0.1):
#         super(GATModel, self).__init__()
#
#         self.gat1 = GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout_rate)
#         self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout_rate, concat=True)
#         self.gat_out = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, concat=False, dropout=dropout_rate)
#
#         # 计算节点重要性的注意力池化层
#         self.att_pool = nn.Sequential(
#             nn.Linear(hidden_dim, 1),
#             nn.Sigmoid()
#         )
#
#         # 用于保存节点的注意力权重
#         self.node_attention = None
#
#     def forward(self, x, edge_index, batch):
#         # GAT层前向传播
#         x = F.elu(self.gat1(x, edge_index))  # [num_nodes, hidden_dim * heads]
#         x = F.elu(self.gat2(x, edge_index))  # [num_nodes, hidden_dim * heads]
#         x = F.elu(self.gat_out(x, edge_index))  # [num_nodes, hidden_dim]
#
#         # 保存节点嵌入
#         self.node_embeds = x.detach()  # 不参与梯度传播
#
#         # 计算节点注意力分数
#         attn_score = self.att_pool(x)  # [num_nodes, 1]
#         self.node_attention = attn_score.squeeze()  # 保存为 [num_nodes]
#
#         # 用注意力分数加权节点特征
#         x = x * attn_score  # [num_nodes, hidden_dim]
#
#         # 通过全局平均池化得到图级别特征
#         x = global_mean_pool(x, batch)  # [batch_size, hidden_dim]
#
#         return x
#
#     def get_node_importance(self):
#         """
#         返回当前 batch 中所有节点的注意力权重，代表脑区重要性。
#         必须在 forward() 之后调用。
#         """
#         if self.node_attention is None:
#             raise ValueError("node_attention is None. Run forward() before calling get_node_importance().")
#         return self.node_attention
