import torch
from torch import nn
from gcn_model import GATModel
from monai.networks.nets import DenseNet121


class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4, dropout=0):
        super(CrossAttentionBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, query, key, value):
        attn_output, _ = self.attn(query, key, value)
        x = self.norm1(query + self.dropout1(attn_output))
        ffn_output = self.ffn(x)
        out = self.norm2(x + self.dropout2(ffn_output))
        return out  # shape: [B, 1, 64]


class GatedFusion(nn.Module):
    def __init__(self, embed_dim=64):
        super(GatedFusion, self).__init__()
        self.gate_layer = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, feat1, feat2):
        concat = torch.cat([feat1, feat2], dim=-1)  # [B, 1, 128]
        gate = self.gate_layer(concat)              # [B, 1, 1]
        fused = gate * feat1 + (1 - gate) * feat2
        return fused

class FusionModel(nn.Module):
    def __init__(self, gcn_input_dim=90, gcn_hidden=64, gcn_out=32, num_heads=4, dropout_rate=0):
        super(FusionModel, self).__init__()

        self.backbone = DenseNet121(spatial_dims=3, in_channels=1, out_channels=64)
        self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-1])
        self.patch_proj = nn.Linear(1024, 64)

        self.gcn = GATModel(
            input_dim=gcn_input_dim, hidden_dim=gcn_hidden,
            output_dim=gcn_out, num_heads=num_heads, dropout_rate=dropout_rate
        )
        self.gcn_proj = nn.Linear(gcn_out, 64)

        self.cross_attn_block = CrossAttentionBlock(embed_dim=64, num_heads=num_heads, dropout=dropout_rate)
        # self.gated_fusion = GatedFusion(embed_dim=64)

        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # 保存节点嵌入和节点重要性（注意力权重）
        self.gcn_node_embeds = None
        self.gcn_node_importance_per_sample = None  # List of tensors [90] per sample

    def forward(self, data):
        img = data.img
        if img.dim() == 4:
            img = img.unsqueeze(1)  # [B, 1, D, H, W]

        # img = img.clone().detach().requires_grad_(True)
        img.requires_grad_(True)
        img.retain_grad()
        data.img = img

        # CNN部分
        cnn_feat_map = self.feature_extractor(img)  # [B, 64, d, h, w]
        cnn_feat_map = cnn_feat_map.flatten(2).transpose(1, 2)  # [B, N, 64]
        cnn_feat_tokens = self.patch_proj(cnn_feat_map)  # [B, N, 64]

        # GCN部分
        gcn_feat = self.gcn(data.x, data.edge_index, data.batch)  # [B, gcn_out]
        gcn_feat_proj = self.gcn_proj(gcn_feat).unsqueeze(1)      # [B, 1, 64]

        # 保存节点嵌入
        self.gcn_node_embeds = self.gcn.node_embeds.detach() if self.gcn.node_embeds is not None else None

        # 取出所有节点注意力权重（[total_nodes]）
        node_attention = self.gcn.get_node_importance().detach()  # [total_num_nodes]

        # 按样本分组，样本数 = batch_size
        batch_size = data.batch.max().item() + 1
        nodes_per_sample = 90

        # 确保总节点数能整除每个样本节点数
        assert node_attention.shape[0] == batch_size * nodes_per_sample, \
            f"节点数不匹配：{node_attention.shape[0]} != {batch_size}*{nodes_per_sample}"

        # 按样本切分注意力权重，形状为 list，元素是 [90]
        self.gcn_node_importance_per_sample = torch.split(node_attention, nodes_per_sample)

        # Cross Attention 和融合
        cross_attended = self.cross_attn_block(query=gcn_feat_proj, key=cnn_feat_tokens, value=cnn_feat_tokens)  # [B,1,64]
        # global_cnn_token = torch.mean(cnn_feat_tokens, dim=1, keepdim=True)  # [B,1,64]
        # fused = self.gated_fusion(cross_attended, global_cnn_token)  # [B,1,64]
        #
        # out = self.fc(fused.squeeze(1)).squeeze(1)  # [B]

        # 不使用Gated Fusion，直接用 cross_attended
        out = self.fc(cross_attended.squeeze(1)).squeeze(1)  # [B]

        return out

# class FusionModel(nn.Module):
#     def __init__(self, gcn_input_dim=90, gcn_hidden=64, num_heads=4, dropout_rate=0):
#         super(FusionModel, self).__init__()
#
#         self.backbone = DenseNet121(spatial_dims=3, in_channels=1, out_channels=64)
#         self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-1])
#         self.patch_proj = nn.Linear(1024, 64)
#
#         self.gcn = GATModel(
#             input_dim=gcn_input_dim, hidden_dim=gcn_hidden, num_heads=num_heads, dropout_rate=dropout_rate
#         )
#         self.gcn_proj = nn.Linear(gcn_hidden, 64)
#
#         self.cross_attn_block = CrossAttentionBlock(embed_dim=64, num_heads=num_heads, dropout=dropout_rate)
#         self.gated_fusion = GatedFusion(embed_dim=64)
#
#         self.fc = nn.Sequential(
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.Linear(32, 1)
#         )
#
#         # 保存节点嵌入和节点重要性（注意力权重）
#         self.gcn_node_embeds = None
#         self.gcn_node_importance_per_sample = None  # List of tensors [90] per sample
#
#     def forward(self, data):
#         img = data.img
#         if img.dim() == 4:
#             img = img.unsqueeze(1)  # [B, 1, D, H, W]
#
#         # img = img.clone().detach().requires_grad_(True)
#         img.requires_grad_(True)
#         img.retain_grad()
#         data.img = img
#
#         # CNN部分
#         cnn_feat_map = self.feature_extractor(img)  # [B, 64, d, h, w]
#         cnn_feat_map = cnn_feat_map.flatten(2).transpose(1, 2)  # [B, N, 64]
#         cnn_feat_tokens = self.patch_proj(cnn_feat_map)  # [B, N, 64]
#
#         # GCN部分
#         gcn_feat = self.gcn(data.x, data.edge_index, data.batch)  # [B, gcn_out]
#         gcn_feat_proj = self.gcn_proj(gcn_feat).unsqueeze(1)      # [B, 1, 64]
#
#         # 保存节点嵌入
#         self.gcn_node_embeds = self.gcn.node_embeds.detach() if self.gcn.node_embeds is not None else None
#
#         # 取出所有节点注意力权重（[total_nodes]）
#         node_attention = self.gcn.get_node_importance().detach()  # [total_num_nodes]
#
#         # 按样本分组，样本数 = batch_size
#         batch_size = data.batch.max().item() + 1
#         nodes_per_sample = 90
#
#         # 确保总节点数能整除每个样本节点数
#         assert node_attention.shape[0] == batch_size * nodes_per_sample, \
#             f"节点数不匹配：{node_attention.shape[0]} != {batch_size}*{nodes_per_sample}"
#
#         # 按样本切分注意力权重，形状为 list，元素是 [90]
#         self.gcn_node_importance_per_sample = torch.split(node_attention, nodes_per_sample)
#
#         # Cross Attention 和融合
#         cross_attended = self.cross_attn_block(query=gcn_feat_proj, key=cnn_feat_tokens, value=cnn_feat_tokens)  # [B,1,64]
#         global_cnn_token = torch.mean(cnn_feat_tokens, dim=1, keepdim=True)  # [B,1,64]
#         fused = self.gated_fusion(cross_attended, global_cnn_token)  # [B,1,64]
#
#         out = self.fc(fused.squeeze(1)).squeeze(1)  # [B]
#
#         return out
