# ============================================================
# Multimodal sMRI (3D MRI) + Functional Connectivity (90x90)
# Vision Transformer + Cross-Attention Fusion (PyTorch)
# ============================================================
# This code is a CLEAN, RESEARCH-ORIENTED skeleton adapted from:
# "A Mixed Deep Neural Network for sMRI and fMRI Features Fusion in AD Detection"
#
# Assumptions:
# - sMRI input: [B, 1, D, H, W]  (gray-matter volume)
# - FC input:   [B, 90, 90]     (functional connectivity matrix)
# - Task: classification or regression (easy to switch)
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------
# 1. 3D Patch Embedding for sMRI (ViT-style)
# ------------------------------------------------------------
class PatchEmbed3D(nn.Module):
    def __init__(self, in_chans=1, embed_dim=192, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: [B, 1, D, H, W]
        x = self.proj(x)              # [B, C, D', H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, N_patches, C]
        return x


# ------------------------------------------------------------
# 2. Transformer Encoder Block
# ------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=4.0, drop=0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


# ------------------------------------------------------------
# 3. sMRI Vision Transformer Encoder
# ------------------------------------------------------------
class SMRIViT(nn.Module):
    def __init__(self, embed_dim=192, depth=4):
        super().__init__()
        self.patch_embed = PatchEmbed3D(embed_dim=embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)  # [B, N, C]
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


# ------------------------------------------------------------
# 4. Functional Connectivity Encoder (90x90)
# ------------------------------------------------------------
class FCEncoder(nn.Module):
    def __init__(self, in_nodes=90, embed_dim=192):
        super().__init__()
        self.proj = nn.Linear(in_nodes, embed_dim)
        self.transformer = TransformerBlock(embed_dim, num_heads=4)

    def forward(self, fc):
        # fc: [B, 90, 90]
        x = self.proj(fc)           # [B, 90, C]
        x = self.transformer(x)    # model inter-node relations
        return x


# ------------------------------------------------------------
# 5. Cross-Attention Fusion Module
# ------------------------------------------------------------
class CrossAttentionFusion(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, query, key_value):
        # query:     sMRI tokens [B, N1, C]
        # key_value: FC tokens   [B, N2, C]
        q = self.norm_q(query)
        kv = self.norm_kv(key_value)
        out, _ = self.cross_attn(q, kv, kv)
        return query + out


# ------------------------------------------------------------
# 6. Multimodal Network (Classification / Regression)
# ------------------------------------------------------------
class MultimodalBrainAgeModel(nn.Module):
    """
    Multimodal Brain AGE Regression Model
    Output: predicted brain age (continuous)
    """
    def __init__(self, embed_dim=192):
        super().__init__()
        self.smri_encoder = SMRIViT(embed_dim=embed_dim)
        self.fc_encoder = FCEncoder(embed_dim=embed_dim)
        self.fusion = CrossAttentionFusion(embed_dim)

        # Regression head (Brain Age)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0),
            nn.Linear(128, 1)
        )

    def forward(self, smri, fc):
        smri_tokens = self.smri_encoder(smri)
        fc_tokens = self.fc_encoder(fc)

        fused = self.fusion(smri_tokens, fc_tokens)
        fused = fused.mean(dim=1)  # global pooling

        age = self.head(fused)
        return age.squeeze(1)



