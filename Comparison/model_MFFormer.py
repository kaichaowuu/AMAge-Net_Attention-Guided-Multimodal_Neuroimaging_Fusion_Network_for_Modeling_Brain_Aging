import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvEncoder2D(nn.Module):
    """Encoder for 2D fMRI time series: [B, 1, T, 1000]"""

    def __init__(self, in_channels=1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.encoder(x)


class ConvEncoder3D(nn.Module):
    """Encoder for 3D T1-weighted volumes: [B, 1, D, H, W]"""

    def __init__(self, in_channels=1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(),
            nn.MaxPool3d(2),
        )

    def forward(self, x):
        return self.encoder(x)


class FusionTransformerModule(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, fMRI_feat, T1_feat):
        B, C, H, W = fMRI_feat.shape
        _, _, D, H3D, W3D = T1_feat.shape

        fMRI_feat_up = fMRI_feat.unsqueeze(2).repeat(1, 1, D, 1, 1)
        f_flat = fMRI_feat_up.view(B, C, -1).permute(0, 2, 1)
        t_flat = T1_feat.view(B, C, -1).permute(0, 2, 1)

        fused = torch.cat([f_flat, t_flat], dim=1)  # [B, N, C]

        # ✅ 动态 positional encoding
        pos_embed = torch.randn(B, fused.size(1), fused.size(2), device=fused.device)
        fused = fused + pos_embed

        attn_out, _ = self.attn(fused, fused, fused)

        N1 = f_flat.size(1)
        fMRI_out = attn_out[:, :N1, :].permute(0, 2, 1).view(B, C, D, H, W)
        T1_out = attn_out[:, N1:, :].permute(0, 2, 1).view(B, C, D, H3D, W3D)

        fMRI_out = fMRI_out.mean(2)
        return fMRI_out, T1_out




class RegressionHead(nn.Module):
    """Regression head for age prediction"""

    def __init__(self, in_channels=128):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 20),
            nn.ReLU(),
            nn.Linear(20, 1)  # output scalar age
        )

    def forward(self, x):
        return self.regressor(x)


class MFFormerAge(nn.Module):
    """MFFormer for brain age prediction"""

    def __init__(self):
        super().__init__()
        self.encoder2d = ConvEncoder2D(in_channels=1)
        self.encoder3d = ConvEncoder3D(in_channels=1)
        self.ftm = FusionTransformerModule(embed_dim=64, num_heads=4)
        self.regressor = RegressionHead(in_channels=128)

    def forward(self, fMRI_ts, T1_vol):
        f_feat = self.encoder2d(fMRI_ts)  # [B, 64, H1, W1]
        s_feat = self.encoder3d(T1_vol)  # [B, 64, D, H2, W2]

        f_fused, s_fused = self.ftm(f_feat, s_feat)
        s_reduced = s_fused.mean(2)  # [B, 64, H2, W2]

        # ✅ 让 s_reduced 和 f_fused 空间维度一致
        if s_reduced.shape[-2:] != f_fused.shape[-2:]:
            s_reduced = F.interpolate(s_reduced, size=f_fused.shape[-2:], mode='bilinear', align_corners=False)

        fused = torch.cat([f_fused, s_reduced], dim=1)  # [B, 128, H, W]
        out = self.regressor(fused)
        return out.squeeze(1)
