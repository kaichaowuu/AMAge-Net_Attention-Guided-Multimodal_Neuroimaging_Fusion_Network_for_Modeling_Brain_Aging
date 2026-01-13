import torch
import torch.nn as nn
import torch.nn.functional as F

# ============ ResNet3D 部分 ============

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BottleneckBlock, self).__init__()
        mid_channels = out_channels // 4

        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_channels)

        self.conv2 = nn.Conv3d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(mid_channels)

        self.conv3 = nn.Conv3d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels)

        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        identity = self.downsample(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out += identity
        return F.relu(out)


class DownResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, stride, ConvBlock):
        super(DownResBlock, self).__init__()
        layers = []
        layers.append(ConvBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ConvBlock(out_channels, out_channels, 1))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ResNet3D(nn.Module):
    def __init__(self, in_channels=1, block=BottleneckBlock, layers=[1, 2, 2], channels=[16, 64, 128, 256]):
        super(ResNet3D, self).__init__()
        self.in_channels = channels[0]

        self.conv1 = nn.Conv3d(in_channels, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = DownResBlock(self.in_channels, channels[1], layers[0], stride=1, ConvBlock=block)
        self.layer2 = DownResBlock(channels[1], channels[2], layers[1], stride=2, ConvBlock=block)
        self.layer3 = DownResBlock(channels[2], channels[3], layers[2], stride=2, ConvBlock=block)

        self.out_channels = channels[3]

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x  # [B, C, D, H, W]

# ============ 脑龄回归模型部分 ============

class SmriModel(nn.Module):
    def __init__(self):
        super(SmriModel, self).__init__()

        self.backbone = ResNet3D(in_channels=1)
        self.feature_dim = self.backbone.out_channels  # 默认256

        self.patch_proj = nn.Linear(self.feature_dim, 64)

        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, img):
        if img.dim() == 4:
            img = img.unsqueeze(1)  # [B, 1, D, H, W]

        feat_map = self.backbone(img)                      # [B, C, d, h, w]
        feat_tokens = feat_map.flatten(2).transpose(1, 2)  # [B, N, C]
        feat_tokens = self.patch_proj(feat_tokens)         # [B, N, 64]

        global_feat = torch.mean(feat_tokens, dim=1)       # [B, 64]
        out = self.fc(global_feat).squeeze(1)              # [B]
        return out
