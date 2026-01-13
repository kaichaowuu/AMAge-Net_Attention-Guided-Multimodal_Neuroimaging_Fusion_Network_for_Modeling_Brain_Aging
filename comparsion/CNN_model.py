import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d(kernel_size=(4, 4, 1))

        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.conv22 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d(kernel_size=(3, 3, 1))

        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        self.pool3 = nn.MaxPool3d(kernel_size=3)

        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm3d(256)
        self.conv44 = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.bn44 = nn.BatchNorm3d(256)
        self.conv444 = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.bn444 = nn.BatchNorm3d(256)
        self.pool4 = nn.MaxPool3d(kernel_size=3)

        self.fc1 = None  # 延迟定义
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn22(self.conv22(F.relu(self.bn2(self.conv2(x)))))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn444(self.conv444(F.relu(self.bn44(self.conv44(F.relu(self.bn4(self.conv4(x))))))))))

        x = x.view(x.size(0), -1)

        # 延迟定义 fc1（动态构建）
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.size(1), 128).to(x.device)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)
        return output
