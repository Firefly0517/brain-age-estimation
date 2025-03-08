import torch.nn as nn
import torch.nn.functional as F
import torch
from mamba_ssm import Mamba

def make_model(args):
    return MambaNet(args)

class
class MambaBlock(nn.Module):
    def __init__(self, args, channels, mlp_ratio=4):
        """
            channels: 输入特征的通道数 (C)
            mlp_ratio: MLP 隐藏层的通道扩展比例
        """
        super(MambaBlock, self).__init__()
        self.args = args

        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

        self.mamba = Mamba(channels)

        hidden_dim = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, channels)

        )

    def forward(self, x):
        assert len(x.shape) == 4, 'x.shape must be (B, C, H, W)'
        B, C, H, W = x.shape

        # print("x.shape", x.shape)
        x = x.permute(0, 2, 3, 1) # x [B, H, W, C]

        x_residual = x # x_residual [B, H, W, C]
        x = self.norm1(x) # x [B, H, W, C]

        x_flat = x.reshape(B, H * W, C) # x_flat [B, n_tokens, C]

        x_mamba = self.mamba(x_flat) # x_mamba [B, n_tokens, C]
        x = x_mamba.reshape(B, H, W, C) # x [B, H, W, C]
        x = x + x_residual # x [B, H, W, C]

        x_residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        # print("x.shape", x.shape)
        # print("x_residual_shape", x_residual.shape)
        x = x + x_residual

        x = x.permute(0, 3, 1, 2)

        return x

class MambaNet(nn.Module):
    def __init__(self, args, in_channels = 1, num_classes = 1, hidden_dim = 64):
        """
           MambaNet: 由多个 MambaBlock 组成，最终通过全连接层输出标量
           :param in_channels: 输入图像通道数
           :param num_classes: 输出的标量个数（任务类别）
           :param hidden_dim: MambaBlock 处理的通道数
        """
        super(MambaNet, self).__init__()
        self.args = args

        # 初始 1x1 卷积用于调整通道数
        self.input_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)

        self.mamba_blocks = nn.Sequential(
            *[MambaBlock(args, hidden_dim) for _ in range(6)]
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)  # 将 (B, C, H, W) -> (B, C, 1, 1)
        self.fc = nn.Linear(hidden_dim, num_classes)  # 输出标量

    def forward(self, x):
        x = self.input_proj(x)
        x = self.mamba_blocks(x)
        x = self.global_pool(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.fc(x)
        return x