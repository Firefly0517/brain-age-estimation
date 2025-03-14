import torch.nn as nn
import torch.nn.functional as F
import torch
from mamba_ssm import Mamba
import mamba_vision
from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
import math
from .vmamba import VSSBlock
from collections import OrderedDict

def make_model(args):
    return MambaNet(args)


class convBlock(nn.Module):
    def __init__(self, inplace, outplace, kernel_size=3, padding=1):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplace, outplace, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(outplace)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x


class Permute(nn.Module):
    """维度重排列模块"""
    def __init__(self, *order):
        super().__init__()
        self.order = order

    def forward(self, x):
        return x.permute(*self.order)

class VGG8(nn.Module):
    def __init__(self, inplace):
        super().__init__()

        ly = [64, 128, 256, 512]

        self.ly = ly

        self.maxp = nn.MaxPool2d(2)

        self.conv11 = convBlock(inplace, ly[0])
        self.conv12 = convBlock(ly[0], ly[0])

        self.conv21 = convBlock(ly[0], ly[1])
        self.conv22 = convBlock(ly[1], ly[1])

        self.conv31 = convBlock(ly[1], ly[2])
        self.conv32 = convBlock(ly[2], ly[2])

        self.conv41 = convBlock(ly[2], ly[3])
        self.conv42 = convBlock(ly[3], ly[3])

    def forward(self, x):
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.maxp(x)

        x = self.conv21(x)
        x = self.conv22(x)
        x = self.maxp(x)

        x = self.conv31(x)
        x = self.conv32(x)
        x = self.maxp(x)

        x = self.conv41(x)
        x = self.conv42(x)
        x = self.maxp(x)

        return x


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class BasicResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # 主路径
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 跳跃连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return self.relu(x)


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
    def __init__(self,
                 args,
                 in_channels=1,
                 num_classes=1,
                 depths=[2, 2, 8, 2],
                 dims=[96, 192, 384, 768],
                 drop_path_rate=0.1,
                 ssm_d_state=1,
                 ssm_ratio=2.0,
                 ssm_conv=3,
                 ssm_conv_bias=False,
                 patch_size=3,
                 channel_first=True):
        """
           MambaNet: 由多个 MambaBlock 组成，最终通过全连接层输出标量
           :param in_channels: 输入图像通道数
           :param num_classes: 输出的标量个数（任务类别）
           :param hidden_dim: MambaBlock 处理的通道数
        """
        super(MambaNet, self).__init__()
        self.args = args

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=patch_size, stride=patch_size, padding=(patch_size - 1) // 2),
            nn.LayerNorm(dims[0]),
            nn.GELU()
        )

        self.stages = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        cur = 0
        for i in range(len(depths)):
            # 每个阶段包含多个VSSBlock
            stage = nn.Sequential(*[
                VSSBlock(
                    hidden_dim=dims[i],
                    drop_path=dp_rates[cur + j],
                    ssm_d_state=ssm_d_state,
                    ssm_ratio=ssm_ratio,
                    ssm_conv=ssm_conv,
                    ssm_conv_bias=ssm_conv_bias,
                    forward_type="v05_noz",
                ) for j in range(depths[i])
            ])
            self.stages.append(stage)
            cur += depths[i]

            # 添加下采样层（最后一个阶段除外）
            if i < len(depths) - 1:
                downsample = nn.Sequential(
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=3, stride=2, padding=1),
                    nn.LayerNorm(dims[i + 1])
                )
                self.downsample_layers.append(downsample)
            else:
                self.downsample_layers.append(nn.Identity())

        self.regressor = nn.Sequential(OrderedDict(
            norm=nn.LayerNorm(dims[-1]) if not channel_first else nn.Identity(),
            permute=Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(dims[-1], num_classes)
        ))

    def forward(self, x1, x2 = None):
        features_fusion = []
        if(self.args.modal_num == 1):
            # 单模态处理
            x1 = self.stem(x1)
            for stage, downsample in zip(self.stages, self.downsample_layers):
                x1 = stage(x1)
                x1 = downsample(x1)
            return self.regressor(x1)
        elif self.args.modal_num == 2:
            x1 = self.stem(x1)
            x2 = self.stem(x2)
            for stage, downsample in zip(self.stages, self.downsample_layers):
                x1 = stage(x1)
                x2 = stage(x2)

                x_fusion = self.fusion(x1, x2)
                features_fusion.append(x_fusion)
                x1 = x1 + x_fusion
                x2 = x2 + x_fusion

                x1 = downsample(x1)
                x2 = downsample(x2)

            return self.regressor(x1), self.regressor(x2)

if __name__ == '__main__':
    model = MambaNet(None)
    model = model.to('cuda')
    x = torch.randn(1, 1, 224, 224)
    y = torch.randn(1, 1, 224, 224)
    x = x.to('cuda')
    y = y.to('cuda')
    out = model(x, y)
    print(out)