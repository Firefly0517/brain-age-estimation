import torch.nn as nn
import torch.nn.functional as F
import torch
from mamba_ssm import Mamba
from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
import math
from .vmamba import VSSBlock
from collections import OrderedDict
from timm.models.layers import DropPath, trunc_normal_

def make_model(args):
    return MambaNet(args,
                    dims = [96, 192, 384, 768],)


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

        ly = [64, 128, 256, 96]

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

class TransformerBlock(nn.Module):
    def __init__(self,
                 hidden_dim,
                 num_heads=4,
                 mlp_ratio=4.0,
                 dropout=0.1,  # 新增dropout参数
                 attention_dropout=0.1  # 注意力专用dropout
                 ):
        super().__init__()
        # 新增多个dropout层
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.mlp_dropout = nn.Dropout(dropout)
        self.post_dropout = nn.Dropout(dropout / 2)

        # 原归一化层保持不变
        self.norm1 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-6)

        # 修改注意力层添加dropout
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=attention_dropout,  # 官方实现的注意力dropout
            bias=False
        )

        # 修改MLP结构添加dropout
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio)),
            nn.GELU(),
            self.mlp_dropout,  # MLP内部dropout
            nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim),
            self.post_dropout  # 输出后dropout
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_perm = x.permute(0, 2, 3, 1)
        x_flat = x_perm.reshape(B, H * W, C)

        # 残差连接1
        residual = x_flat
        x_flat = self.norm1(x_flat)
        x_flat, _ = self.attn(x_flat, x_flat, x_flat)
        x_flat = self.attention_dropout(x_flat)  # 注意力后dropout
        x_flat = residual + x_flat

        # 残差连接2
        residual = x_flat
        x_flat = self.norm2(x_flat)
        x_flat = self.mlp(x_flat)
        x_flat = residual + x_flat

        # 最终的dropout
        x_flat = self.post_dropout(x_flat)

        # 形状恢复
        x = x_flat.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        return x


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
                 depths=[2, 2, 2, 2],
                 dims=[96, 192, 384, 768],
                 drop_path_rate=[0.1, 0.1],
                 ssm_d_state=64,
                 ssm_ratio=4.0,
                 ssm_conv=3,
                 ssm_conv_bias=True,
                 mlp_ratio=4.0,
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
        self.channel_first = channel_first
        # self.stem = nn.Sequential(
        #     nn.Conv2d(in_channels, dims[0], kernel_size=patch_size, stride=patch_size, padding=(patch_size - 1) // 2),
        #     Permute(0, 2, 3, 1) if channel_first else nn.Identity(),  # 调整为 [B, H, W, C]
        #     nn.LayerNorm(dims[0]),
        #     Permute(0, 3, 1, 2) if channel_first else nn.Identity(),  # 恢复 [B, C, H, W]
        #     nn.GELU()
        # )

        # self.stem = BasicResBlock(in_channels, dims[0], stride=2)
        # self.stem = VGG8(1)

        self.stages = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        # dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.patch_embed = self.patch_embedding()
        self.pos_embed = self.pos_embedding(dims[0])
        self.apply(self._init_weights)
        for i in range(len(depths)):
            # 每个阶段包含多个VSSBlock
            stage = nn.Sequential(*[

                # Permute(0, 2, 3, 1),
                *[VSSBlock(
                    hidden_dim=dims[i],
                    drop_path=drop_path_rate[j],
                    ssm_d_state=ssm_d_state,
                    ssm_ratio=ssm_ratio,
                    ssm_conv=ssm_conv,
                    ssm_conv_bias=ssm_conv_bias,
                    forward_type="v05_noz",
                    channel_first=False
                    ) for j in range(depths[i])
                ],
                # Permute(0, 3, 1, 2)
            ])

            # stage = nn.Sequential(
            #     *[TransformerBlock(
            #         hidden_dim=dims[i],
            #         num_heads=4,
            #         mlp_ratio=mlp_ratio,
            #         dropout=0.1
            #     ) for _ in range(depths[i])]
            # )

            # stage = nn.Sequential(*[
            #     MambaBlock(
            #         args=args,
            #         channels=dims[i],
            #         mlp_ratio=mlp_ratio
            #     ) for j in range(depths[i])
            # ])
            self.stages.append(stage)

            # 添加下采样层（最后一个阶段除外）
            if i < len(depths) - 1:
                downsample = nn.Sequential(
                    Permute(0, 3, 1, 2),
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=3, stride=2, padding=1),
                    Permute(0, 2, 3, 1),
                    nn.LayerNorm(dims[i + 1]),

                )
                self.downsample_layers.append(downsample)
            else:
                self.downsample_layers.append(nn.Identity())

        self.regressor = nn.Sequential(OrderedDict(
            # permute1=Permute(0, 2, 3, 1) if channel_first else nn.Identity(),
            norm=nn.LayerNorm(dims[-1]),
            # norm=nn.BatchNorm2d(dims[-1]),
            permute2=Permute(0, 3, 1, 2) if channel_first else nn.Identity(),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(dims[-1], num_classes)
        ))

    def forward(self, x1, x2 = None):
        features_fusion = []
        if(self.args.modal_num == 1):
            # 单模态处理
            x1 = self.patch_embed(x1)
            if self.pos_embed is not None:
                pos_embed = self.pos_embed.permute(0, 2, 3, 1) if self.channel_first else self.pos_embed
                x1 = x1 + pos_embed
            i = 0;
            for stage, downsample in zip(self.stages, self.downsample_layers):
                # print(f"Before stage: {x1.shape}")
                x1 = stage(x1)
                # print(f"After stage: {x1.shape}")
                x1 = downsample(x1)
                # print(f"After downsample: {x1.shape}")



            return self.regressor(x1)
            # return self.fc(self.global_pool(x1).squeeze(-1).squeeze(-1))
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

            age1 = self.regressor(x1)
            age2 = self.regressor(x2)
            avg_age = (age1 + age2) / 2
            return avg_age

    def patch_embedding(self, in_chans=1, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm,
                             channel_first=True):
        # if channel first, then Norm and Output are both channel_first
        stride = patch_size // 2
        kernel_size = stride + 1
        padding = 1
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=kernel_size, stride=stride, padding=padding),
            Permute(0, 2, 3, 1) if channel_first else nn.Identity(),
            (norm_layer(embed_dim // 2) if patch_norm else nn.Identity()),
            Permute(0, 3, 1, 2) if channel_first else nn.Identity(),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding),
            Permute(0, 2, 3, 1) if channel_first else nn.Identity(),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    def pos_embedding(self, embed_dims, patch_size = 4, img_size1 = 160, img_size2 = 192):
        patch_height, patch_width = (img_size1 // patch_size, img_size2 // patch_size)
        pos_embed = nn.Parameter(torch.zeros(1, embed_dims, patch_height, patch_width))
        trunc_normal_(pos_embed, std=0.02)
        return pos_embed

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

if __name__ == '__main__':
    model = MambaNet(None)
    model = model.to('cuda')
    x = torch.randn(1, 1, 224, 224)
    y = torch.randn(1, 1, 224, 224)
    x = x.to('cuda')
    y = y.to('cuda')
    out = model(x, y)
    print(out)