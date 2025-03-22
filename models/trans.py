import torch.nn as nn
import torch.nn.functional as F
import torch
from mamba_ssm import Mamba
from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
import math
from torch.nn import Linear

from .vmamba import VSSBlock
from collections import OrderedDict

def make_model(args):
    return MambaNet(args,
                    dims = [96, 96, 96, 96])

class Permute(nn.Module):
    """维度重排列模块"""
    def __init__(self, *order):
        super().__init__()
        self.order = order

    def forward(self, x):
        return x.permute(*self.order)

class Linear2d(nn.Linear):
    def forward(self, x: torch.Tensor):
        # B, C, H, W = x.shape
        return F.conv2d(x, self.weight[:, :, None, None], self.bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        state_dict[prefix + "weight"] = state_dict[prefix + "weight"].view(self.weight.shape)
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                             error_msgs)

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


class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DWConv, self).__init__()
        # 深度卷积
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels)
        # 逐点卷积
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class FusionBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 mlp_ratio=4.0,
                 ):
        super().__init__()
        self.norm1 = nn.Sequential(
            Permute(0, 2, 3, 1),
            nn.LayerNorm(in_channels),
            Permute(0, 3, 1, 2)
        )

        self.mlp_num = int(in_channels * mlp_ratio)
        self.Linear1 = Linear2d(in_channels, self.mlp_num)

        self.conv1 = DWConv(self.mlp_num, self.mlp_num)


        self.SS2D = nn.Sequential(
                    Permute(0, 2, 3, 1),
                    VSSBlock(hidden_dim=self.mlp_num,
                             forward_type="v05_noz",),
                    Permute(0, 3, 1, 2)
                    )
        self.norm2 = nn.Sequential(
            Permute(0, 2, 3, 1),
            nn.LayerNorm(self.mlp_num),
            Permute(0, 3, 1, 2)
        )

        self.Linear2 = Linear2d(self.mlp_num, in_channels)

    def forward(self, x1, x2, x3 = None):

        x1 = self.norm1(x1)
        x2 = self.norm1(x2)

        x1_residual = x1
        x2_residual = x2

        x1 = self.Linear1(x1)
        x2 = self.Linear1(x2)
        x1_residual = self.Linear1(x1_residual)
        x2_residual = self.Linear1(x2_residual)

        x1 = self.conv1(x1)
        x2 = self.conv1(x2)

        x_fus = x1 * x2
        x_tot = x_fus + x1 + x2

        x_tot = self.SS2D(x_tot)
        x_tot = self.norm2(x_tot)

        x1_ = x1_residual * x_tot
        x2_ = x2_residual * x_tot
        x_tot = x_tot + x1_ + x2_
        x_fusion = self.Linear2(x_tot)

        return x_fusion


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, act_layer=nn.GELU):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        B, H, W, C = x.shape
        x = x.reshape(B, H * W, C)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x.reshape(B, H, W, C)

class TransformerBlock(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=4,
                 mlp_ratio=4,
                 drop_path_rate=0.,
                 qkv_bias=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiheadAttention2D(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias
        )
        self.drop_path = nn.DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=nn.GELU
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]

        # Attention分支
        residual = x
        x = self.norm1(x)
        x = self.attn(x)  # [B, H, W, C]
        x = residual + self.drop_path(x)

        # MLP分支
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + self.drop_path(x)

        return x.permute(0, 3, 1, 2)  # 恢复原始维度


class MultiheadAttention2D(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 使用3D卷积实现位置编码
        self.pos_embed = nn.Conv3d(dim, dim, kernel_size=(3, 3, 3), padding=1, groups=dim)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, H, W, C = x.shape

        # 添加3D位置编码
        x = x.permute(0, 3, 1, 2).unsqueeze(2)  # [B, C, 1, H, W]
        x = x + self.pos_embed(x).squeeze(2)
        x = x.permute(0, 2, 3, 1)  # 恢复[B, H, W, C]

        # 生成QKV
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # [B, num_heads, HW, head_dim]

        # 注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
        return self.proj(x)

class MambaNet(nn.Module):
    def __init__(self,
                 args,
                 in_channels=1,
                 num_classes=1,
                 depths=[1, 1, 1, 1],
                 dims=[96, 192, 384, 768],
                 drop_path_rate=0.,
                 ssm_d_state=16,
                 ssm_ratio=2.0,
                 ssm_conv=3,
                 ssm_conv_bias=True,
                 mlp_ratio=4.0,
                 patch_size=3,
                 channel_first=False,
                 ):
        """
           MambaNet: 由多个 MambaBlock 组成，最终通过全连接层输出标量
           :param in_channels: 输入图像通道数
           :param num_classes: 输出的标量个数（任务类别）
           :param hidden_dim: MambaBlock 处理的通道数
        """
        super(MambaNet, self).__init__()
        self.args = args

        # self.stem = nn.Sequential(
        #     nn.Conv2d(in_channels, dims[0], kernel_size=patch_size, stride=patch_size, padding=(patch_size - 1) // 2),
        #     Permute(0, 2, 3, 1) if channel_first else nn.Identity(),  # 调整为 [B, H, W, C]
        #     nn.LayerNorm(dims[0]),
        #     Permute(0, 3, 1, 2) if channel_first else nn.Identity(),  # 恢复 [B, C, H, W]
        #     nn.GELU()
        # )

        self.stem = BasicResBlock(in_channels, dims[0], stride=1)

        # self.stem = VGG8(1)

        self.stages = nn.ModuleList()
        self.fusion_layers = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        cur = 0
        for i in range(len(depths)):
            # 每个阶段包含多个VSSBlock
            # stage = nn.Sequential(*[
            #
            #     Permute(0, 2, 3, 1),
            #     *[VSSBlock(
            #         hidden_dim=dims[i],
            #         drop_path=dp_rates[cur + j],
            #         ssm_d_state=ssm_d_state,
            #         ssm_ratio=ssm_ratio,
            #         ssm_conv=ssm_conv,
            #         ssm_conv_bias=ssm_conv_bias,
            #         forward_type="v05_noz",
            #         channel_first=False
            #         ) for j in range(depths[i])
            #     ],
            #     Permute(0, 3, 1, 2)
            # ])

            # stage = nn.Sequential(*[
            #     MambaBlock(
            #         args=args,
            #         channels=dims[i],
            #         mlp_ratio=mlp_ratio
            #     ) for j in range(depths[i])
            # ])

            stage = nn.Sequential(*[
                TransformerBlock(
                    dim=dims[i],
                    num_heads=8,
                    mlp_ratio=mlp_ratio,
                    drop_path_rate=dp_rates[cur + j]
                ) for j in range(depths[i])
            ])
            self.stages.append(stage)
            cur += depths[i]

            fusion_block = FusionBlock(dims[i],)
            self.fusion_layers.append(fusion_block)

            # 添加下采样层（最后一个阶段除外）
            if i < len(depths) - 1:
                downsample = nn.Sequential(
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=3, stride=2, padding=1),
                    Permute(0, 2, 3, 1),
                    nn.LayerNorm(dims[i + 1]),
                    Permute(0, 3, 1, 2)
                )
                self.downsample_layers.append(downsample)
            else:
                self.downsample_layers.append(nn.Identity())

        self.regressor = nn.Sequential(OrderedDict(
            permute1=Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            norm=nn.LayerNorm(dims[-1]),
            permute2=Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(dims[-1], num_classes)
        ))

        # 计算 Mlp 输入维度
        mlp_in_dim = sum([dims[i] for i in range(len(depths))])
        # self.mlp = Mlp(in_features=1382400, hidden_features=256, out_features=num_classes, drop=0.1, channels_first=False)

        self.global_pool = nn.AdaptiveAvgPool2d(1)  # 将 (B, C, H, W) -> (B, C, 1, 1)
        self.fc = nn.Linear(dims[-1], num_classes)  # 输出标量

    def forward(self, x1, x2 = None):

        if(self.args.modal_num == 1):
            # 单模态处理
            outputs = []
            x1 = self.stem(x1)
            i = 0;
            for stage, downsample in zip(self.stages, self.downsample_layers):
                i += 1
                print(f"stage{i}:, x1.shape:{x1.shape}")
                x1 = stage(x1)
                # outputs.append(x1)
                # x1 = downsample(x1)

            age = self.regressor(x1)
            # # 对保存的输出进行 flatten 和 concat
            # flattened_outputs = [torch.flatten(output, start_dim=1) for output in outputs]
            # # print("flatten_shape:",flattened_outputs.shape)
            # concat_output = torch.cat(flattened_outputs, dim=1)
            # print(f"concat_shape:{concat_output.shape}")

            # age = self.mlp(concat_output)
            return age
            # return self.regressor(x1)
        elif self.args.modal_num == 2:
            features_fusion = []
            x1 = self.stem(x1)
            print("x1.shape", x1.shape)
            x2 = self.stem(x2)
            for stage, fusion, downsample in zip(self.stages, self.fusion_layers, self.downsample_layers):
                x1 = stage(x1)
                x2 = stage(x2)

                x_fusion = fusion(x1, x2)
                features_fusion.append(x_fusion)
                x1 = x1 + x_fusion
                x2 = x2 + x_fusion

                # x1 = downsample(x1)
                # x2 = downsample(x2)

            age1 = self.regressor(x1)
            age2 = self.regressor(x2)
            avg_age = (age1 + age2) / 2
            return avg_age

if __name__ == '__main__':
    model = MambaNet(None)
    model = model.to('cuda')
    x = torch.randn(1, 1, 224, 224)
    y = torch.randn(1, 1, 224, 224)
    x = x.to('cuda')
    y = y.to('cuda')
    out = model(x, y)
    print(out)