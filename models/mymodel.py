import torch.nn as nn
import torch.nn.functional as F
import torch
from mamba_ssm import Mamba

def make_model(args):
    return mymodel(args)

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]

            return x

class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, num_slices=None):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor

            # nslices=num_slices,
        )

    def forward(self, x):
        B, C = x.shape[:2]
        x_skip = x
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)

        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        out = out + x_skip

        return out


class MlpChannel(nn.Module):
    def __init__(self, hidden_size, mlp_dim, ):
        super().__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class GSC(nn.Module):
    def __init__(self, in_channles) -> None:
        super().__init__()

        self.proj = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.norm = nn.InstanceNorm3d(in_channles)
        self.nonliner = nn.ReLU()

        self.proj2 = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.norm2 = nn.InstanceNorm3d(in_channles)
        self.nonliner2 = nn.ReLU()

        self.proj3 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.norm3 = nn.InstanceNorm3d(in_channles)
        self.nonliner3 = nn.ReLU()

        self.proj4 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.norm4 = nn.InstanceNorm3d(in_channles)
        self.nonliner4 = nn.ReLU()

    def forward(self, x):
        x_residual = x

        x1 = self.proj(x)
        x1 = self.norm(x1)
        x1 = self.nonliner(x1)

        x1 = self.proj2(x1)
        x1 = self.norm2(x1)
        x1 = self.nonliner2(x1)

        x2 = self.proj3(x)
        x2 = self.norm3(x2)
        x2 = self.nonliner3(x2)

        x = x1 + x2
        x = self.proj4(x)
        x = self.norm4(x)
        x = self.nonliner4(x)

        return x + x_residual

class MambaEncoder(nn.Module):
    def __init__(self, in_chans=1, depths=[2, 2, 2, 2], dims=[48, 96, 192, 384],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3]):
        super().__init__()

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv3d(in_chans, dims[0], kernel_size=7, stride=2, padding=3),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.InstanceNorm3d(dims[i]),
                nn.Conv3d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        self.gscs = nn.ModuleList()
        num_slices_list = [64, 32, 16, 8]
        cur = 0
        for i in range(4):
            gsc = GSC(dims[i])

            stage = nn.Sequential(
                *[MambaLayer(dim=dims[i], num_slices=num_slices_list[i]) for j in range(depths[i])]
            )

            self.stages.append(stage)
            self.gscs.append(gsc)
            cur += depths[i]

        self.out_indices = out_indices

        self.mlps = nn.ModuleList()
        for i_layer in range(4):
            layer = nn.InstanceNorm3d(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
            self.mlps.append(MlpChannel(dims[i_layer], 2 * dims[i_layer]))

    def forward_features(self, x, y, flag):
        if flag == 0:
            outs = []
            for i in range(4):
                x = self.downsample_layers[i](x)
                x = self.gscs[i](x)
                x = self.stages[i](x)

                if i in self.out_indices:
                    norm_layer = getattr(self, f'norm{i}')
                    x_out = norm_layer(x)
                    x_out = self.mlps[i](x_out)
                    outs.append(x_out)
        elif flag == 1:
            outs = []
            for i in range(4):
                x = self.downsample_layers[i](x)
                x = self.gscs[i](x)
                x = self.stages[i](x)

                if i in self.out_indices:

                    norm_layer = getattr(self, f'norm{i}')
                    x_out = norm_layer(x)
                    x_out = self.mlps[i](x_out)

                    y_out = y[i]
                    final_out = y_out + x_out
                    outs.append(final_out)
        else:
            raise ValueError("flag wrong")
        return outs

    def forward(self, x, y, flag):
        x = self.forward_features(x, y, flag)
        return x


class MultiScale3DNet(nn.Module):
    def __init__(self):
        super().__init__()

        # 分支1处理：48通道输入
        self.branch1 = nn.Sequential(
            nn.Conv3d(48, 24, kernel_size=3, stride=2, padding=1),  # [1,24,40,48,40]
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),  # 全局平均池化 [1,24,1,1,1]
            nn.Flatten(),
            nn.Linear(24, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        # 分支2处理：96通道输入
        self.branch2 = nn.Sequential(
            nn.Conv3d(96, 48, kernel_size=3, stride=2, padding=1),  # [1,48,20,24,20]
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Linear(24, 1))

        # 分支3处理：192通道输入
        self.branch3 = nn.Sequential(
            nn.Conv3d(192, 96, kernel_size=3, padding=1),  # 保持空间维度 [1,96,20,24,20]
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Linear(48, 1))

        # 分支4处理：384通道输入
        self.branch4 = nn.Sequential(
            nn.Conv3d(384, 192, kernel_size=3, padding=1),  # [1,192,10,12,10]
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(192, 96),
            nn.ReLU(),
            nn.Linear(96, 1))

    def forward(self, x1, x2, x3, x4):
        # 各分支独立处理
        out1 = self.branch1(x1)  # [1,1]
        out2 = self.branch2(x2)  # [1,1]
        out3 = self.branch3(x3)  # [1,1]
        out4 = self.branch4(x4)  # [1,1]

        return out1 + out2 + out3 + out4


class mymodel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_blocks = args.num_blocks
        self.conv_first_layers = nn.ModuleList([
            nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=1)
            for dim in [48, 96, 192, 384]
        ])
        self.upsample_layers = nn.ModuleList([
            nn.ConvTranspose3d(in_channels=dim, out_channels=dim//2, kernel_size=2, stride=2)
            for dim in [48, 96, 192, 384]
        ])
        self.MambaEncoder = MambaEncoder()
        self.conv1x1x1_layers = nn.ModuleList([
            nn.Conv3d(channel_dim, 100, kernel_size=1)
            for channel_dim in [48, 96, 192, 384]
        ])

        self.mlp = nn.Sequential(
            nn.Linear(100, 128),  # 40 -> 128
            nn.ReLU(),
            nn.Linear(128, 1)  # 128 -> 1 (最终的脑龄预测)
        )
        self.GSCs = nn.ModuleList([
            GSC(dim)
            for dim in [48, 96, 192, 384]
        ])
        self.MultiScale3DNet = MultiScale3DNet()

    def forward(self, x1, x2):
        enc_modal2 = self.MambaEncoder(x2, x2, 0) # DTI
        enc = self.MambaEncoder(x1, enc_modal2, 1) # T1 & DTI
        upsampled = []


        for i in range(self.num_blocks):
            enc[i] = self.conv_first_layers[i](enc[i])
            x_upsampled = self.upsample_layers[i](enc[i])
            upsampled.append(x_upsampled)
        for i in range(self.num_blocks - 1):
            enc[i] = enc[i] + upsampled[i + 1]
            # print("x1.shape",enc[i].shape)
            # print("upsampled.shape",upsampled[i + 1].shape)

        out = []
        # for i in range(self.num_blocks):
        #     x = enc[i]
        #     print("x_shape:", x.shape)
        #     x = F.adaptive_avg_pool3d(x, (1, 1, 1))  #  [batch_size, channels, 1, 1, 1]
        #     x = self.conv1x1x1_layers[i](x) # [1, 100, 1, 1, 1]
        #     x = x.view(x.shape[0], -1)
        #     # print("x4.shape",x.shape) # [1, 100]
        #     x = self.mlp(x) # [1, 1]
        #     #x = F.log_softmax(x, dim=1)
        #     x = x.view(x.shape[0], x.shape[1])  # 最终变成 [batch_size, 100]
        #     # print("x2shape", x.shape) [batch_size, 1]
        #     out.append(x)
        for i in range(self.num_blocks):
            enc[i] = self.GSCs[i](enc[i])
            # print(enc[i].shape)
        out = self.MultiScale3DNet(enc[0], enc[1], enc[2], enc[3])

        return out


