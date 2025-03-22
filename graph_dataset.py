import torch
import torch.nn as nn
from torchvision.models import resnet50
import torch.nn.functional as F
import numpy as np

class ModalityEncoders(nn.Module):
    def __init__(self):
        super().__init__()

        self.medical_encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            ResBlock3D(16, 32),
            ResBlock3D(32, 64),
            nn.AdaptiveAvgPool3d(1)
        )

        # 性别编码器
        self.gender_encoder = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 16)
        )

    def forward(self, t1, dti, gender):
        # T1处理
        t1_feat = self.medical_encoder(t1).squeeze()  # [B, 64]

        # DTI处理
        dti_feat = self.medical_encoder(dti).squeeze()  # [B, 64]

        # 性别处理
        gender_feat = self.gender_encoder(gender)  # [B, 16]

        return t1_feat, dti_feat, gender_feat


class ResBlock3D(nn.Module):
    """3D残差块"""

    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv3d(in_c, out_c, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_c)
        self.conv2 = nn.Conv3d(out_c, out_c, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_c)
        self.shortcut = nn.Conv3d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class MultimodalGraphBuilder(nn.Module):
    def __init__(self, feat_dims=[64, 64, 16]):
        super().__init__()
        # 特征融合层
        self.fusion = nn.Linear(sum(feat_dims), 128)

        # 图结构参数
        self.k_neighbors = 8  # 最近邻数量

    def forward(self, t1_feat, dti_feat, gender_feat):
        # 特征拼接
        fused = torch.cat([t1_feat, dti_feat, gender_feat], dim=1)
        fused = F.relu(self.fusion(fused))  # [B, 128]

        # 构建全连接图
        adj = self.build_graph(fused)  # [B, B]
        return fused, adj

    def build_graph(self, features):
        """基于余弦相似度构建邻接矩阵"""
        cos_sim = F.cosine_similarity(
            features.unsqueeze(1),
            features.unsqueeze(0),
            dim=2
        )  # [B, B]

        # 保留top-k连接
        topk = torch.topk(cos_sim, self.k_neighbors, dim=1)
        mask = torch.zeros_like(cos_sim)
        mask.scatter_(1, topk.indices, 1.0)
        return mask * cos_sim