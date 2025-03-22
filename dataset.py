import os
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data import DataLoader
from sklearn.model_selection import KFold
import random
import torch.nn.functional as F
from torchvision import models


# 定义ResNet3D/VGG和MLP模型

class ResNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=256):
        super(ResNet3D, self).__init__()

        # 采用 3D 版本的 ResNet（如果使用 2D 版本，会在 maxpool 时报错）
        self.resnet = models.video.r3d_18(pretrained=False)  # 适用于 3D 数据

        # 修改第一层 `conv1` 以适配 `in_channels`
        self.resnet.stem[0] = nn.Conv3d(in_channels, 64, kernel_size=(3, 7, 7),
                                        stride=(1, 2, 2), padding=(1, 3, 3), bias=False)

        # 修改全连接层（FC），调整输出维度
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, out_channels)

    def forward(self, x):
        return self.resnet(x)



class MLP(nn.Module):
    def __init__(self, in_channels=1, out_channels=256):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_channels, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, pre_transform=None, fold=0, mode="random"):
        self.root = root
        self.fold = fold
        self.mode = mode
        self.file_list = [f for f in os.listdir(os.path.join(root, 'T1')) if f.endswith('.npy')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        person_id = filename.split('.')[0]

        # 加载原始数据
        t1_image = np.load(os.path.join(self.root, 'T1', filename))  # 假设形状为 (D, H, W)
        dti_image = np.load(os.path.join(self.root, 'DTI', filename))
        gender = np.load(os.path.join(self.root, 'Gender', filename))  # 标量
        age = np.load(os.path.join(self.root, 'Age', filename))       # 标量

        # 转换为张量并调整维度
        t1_tensor = torch.from_numpy(t1_image).float().unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
        dti_tensor = torch.from_numpy(dti_image).float().unsqueeze(0).unsqueeze(0)
        gender_tensor = torch.tensor(gender, dtype=torch.float).view(1, 1)        # (1, 1)
        age_tensor = torch.tensor(age, dtype=torch.float)                         # (1,)

        return Data(
            x=torch.cat([t1_tensor, dti_tensor, gender_tensor], dim=1),  # 伪代码需根据实际特征维度调整
            edge_index=edge_index,
            edge_attr=edge_attrs,
            y=age_tensor
        )


class FusionNetwork(nn.Module):
    def __init__(self):
        super(FusionNetwork, self).__init__()

        # 特征提取分支
        self.resnet3d = ResNet3D(in_channels=1, out_channels=256)
        self.mlp = MLP(in_channels=1, out_channels=64)

        # 图处理部分（示例：使用GCN）
        self.gcn = GCNConv(256 * 2 + 64, 128)
        self.fc = nn.Linear(128, 1)

    def forward(self, data):
        # 提取特征
        t1_feat = self.resnet3d(data['t1'])  # (B, 256)
        dti_feat = self.resnet3d(data['dti'])
        gender_feat = self.mlp(data['gender'])  # (B, 64)

        # 拼接特征作为节点特征
        node_feat = torch.cat([t1_feat, dti_feat, gender_feat], dim=1)  # (B, 576)

        # 构建图（示例：每个样本为一个节点，全连接边）
        edge_index = torch.combinations(torch.arange(node_feat.size(0)), r=2).t().contiguous()

        # GCN前向传播
        x = self.gcn(node_feat, edge_index)
        x = F.relu(x)
        x = self.fc(x)

        return x


"""class MyDataset1(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, pre_transform=None, fold=0, mode="random"):
        self.root = root
        self.fold = fold
        self.mode = mode
        self.file_list = [f for f in os.listdir(os.path.join(root, 'T1')) if f.endswith('.npy')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        person_id = filename.split('.')[0]

        # 加载数据
        t1_image = np.load(os.path.join(self.root, 'T1', filename))
        dti_image = np.load(os.path.join(self.root, 'DTI', filename))
        gender = np.load(os.path.join(self.root, 'Gender', filename))
        age = np.load(os.path.join(self.root, 'Age', filename))

        # 转换为 PyTorch 张量
        t1_tensor = torch.from_numpy(t1_image).float()
        dti_tensor = torch.from_numpy(dti_image).float()
        gender_tensor = torch.tensor([gender], dtype=torch.float)
        age_tensor = torch.tensor(age, dtype=torch.float)

        # 提取特征
        resnet_model = ResNet3D(in_channels=1, out_channels=256)
        t1_features = resnet_model(t1_tensor)
        dti_features = resnet_model(dti_tensor)

        mlp_model = MLP(in_channels=1, out_channels=64)
        gender_features = mlp_model(gender_tensor)

        # 构建节点特征
        node_features = torch.cat((t1_features, dti_features, gender_features), dim=1)

        # 构建边和边特征
        edges = []
        edge_attrs = []
        if self.mode == 'random':
            num_nodes = node_features.size(0)
            edges = [(i, j) for i in range(num_nodes) for j in range(i + 1, num_nodes)]
            edge_attrs = [torch.rand(1) for _ in edges]

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attrs = torch.tensor(edge_attrs, dtype=torch.float)

        # 返回图数据
        return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attrs, y=age_tensor)
"""

'''
# 定义MyDataset类来构建图数据
class MyDataset(InMemoryDataset):
    def __init__(self, root=os.path.abspath('.') + '/graphdata', transform=None, pre_transform=None, fold=0,
                 mode="random"):
        self.fold = fold
        self.mode = mode
        super(MyDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])  # 加载处理后的数据

    @property
    def raw_file_names(self):
        return ['T1', 'DTI', 'Gender', 'Age']

    @property
    def processed_file_names(self):
        return [f'data{self.fold}.pt']

    def process(self):
        data_list = []

        # 手动指定数据文件夹路径
        base_path = '/data2/wangchangmiao/liuxiaoshuai/IXI/'
        #base_path = '/mntnfs/med_data2/liuxiaoshuai/IXI/'
        t1_path = os.path.join(base_path, 'T1')  # T1图像文件夹路径
        dti_path = os.path.join(base_path, 'DTI')  # DTI图像文件夹路径
        gender_path = os.path.join(base_path, 'Gender')  # 性别数据路径
        age_path = os.path.join(base_path, 'Age')  # 大脑年龄标签路径

        # 获取T1文件夹下的所有.npy文件
        t1_files = sorted([f for f in os.listdir(t1_path) if f.endswith('.npy')])

        for filename in t1_files:
            person_id = filename.split('.')[0]  # 去掉扩展名，得到ID

            # 构造其他模态的文件路径
            t1_file = os.path.join(t1_path, filename)
            dti_file = os.path.join(dti_path, filename)
            gender_file = os.path.join(gender_path, filename)
            age_file = os.path.join(age_path, filename)

            # 读取数据
            #t1_image = np.load(t1_file)
            #dti_image = np.load(dti_file)
            #gender = np.load(gender_file)  # 性别：0（女）或 1（男）
            #age = np.load(age_file)  # 年龄（回归任务）
            t1_image = np.load(t1_file, mmap_mode="r")  # ✅ 避免一次性加载整个 .npy
            dti_image = np.load(dti_file, mmap_mode="r")
            gender = np.load(gender_file, mmap_mode="r")
            age = np.load(age_file, mmap_mode="r")

            # 转换为 PyTorch 张量
            t1_tensor = torch.tensor(t1_image, dtype=torch.float).unsqueeze(0).unsqueeze(0)  # 添加 batch 维度
            dti_tensor = torch.tensor(dti_image, dtype=torch.float).unsqueeze(0).unsqueeze(0)
            gender_tensor = torch.tensor([gender], dtype=torch.float)
            age_tensor = torch.tensor(age, dtype=torch.float)  # 大脑年龄标签

            # 提取特征
            resnet_model = ResNet3D(in_channels=1, out_channels=256)
            t1_features = resnet_model(t1_tensor)
            dti_features = resnet_model(dti_tensor)

            mlp_model = MLP(in_channels=1, out_channels=64)
            gender_features = mlp_model(gender_tensor)

            # 构建节点特征
            node_features = torch.cat((t1_features, dti_features, gender_features), dim=1)

            # 构建边和边特征（可以使用互信息计算或其他方式）
            edges = []
            edge_attrs = []

            if self.mode == 'random':
                num_nodes = node_features.size(0)
                edges = [(i, j) for i in range(num_nodes) for j in range(i + 1, num_nodes)]
                edge_attrs = [torch.rand(1) for _ in edges]  # 这里使用随机权重

            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_attrs = torch.tensor(edge_attrs, dtype=torch.float)

            # 创建图数据对象
            data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attrs, y=age_tensor)
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
'''


# 主函数中的修改部分保持不变，您只需要确保数据集和模型的适配
if __name__ == "__main__":
    dataset = MyDataset(root = "/data2/wangchangmiao/liuxiaoshuai/IXI")
    data = dataset[0]
    print(data)


'''
class VGG3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=256):
        super(VGG3D, self).__init__()

        # 加载预训练的 VGG16 模型，并修改其输入通道数
        self.vgg = models.vgg16(pretrained=True)

        # 修改第一层卷积以适应 3D 输入（更小的输入通道）
        self.vgg.features[0] = nn.Conv3d(in_channels, 64, kernel_size=(3, 3, 3), stride=1, padding=1)

        # 修改全连接层
        self.vgg.classifier[6] = nn.Linear(self.vgg.classifier[6].in_features, out_channels)

    def forward(self, x):
        return self.vgg(x)
'''