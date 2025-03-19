import os
import torch
from torch_geometric.data import InMemoryDataset, Data
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
import random
import torchvision.models as models
import torch.nn as nn


class MyDataset(InMemoryDataset):
    def __init__(self, root=os.path.abspath('.') + '/ADNI', transform=None, pre_transform=None, fold=0, groups=[21, 6, 17, 14, 20], mode="random"):
        self.fold = fold
        self.groups = groups
        self.mode = mode
        # 初始化VGG和MLP模型
        self.vgg = models.vgg16(pretrained=False)
        self.vgg.classifier = nn.Sequential(*list(self.vgg.classifier.children())[:-1])  # 移除最后一层
        self.mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 32)
        )
        super(MyDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['T1', 'DTI', 'gender']

    @property
    def processed_file_names(self):
        return [f'data{self.fold}.pt']

    def process(self):
        data_list = []
        t1_dir = os.path.join(self.raw_dir, 'T1')
        dti_dir = os.path.join(self.raw_dir, 'DTI')
        gender_dir = os.path.join(self.raw_dir, 'gender')

        for file_name in os.listdir(t1_dir):
            if file_name.endswith('.npy'):
                # 加载T1、DTI和gender数据
                t1_data = np.load(os.path.join(t1_dir, file_name))
                dti_data = np.load(os.path.join(dti_dir, file_name))
                gender_data = np.load(os.path.join(gender_dir, file_name))

                # 将T1和DTI数据转换为torch.Tensor并调整维度
                t1_tensor = torch.from_numpy(t1_data).float().unsqueeze(0)
                dti_tensor = torch.from_numpy(dti_data).float().unsqueeze(0)
                gender_tensor = torch.from_numpy(gender_data).float().unsqueeze(0)

                # 通过VGG和MLP处理数据
                t1_output = self.vgg(t1_tensor)
                dti_output = self.vgg(dti_tensor)
                gender_output = self.mlp(gender_tensor)

                # 拼接输出
                combined_output = torch.cat([t1_output, dti_output, gender_output], dim=1)

                # 构建节点特征
                x_list = combined_output.squeeze().tolist()
                x_tensor = torch.tensor(x_list, dtype=torch.float).unsqueeze(-1)

                # 计算互信息（这里可以根据实际情况修改）
                mutual_infos = {}
                offset = 0
                for i in range(len(self.groups)):
                    _x = _y = x_list[offset: offset + self.groups[i]]
                    _x = np.reshape(_x, (-1, 1))
                    mutual_info = mutual_info_regression(_x, _y)[0]
                    mutual_infos[f"{i}_{i}"] = mutual_info

                offset = 0
                for i in range(len(self.groups)):
                    offset_2 = offset + self.groups[i]
                    for j in range(i + 1, len(self.groups)):
                        _x = x_list[offset: offset + self.groups[i]]
                        _y = x_list[offset_2: offset_2 + self.groups[j]]
                        min_length = min(len(_x), len(_y))
                        _x = np.reshape(_x[:min_length], (-1, 1))
                        mutual_info = mutual_info_regression(_x, _y[:min_length])[0]
                        mutual_infos[f"{i}_{j}"] = mutual_info
                        offset_2 += self.groups[j]
                    offset += self.groups[i]

                mutual_infos_smooth = {key: 1 / (1 + np.exp(-value)) for key, value in mutual_infos.items()}
                mutual_infos_smooth = {key: value if value > 0.1 else 0.1 for key, value in mutual_infos_smooth.items()}

                # 构建边索引
                edges = []
                edge_attrs = []
                if self.mode == 'FC':
                    for i in range(sum(self.groups)):
                        for j in range(sum(self.groups)):
                            if i != j:
                                edges.append((i, j))
                                edge_attrs.append([0.5])

                elif self.mode == 'random':
                    # 组内全连接
                    offset = 0
                    index = 0
                    for num in self.groups:
                        for i in range(num):
                            for j in range(num):
                                if i + offset != j + offset:
                                    edges.append((i + offset, j + offset))
                                    edge_attrs.append([mutual_infos_smooth[f"{index}_{index}"]])
                        offset += num
                        index += 1
                    # 组间随机连接
                    offset = 0
                    for i in range(5):
                        offset_2 = self.groups[i]
                        for j in range(i + 1, 5):
                            for _i in range(self.groups[i]):
                                for _j in range(self.groups[j]):
                                    if _i + offset != _j + offset_2 and random.randrange(1, 101, 1) < 5:
                                        edges.append((_i + offset, _j + offset_2))
                                        edge_attrs.append([mutual_infos_smooth[f"{i}_{j}"]])
                            offset_2 += self.groups[j]
                        offset += self.groups[i]

                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
                edge_attrs = torch.tensor(edge_attrs, dtype=torch.float)

                # 这里假设标签为0，实际中需要根据数据集进行修改
                y = torch.tensor(0, dtype=torch.long)

                # 创建图数据
                data = Data(x=x_tensor, edge_index=edge_index, edge_attr=edge_attrs, y=y)

                # 添加到数据列表
                data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

if __name__ == "__main__":
    dataset = MyDataset(

    )

