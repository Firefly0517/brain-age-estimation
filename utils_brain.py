import os
import sys
import time
from sklearn.model_selection import KFold
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from tqdm import tqdm
#from graph_dataset import MyDataset
from dataset import MyDataset
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from observer import Runtime_Observer

from mamba_ssm import Mamba


class Mamba_in(torch.nn.Module):
    '''
    # d_model * expand / headdim = multiple of 8
    '''

    def __init__(self, in_channels, d_state=16, d_conv=4, headdim=1):
        super().__init__()
        expand = 8 * headdim / in_channels
        self.mamba = Mamba(d_model=in_channels, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x):
        '''
        x: (B, L, C)
        '''
        return self.mamba(x)


class Mamba_node(torch.nn.Module):
    '''
    # d_model * expand / headdim = multiple of 8
    '''

    def __init__(self, in_channels, d_state=64, d_conv=4):
        super().__init__()
        self.mamba = Mamba(d_model=in_channels, d_state=d_state, d_conv=d_conv, expand=2, headdim=2)
        self.dp = nn.Dropout(0.25)

    def forward(self, x):
        '''
        x: (node_nums, node_features)
        '''
        srcx = x
        x = x.view(1, -1, x.size(-1))
        x = self.mamba(x)
        x = x.view(-1, x.size(-1)) + srcx
        x = self.dp(x)
        return x


class GraphMedMamba(nn.Module):
    def __init__(self, num_nodes, node_in_channels=1, edge_in_channels=1, hidden_scale=2, dropout_p=0.25, depth=4):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_in_channels = node_in_channels
        self.edge_in_channels = edge_in_channels
        self.hidden_scale = hidden_scale
        self.dropout_p = dropout_p

        # self.edge_embed = nn.Embedding((num_nodes + 1) ** 2, hidden_scale)
        self.GNNlayers = nn.ModuleList(
            [MambaLayer(node_in_channels=node_in_channels, edge_in_channels=hidden_scale + edge_in_channels) for _ in
             range(depth)])
        self.activate = nn.LeakyReLU()
        self.node_norm = nn.LayerNorm(node_in_channels)
        self.edge_norm = nn.LayerNorm(hidden_scale + edge_in_channels)

        self.drop = nn.Dropout(dropout_p)
        self.lin_edge = nn.Linear(hidden_scale + edge_in_channels, 1)
        self.classifier = nn.Linear(num_nodes * node_in_channels, 2)

        self.apply(self._initialize_weights)

    def _initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            print(f'layer {m} initialized')
            nn.init.xavier_uniform_(m.weight)  # Xavier initialization
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, edge_index, edge_attr=None):
        # 获得位置索引
        edge_embed = edge_index.view(-1, 2)
        edge_attr_pos = torch.cat([edge_embed, edge_attr], dim=1)
        for layer in self.GNNlayers:
            x, edge_attr_pos = self.node_norm(x), self.edge_norm(edge_attr_pos)
            x, edge_attr_pos = layer(x, edge_index, edge_attr_pos)
            x = self.activate(x)

        _out_x = self.drop(x.view(-1))
        out = self.classifier(_out_x)
        edge_attr_pos = self.lin_edge(edge_attr_pos)

        return out, x, edge_index, edge_attr_pos


class MambaLayer(MessagePassing):
    def __init__(self, node_in_channels, edge_in_channels, dropout_p=0.25, aggr='mean'):
        super().__init__(aggr=aggr)  # 使用平均聚合

        # basic settings
        self.node_in_channels = node_in_channels
        self.edge_in_channels = edge_in_channels
        # Module Settings
        self.mamba_node = Mamba_in(in_channels=node_in_channels)  # 用于节点特征的 Mamba 层
        self.mamba_edge = Mamba_in(in_channels=edge_in_channels)  # 用于边特征的 Mamba 层
        self.mamba_out = Mamba_in(in_channels=node_in_channels)  # 输出 Mamba 层
        self.mamba_inner = Mamba_in(in_channels=1)  # 用于内部计算的 Mamba 层

        self.lin_node = nn.Linear(edge_in_channels, node_in_channels)
        self.edge_weight_scale = nn.Linear(edge_in_channels, 1)
        self.edge_weight_smooth = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, edge_index, edge_attr=None):
        '''
        Args:
            x: (num_nodes, node_in_channels) - 节点特征
            edge_index: (2, num_edges) - 边索引
            edge_attr: (num_edges, edge_in_channels) - 边特征（可选）
        Returns:
            Tensor: 更新后的节点特征
        '''
        src_node, src_edge_attr = self.dropout(x), self.dropout(edge_attr)
        # 处理节点特征
        x = x.view(1, -1, self.node_in_channels)
        x = self.mamba_node(x).view(-1, self.node_in_channels)
        # 处理边特征
        if edge_attr is not None:
            edge_attr = edge_attr.view(1, -1, self.edge_in_channels)
            edge_attr = self.mamba_edge(edge_attr).view(-1, self.edge_in_channels)
        # 位置索引 ->  自适应权重
        scaled_edge_weight = self.edge_weight_scale(src_edge_attr)
        scaled_edge_weight_s = self.edge_weight_smooth(scaled_edge_weight)
        # 进行消息传递
        out_x = self.propagate(edge_index, x=x, edge_attr=edge_attr, edge_weight=scaled_edge_weight_s)
        out_x = self.lin_node(out_x)
        # 使用 Mamba 输出层处理最终输出
        out_x = self.mamba_out(out_x.view(1, -1, 1)).view(-1, self.node_in_channels)

        return out_x + src_node, edge_attr + src_edge_attr

    def message(self, x_j, edge_attr=None, edge_weight=None):
        '''
        Args:
            x_j: (num_edges, out_channels) - 每条边的源节点特征
            edge_attr: (num_edges, out_channels) - 每条边的边特征
            edge_weight: (num_edges,) - 可学习的自适应边权重
        Returns:
            Tensor: 经过边特征调节后的源节点特征
        '''
        edge_attr = self.dropout(edge_attr)
        if edge_attr is not None:
            x_j = x_j + edge_attr  # 调整节点特征

        # 动态调整边权重
        if edge_weight is not None:
            x_j = x_j * edge_weight.view(-1, 1)
        return x_j

    def debug_tensor(self, x):
        print(x.shape)
        print(x)


class EdgeWeightContrastiveLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, node_features, edge_index, edge_weight):
        # 计算边的端点节点特征差异
        src, dst = edge_index
        edge_diff = torch.norm(node_features[src] - node_features[dst], p=2, dim=1)
        # 对比损失：使得边权重与特征相似度相反
        return torch.mean((edge_weight * edge_diff) ** 2)


class EdgeWeightEntropyRegularization(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, edge_weight):
        # 计算边权重的概率分布
        weight_prob = F.softmax(edge_weight, dim=0)
        # 负熵，促使权重分布均匀
        entropy = -torch.sum(weight_prob * torch.log(weight_prob + 1e-10))
        return entropy


class MixLoss(nn.Module):
    def __init__(self, w1=0.2, w2=0.3):
        super().__init__()
        self.contrastive_loss = EdgeWeightContrastiveLoss()
        self.entropy_reg = EdgeWeightEntropyRegularization()
        self.ce_loss = nn.CrossEntropyLoss()
        self.w1 = w1
        self.w2 = w2

    def forward(self, pred, node_features, edge_index, edge_weight, label):
        # 计算对比损失
        contrastive_loss = self.contrastive_loss(node_features, edge_index, edge_weight)
        # 计算熵正则化损失
        entropy_reg_loss = self.entropy_reg(edge_weight)
        # 计算交叉熵损失
        ce_loss = self.ce_loss(pred, label)

        return self.w1 * contrastive_loss + self.w2 * entropy_reg_loss + ce_loss


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    kf = KFold(n_splits=5, shuffle=True, random_state=666)
    torch.manual_seed(666)

    for i in range(5):
        # 初始化数据集
        dataset = MyDataset(root = "/data2/wangchangmiao/liuxiaoshuai/IXI", fold=i)
        train_index, test_index = [[t1, t2] for t1, t2 in kf.split(dataset)][i]

        # 设置日志目录
        if not os.path.exists(f"debug/test{i}"):
            os.makedirs(f"debug/test{i}")
        observer = Runtime_Observer(log_dir=f"debug/test{i}", device=device, name="debug", seed=666)

        # 使用Subset获取训练集和测试集
        train_dataset = torch.utils.data.Subset(dataset, train_index)
        test_dataset = torch.utils.data.Subset(dataset, test_index)
        #train_loader = DataLoader(torch.utils.data.Subset(dataset, train_index), batch_size=4, shuffle=True,
                                  #num_workers=2, pin_memory=True)
        #test_loader = DataLoader(torch.utils.data.Subset(dataset, test_index), batch_size=4, shuffle=False,
                                 #num_workers=2, pin_memory=True)
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=1, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=1, pin_memory=True)

        # 定义模型和优化器
        model = GraphMedMamba(node_in_channels=1, edge_in_channels=1, num_nodes=78).to(device)
        optimizer = Adam(model.parameters(), lr=0.0001)

        # 使用MSELoss进行回归任务的损失计算
        criterion = nn.MSELoss()

        # 统计模型参数数量
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        observer.log("\n===============================================\n")
        observer.log("model parameters: " + str(num_params))
        observer.log("\n===============================================\n")

        epochs = 200
        model = model.to(device)
        observer.log("start training\n")
        start_time = time.time()


        # 训练过程
        for epoch in range(epochs):
            print(f"Epoch: {epoch + 1}/{epochs}")

            observer.reset()
            model.train()
            train_bar = tqdm(train_loader, leave=True, file=sys.stdout)  # ✅ 改为 train_loader
            running_loss = 0.0

            for batch in train_bar:
                optimizer.zero_grad()
                x, edge_index, edge_attr, y = batch.x.to(device), batch.edge_index.to(device), batch.edge_attr.to(
                    device), batch.y.to(device)

                # 前向传播
                out, out_x, out_edge_index, out_scaled_edge_weight = model(x, edge_index, edge_attr)

                # 计算损失（回归任务使用 MSELoss）
                loss = criterion(out, y)  # ✅ 不再需要 unsqueeze(0)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            # 计算平均训练损失
            train_loss = running_loss / len(train_loader)
            observer.log(f"Train Loss: {train_loss:.4f}\n")

            # 测试过程
            with torch.no_grad():
                model.eval()
                test_bar = tqdm(test_loader, leave=True, file=sys.stdout)  # ✅ 改为 test_loader
                test_loss = 0.0

                for batch in test_bar:
                    x, edge_index, edge_attr, y = batch.x.to(device), batch.edge_index.to(device), batch.edge_attr.to(
                        device), batch.y.to(device)

                    out, out_x, out_edge_index, out_scaled_edge_weight = model(x, edge_index, edge_attr)

                    loss = criterion(out, y)  # ✅ 不再需要 unsqueeze(0)
                    test_loss += loss.item()

                test_loss = test_loss / len(test_loader)
                observer.log(f"Test Loss: {test_loss:.4f}\n")

            # 记录损失和提前停止
            observer.record_loss(epoch, train_loss, test_loss)
            if observer.excute(epoch):
                print("Early stopping")
                break

        end_time = time.time()
        observer.log(f"\nRunning time: {end_time - start_time:.2f} seconds\n")
        observer.finish()


'''
        # 训练过程
        for epoch in range(epochs):
            print(f"Epoch: {epoch + 1}/{epochs}")

            observer.reset()
            model.train()
            train_bar = tqdm(train_dataset, leave=True, file=sys.stdout)
            running_loss = 0.0
            for i, (x, edge_index, edge_attr, y) in enumerate(train_bar):
                optimizer.zero_grad()
                x, edge_index, edge_attr, y = x[1].to(device), edge_index[1].to(device), edge_attr[1].to(device), y[
                    1].to(device)

                # 前向传播
                out, out_x, out_edge_index, out_scaled_edge_weight = model(x, edge_index, edge_attr)

                # 计算损失（回归任务使用MSELoss）
                loss = criterion(out.unsqueeze(0), y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            # 平均训练损失
            train_loss = running_loss / len(train_dataset.dataset)
            observer.log(f"Train Loss: {train_loss:.4f}\n")

            # 测试过程
            with torch.no_grad():
                model.eval()
                test_bar = tqdm(test_dataset, leave=True, file=sys.stdout)
                test_loss = 0.0

                for i, (x, edge_index, edge_attr, y) in enumerate(test_bar):
                    x, edge_index, edge_attr, y = x[1].to(device), edge_index[1].to(device), edge_attr[1].to(device), y[
                        1].to(device)

                    out, out_x, out_edge_index, out_scaled_edge_weight = model(x, edge_index, edge_attr)

                    # 计算测试损失
                    loss = criterion(out.unsqueeze(0), y)
                    test_loss += loss.item()

                test_loss = test_loss / len(test_dataset.dataset)
                observer.log(f"Test Loss: {test_loss:.4f}\n")
        '''
