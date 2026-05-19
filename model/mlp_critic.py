import torch.nn as nn
import torch


class Value(nn.Module):
    def __init__(self, state_dim, hidden_size=(128, 128), activation='tanh'):
        super().__init__()

        # 指定激活函数
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        # 创建线性层列表，用于设置全连接层
        self.affine_layers = nn.ModuleList()
        last_dim = state_dim
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        # 创建值估计头部的线性层
        self.value_head = nn.Linear(last_dim, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, x):
        """
        前向传播函数
        :param x: 输入状态向量
        :return: 值估计结果
        """
        # 通过激活函数和全连接层进行前向传播
        for affine in self.affine_layers:
            x = self.activation(affine(x))

        # 通过值估计头部进行输出
        value = self.value_head(x)
        return value