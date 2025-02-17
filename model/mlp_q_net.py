import torch
import torch.nn as nn
import torch.nn.functional as F

class Q(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=(256, 256), activation='tanh'):
        super(Q, self).__init__()
        # 初始化Q网络
        # state_dim: 状态维度
        # action_dim: 行动维度
        # hidden_size: 隐藏层维度，默认为(256, 256)
        # activation: 激活函数，默认为tanh函数
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        # 全连接层
        # 将状态和行动进行连接，得到输入
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], 1)
        # 对fc3的权重进行缩放和偏置进行设置
        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)

    def forward(self, s, a):
        # 前向传播
        # 输入s和a，通过前向传播得到Q值
        x = torch.cat((s, a), -1)  # 拼接s和a
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x