import random

import torch
from torch import nn
import numpy as np

class Net(nn.Module):
    """
    Net模型类，继承自nn.Module
    """

    def __init__(self, state_dim, action_dim, hidden_size=(256, 256)):
        """
        Net模型的初始化方法
        参数:
            - state_dim: 状态的维度
            - action_dim: 动作的维度
            - hidden_size: 隐藏层大小，默认为(256, 256)
        """
        super(Net, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_size[0])    # 第一个全连接层
        self.fc1.weight.data.normal_(0, 0.1)    # 初始化第一个全连接层的权重
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])    # 第二个全连接层
        self.fc2.weight.data.normal_(0, 0.1)    # 初始化第二个全连接层的权重
        self.out = nn.Linear(hidden_size[1], action_dim)    # 输出层
        self.out.weight.data.normal_(0, 0.1)    # 初始化输出层的权重
        self.state_dim = state_dim    # 状态的维度
        self.action_dim = action_dim    # 动作的维度

    def forward(self, x):
        """
        Net的前向传播
        参数:
            - x: 输入的状态
        返回:
            - 动作的概率分布
        """
        x = self.fc1(x) # 第一个全连接层
        x = torch.relu(x)    # 使用ReLU激活函数
        x = self.fc2(x) # 第二个全连接层
        x = torch.relu(x)    # 使用ReLU激活函数
        action_prob = self.out(x)    # 输出层，得到动作的概率分布
        return action_prob

    def select_action(self, state, epsilon):
        """
        根据当前状态选择动作和相应的动作值函数
        参数:
            - state: 当前状态
            - epsilon: 探索率
        返回:
            - 选择的动作和相应的动作值函数
        """
        action_value = self.forward(state)
        if random.random() < epsilon:    # 使用epsilon-greedy策略
            action = torch.max(action_value, 1)[1].data.numpy()
        else:
            action = np.random.randint(0, self.action_dim)    # 随机选择一个动作
            action = np.array(action).reshape(1)
        return action, action_value


class DuelingDQN(nn.Module):
    """
    DuelingDQN模型类，继承自nn.Module
    """

    def __init__(self, state_dim, action_dim, hidden_size=(256, 256)):
        """
        DuelingDQN模型的初始化方法
        参数:
            - state_dim: 状态的维度
            - action_dim: 动作的维度
            - hidden_size: 隐藏层大小，默认为(256, 256)
        """
        super(DuelingDQN, self).__init__()

        # 特征提取部分
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_size[0]),    # 第一个全连接层
            nn.LeakyReLU()    # 使用Leaky ReLU作为激活函数
        )

        # 优势估计部分
        self.advantage = nn.Sequential(
            nn.Linear(hidden_size[0], hidden_size[1]),    # 第二个全连接层
            nn.LeakyReLU(),    # 使用Leaky ReLU作为激活函数
            nn.Linear(hidden_size[1], action_dim)    # 输出层，输出动作的Q值
        )

        # 状态值估计部分
        self.value = nn.Sequential(
            nn.Linear(hidden_size[0], hidden_size[1]),    # 第三个全连接层
            nn.LeakyReLU(),    # 使用Leaky ReLU作为激活函数
            nn.Linear(hidden_size[1], 1)    # 输出层，输出状态的值
        )

        self.state_dim = state_dim    # 状态的维度
        self.action_dim = action_dim    # 动作的维度

    def forward(self, x):
        """
        DuelingDQN的前向传播
        参数:
            - x: 输入的状态
        返回:
            - Q值，即状态和动作的值函数
        """
        x = self.feature(x)    # 特征提取部分
        advantage = self.advantage(x)    # 优势估计部分
        value = self.value(x)    # 状态值估计部分
        return value + advantage - advantage.mean()    # 返回Q值

    def select_action(self, state, epsilon):
        """
        根据当前状态选择动作
        参数:
            - state: 当前状态
            - epsilon: 探索率
        返回:
            - 选择的动作和相应的Q值
        """
        action_value = self.forward(state)
        if random.random() > epsilon:    # 使用epsilon-greedy策略
            action = torch.max(action_value, 1)[1].data.numpy()
        else:
            action = np.random.randint(0, self.action_dim)    # 随机选择一个动作
            action = np.array(action).reshape(1)
        return action, action_value