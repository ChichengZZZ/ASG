import torch.nn as nn
import torch
from Utils.math import *


class DiscretePolicy(nn.Module):
    def __init__(self, state_dim, action_num, hidden_size=(128, 128), activation='tanh'):
        super().__init__()
        self.is_disc_action = True
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.affine_layers = nn.ModuleList()
        last_dim = state_dim
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.action_head = nn.Linear(last_dim, action_num)
        self.action_head.weight.data.mul_(0.1)
        self.action_head.bias.data.mul_(0.0)

    def forward(self, x):
        """
        前向传播函数

        Args:
            x (tensor): 输入张量

        Returns:
            tensor: 动作概率张量
        """
        for affine in self.affine_layers:
            x = self.activation(affine(x))

        action_prob = torch.softmax(self.action_head(x), dim=1)
        return action_prob

    def select_action(self, x):
        """
        随机选择动作

        Args:
            x (tensor): 输入张量

        Returns:
            tuple: 包含选择的动作和动作概率张量的元组
        """
        action_prob = self.forward(x)
        action = action_prob.multinomial(1)
        return action, action_prob

    def get_kl(self, x):
        """
        计算KL散度

        Args:
            x (tensor): 输入张量

        Returns:
            tensor: KL散度张量
        """
        action_prob1 = self.forward(x)
        action_prob0 = action_prob1.detach()
        kl = action_prob0 * (torch.log(action_prob0) - torch.log(action_prob1))
        return kl.sum(1, keepdim=True)

    def get_log_prob(self, x, actions):
        """
        计算动作的对数概率

        Args:
            x (tensor): 输入张量
            actions (tensor): 选择的动作

        Returns:
            tensor: 对数概率张量
        """
        action_prob = self.forward(x)
        return torch.log(action_prob.gather(1, actions.long().unsqueeze(1)))

    def get_fim(self, x):
        """
        获取Fisher信息矩阵

        Args:
            x (tensor): 输入张量

        Returns:
            tuple: 包含Fisher信息矩阵逆矩阵、动作概率张量和空字典的元组
        """
        action_prob = self.forward(x)
        M = action_prob.pow(-1).view(-1).detach()
        return M, action_prob, {}