import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utils import SinusoidalPosEmb


class MLP(nn.Module):
    """
    多层感知机（MLP）模型类
    """
    def __init__(self,
                 state_dim,
                 action_dim,
                 device,
                 t_dim=16):
        """
        MLP模型的初始化函数
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            device: 设备信息（CPU或GPU）
            t_dim: 时间维度，默认为16
        """
        super(MLP, self).__init__()
        self.device = device

        # 时间特征的MLP
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),  # 周期性位置编码
            nn.Linear(t_dim, t_dim * 2),  # 线性层
            nn.Mish(),  # Mish激活函数
            nn.Linear(t_dim * 2, t_dim),
        )

        input_dim = state_dim + action_dim + t_dim
        # 中间层的MLP
        self.mid_layer = nn.Sequential(nn.Linear(input_dim, 256),
                                       nn.Mish(),
                                       nn.Linear(256, 256),
                                       nn.Mish(),
                                       nn.Linear(256, 256),
                                       nn.Mish())

        self.final_layer = nn.Linear(256, action_dim)

    def forward(self, x, time, state):
        """
        MLP模型的前向传播函数
        Args:
            x: 输入向量
            time: 时间向量
            state: 状态向量
        Returns:
            输出向量
        """
        t = self.time_mlp(time)
        x = torch.cat([x, t, state], dim=1)
        x = self.mid_layer(x)

        return self.final_layer(x)