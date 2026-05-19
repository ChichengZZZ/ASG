import torch.nn as nn
import torch
from torch.distributions import Normal
from Utils.math import *  # 导入自定义的数学函数
from fvcore.nn import FlopCountAnalysis, parameter_count_table  # 导入FLOPs分析和参数数量统计的工具
import scipy.stats as st
# st.norm.pdf()


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_size=(400, 300), activation='tanh'):
        super().__init__()
        self.is_disc_action = False  # 是否离散行动，默认为False
        if activation == 'tanh':
            self.activation = torch.nn.Tanh()  # Tanh激活函数
        elif activation == 'relu':
            self.activation = torch.relu  # ReLU激活函数
        elif activation == 'leakyrelu':
            self.activation = torch.nn.functional.leaky_relu  # LeakyReLU激活函数
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid  # Sigmoid激活函数
        elif activation == 'mish':  # nn.Mish()
            self.activation = torch.nn.functional.mish  # Mish激活函数

        self.max_action = max_action  # 行动的最大取值范围
        self.state_dim = state_dim  # 状态维度
        self.action_dim = action_dim  # 行动维度
        self.affine_layers = nn.ModuleList()  # 线性变换层的列表
        last_dim = state_dim
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))  # 添加一个线性变换层
            last_dim = nh

        self.action_head = nn.Linear(last_dim, action_dim)  # 行动的线性层

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))  # 使用激活函数进行线性变换

        action = self.activation(self.action_head(x)) * self.max_action  # 计算行动，乘以最大取值范围进行缩放
        return action

    def select_action(self, x):
        action = self.forward(x)  # 前向传播计算行动
        return action, torch.zeros((1, self.action_dim)), torch.zeros((1, self.action_dim)), torch.zeros((1, self.action_dim))


if __name__ == "__main__":
    import time
    input = torch.randn((3, 10))
    print(input.size())
    t0 = time.time()
    max_cation = torch.from_numpy(np.array([1, 1]).reshape((-1, 2)))  # 最大取值范围
    policy = Policy(state_dim=10, action_dim=2, max_action=max_cation)  # 创建Policy对象
    print(time.time()-t0)
    # 分析FLOPs
    flops = FlopCountAnalysis(policy, input)  # 计算FLOPs
    print("FLOPs: ", flops.total())
    print(parameter_count_table(policy))  # 输出模型参数数量表
    total = sum([param.nelement() for param in policy.parameters()])  # 计算总的参数数量
    print("Number of parameter: %.2fk" % (total / 1e3))
    action, action_mean, action_std, action_log_prob = policy.select_action(input)  #