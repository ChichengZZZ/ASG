import torch.nn as nn
import torch
from torch.distributions import Normal
from Utils.math import *
from fvcore.nn import FlopCountAnalysis, parameter_count_table
import scipy.stats as st
# st.norm.pdf()


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=(128, 128), activation='tanh'):
        super().__init__()
        self.is_disc_action = False

        # 激活函数可选项
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        elif activation == 'mish':# nn.Mish()
            self.activation = torch.nn.functional.mish

        self.affine_layers = nn.ModuleList()
        last_dim = state_dim
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        # 线性层：将上一层输出的特征转换为动作均值
        self.action_mean = nn.Linear(last_dim, action_dim)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)

        # 线性层：将上一层输出的特征转换为动作标准差
        self.action_std = nn.Linear(last_dim, action_dim)
        self.action_std.weight.data.mul_(0.1)
        self.action_std.bias.data.mul_(0.0)

    def forward(self, x):
        # 前向传播，模型的核心逻辑
        for affine in self.affine_layers:
            x = self.activation(affine(x))

        # 计算动作的均值和标准差
        action_mean = self.action_mean(x)
        action_std = torch.exp(self.action_std(x))
        return action_mean, action_std

    def select_action(self, x):
        # 根据输入的状态选择动作
        action_mean, action_std = self.forward(x)
        # 构建正态分布并从中采样得到动作
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        # 计算动作的对数概率
        action_log_prob = dist.log_prob(action)
        return action, action_mean, action_std, action_log_prob


if __name__ == "__main__":
    import time
    input = torch.randn((3, 10))
    print(input.size())
    t0 = time.time()
    policy = Policy(state_dim=10, action_dim=2)
    print(time.time()-t0)
    # 分析FLOPs
    flops = FlopCountAnalysis(policy, input)
    print("FLOPs: ", flops.total())
    print(parameter_count_table(policy))
    total = sum([param.nelement() for param in policy.parameters()])
    print("Number of parameter: %.2fk" % (total / 1e3))
    action, action_mean, action_std, action_log_prob = policy.select_action(input)
    print(policy.select_action(input))
    print(torch.exp(action_log_prob))
    print(torch.exp(action_log_prob[:, 0]) * torch.exp(action_log_prob[:, 1]))
    #12th Gen Intel® Core™ i7-12700K × 20 0.5ms, 17920 18.18k