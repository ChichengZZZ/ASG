import torch.nn as nn
import torch
from torch.distributions import Normal
from Utils.math import *  # 导入自定义的数学函数
from fvcore.nn import FlopCountAnalysis, parameter_count_table  # 导入FLOPs分析和参数数量统计的工具
import scipy.stats as st
# st.norm.pdf()


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=(128, 128), activation='tanh'):
        super().__init__()
        self.is_disc_action = False  # 是否离散行动，默认为False
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        elif activation == 'mish':  # nn.Mish()
            self.activation = torch.nn.functional.mish

        self.affine_layers = nn.ModuleList()  # 线性变换层的列表
        last_dim = state_dim
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))  # 添加一个线性变换层
            last_dim = nh

        self.action_mean = nn.Linear(last_dim, action_dim)  # 行动的均值估计线性层
        self.action_mean.weight.data.mul_(0.1)  # 初始化权重
        self.action_mean.bias.data.mul_(0.0)  # 初始化偏置

        self.action_std = nn.Linear(last_dim, action_dim)  # 行动的标准差估计线性层
        self.action_std.weight.data.mul_(0.1)  # 初始化权重
        self.action_std.bias.data.mul_(0.0)  # 初始化偏置

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))

        action_mean = self.action_mean(x)  # 计算行动均值
        action_std = torch.exp(self.action_std(x))  # 计算行动标准差
        return action_mean, action_std

    def select_action(self, x):
        action_mean, action_std = self.forward(x)  # 前向传播计算均值和标准差
        dist = Normal(action_mean, action_std)  # 创建一个正态分布对象
        z = dist.sample()  # 从正态分布中采样得到z
        action = torch.tanh(z)  # 将z映射到[-1,1]的范围得到行动
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)  # 计算对数概率
        return action, action_mean, action_std, log_prob

    def get_action_log_prob(self, state):
        batch_mu, batch_log_sigma = self.forward(state)  # 前向传播计算均值和标准差
        batch_sigma = torch.exp(batch_log_sigma)  # 计算标准差
        dist = Normal(batch_mu, batch_sigma)  # 创建一个正态分布对象
        z = dist.sample()  # 从正态分布中采样得到z
        action = torch.tanh(z)  # 将z映射到[-1,1]的范围得到行动
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)  # 计算对数概率
        return action, torch.sum(log_prob, dim=1, keepdim=True), z, batch_mu, batch_log_sigma


if __name__ == "__main__":
    import time
    input = torch.randn((3, 10))
    print(input.size())
    t0 = time.time()
    policy = Policy(state_dim=10, action_dim=2)  # 创建Policy对象
    print(time.time()-t0)
    # 分析FLOPs
    flops = FlopCountAnalysis(policy, input)  # 计算FLOPs
    print("FLOPs: ", flops.total())
    print(parameter_count_table(policy))  # 输出模型参数数量表
    total = sum([param.nelement() for param in policy.parameters()])  # 计算总的参数数量
    print("Number of parameter: %.2fk" % (total / 1e3))
    action, action_mean, action_std, action_log_prob = policy.select_action(input)  # 选择行动
    print(policy.select_action(input))  # 打印选择行动的结果
    print(torch.exp(action_log_prob))  # 打印行动的指数概率
    print(torch.exp(action_log_prob[:, 0]) * torch.exp(action_log_prob[:, 1]))  # 打印行动的指数概率乘积
    a = policy.get_action_log_prob(input)  # 获取行动和对数概率的结果
    print("")
    # 12th Gen Intel® Core™ i7-12700K × 20 0.5ms, 17920 18.18k