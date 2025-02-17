import torch.nn as nn
import torch
from Utils.math import *
import scipy.stats as st
# st.norm.pdf() 表示导入scipy.stats模块中的概率密度函数norm.pdf()
import sys

class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=(128, ), activation='tanh', log_std=0):
        super().__init__()
        self.is_disc_action = False
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2),  # 输入通道为3，输出通道为64，卷积核大小为3，步长为2
            nn.BatchNorm2d(64),  # 标准化操作，Batch Normalization
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.MaxPool2d(kernel_size=3, stride=2),  # 最大池化层，pooling核大小为3， 步长为2
            nn.Conv2d(64, 128, kernel_size=3, stride=2),  # 输入通道为64，输出通道为128，卷积核大小为3，步长为2
            nn.BatchNorm2d(128),  # 标准化操作，Batch Normalization
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.Conv2d(128, 256, kernel_size=3, stride=2),  # 输入通道为128，输出通道为256，卷积核大小为3，步长为2
            nn.BatchNorm2d(256),  # 标准化操作，Batch Normalization
            nn.AvgPool2d((5, 8)),  # 平均池化层，pooling核大小(5,8)
            # nn.MaxPool2d(kernel_size=3, stride=2),  # 最大池化层，pooling核大小为3， 步长为2
        )

        self.affine_layers = nn.ModuleList()  # 定义线性层列表
        last_dim = 256  # 上一层输出通道数为256
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))  # 添加线性层，输入维度为last_dim，输出维度为nh
            last_dim = nh  # 更新last_dim为输出维度

        self.action_mean = nn.Linear(last_dim, action_dim)  # 动作均值预测，输入维度为last_dim， 输出维度为action_dim
        self.action_mean.weight.data.mul_(0.1)  # 权重初始化
        self.action_mean.bias.data.mul_(0.0)  # bias初始化

        self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * log_std)  # 动作的对数标准差， 是个可训练的参数

    def forward(self, x):
        x = self.features(x)  # CNN特征提取
        x = x.view(x.size(0), -1)  # 展平特征图
        print(x.size())

        for affine in self.affine_layers:
            x = self.activation(affine(x))  # 进行激活操作

        action_mean = self.action_mean(x)  # 计算动作均值
        # print('self.action_log_std', self.action_log_std)
        action_log_std = self.action_log_std.expand_as(action_mean)  # 平铺动作对数标准差

        action_std = torch.exp(action_log_std)  # 计算动作标准差

        return action_mean, action_log_std, action_std  # 返回动作均值，动作对数标准差和动作标准差

    def select_action(self, x):
        action_mean, _, action_std = self.forward(x)  # 调用forward函数
        action = torch.normal(action_mean, action_std)  # 从均值为action_mean，标准差为action_std的正态分布中采样
        # action
        return action  # 返回动作

    def get_kl(self, x):
        mean1, log_std1, std1 = self.forward(x)  # 调用forward函数

        mean0 = mean1.detach()  # 分离计算图并返回mean1的副本
        log_std0 = log_std1.detach()  # 分离计算图并返回log_std1的副本
        std0 = std1.detach()  # 分离计算图并返回std1的副本
        # print(mean1, log_std1, std1)
        # print(mean0, log_std0, std0)
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5  # 计算KL散度
        return kl.sum(1, keepdim=True)  # 返回KL散度

    def get_log_prob(self, x, actions):
        action_mean, action_log_std, action_std = self.forward(x)  # 调用forward函数
        return normal_log_density(actions, action_mean, action_log_std, action_std)  # 返回动作的对数概率密度

    def get_fim(self, x):
        mean, _, _ = self.forward(x)  # 调用forward函数
        cov_inv = self.action_log_std.exp().pow(-2).squeeze(0).repeat(x.size(0))  # 计算协方差矩阵的逆
        param_count = 0
        std_index = 0
        id = 0
        for name, param in self.named_parameters():
            if name == "action_log_std":
                std_id = id
                std_index = param_count
            param_count += param.view(-1).shape[0]  # 加上参数的数量
            id += 1
        return cov_inv.detach(), mean, {'std_id': std_id, 'std_index': std_index}

if __name__ == "__main__":
    input = torch.randn((1, 3, 100, 150))  # 生成一个大小为(1,3,100,150)的张量
    print(input.size())  # 打印张量的大小
    policy = Policy(state_dim=32, action_dim=2)  # 实例化Policy类
    print(policy(input))  # 调用policy的forward函数输出结果