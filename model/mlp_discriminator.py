import torch.nn as nn
import torch

class Discriminator(nn.Module):
    def __init__(self, num_inputs, hidden_size=(128, 128), activation='tanh'):
        """
        初始化判别器模型

        参数:
            num_inputs (int): 输入特征的数量
            hidden_size (tuple, optional): 隐藏层的大小，默认为(128, 128)
            activation (str, optional): 激活函数的类型，默认为'tanh'
        """
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.affine_layers = nn.ModuleList()
        last_dim = num_inputs
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.logic = nn.Linear(last_dim, 1)
        self.logic.weight.data.mul_(0.1)
        self.logic.bias.data.mul_(0.0)

    def forward(self, x):
        """
        判别器的前向传播方法

        参数:
            x (torch.Tensor): 输入数据张量

        返回:
            prob (torch.Tensor): 预测每个输入样本为真实样本的概率张量
        """
        for affine in self.affine_layers:
            x = self.activation(affine(x))

        prob = torch.sigmoid(self.logic(x))
        return prob