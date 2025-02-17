import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)  # 计算位置编码的系数
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)  # 计算位置编码的指数部分
        emb = x[:, None] * emb[None, :]  # 用输入数据的每个位置乘以位置编码的指数部分
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # 使用正弦和余弦函数将位置编码合并起来
        return emb

#-----------------------------------------------------------------------------#
#---------------------------------- sampling ---------------------------------#
#-----------------------------------------------------------------------------#


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)  # 从张量a中提取指定索引t处的元素
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))  # 调整形状以匹配x_shape


def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)  # 在0到(steps+1)之间生成步长为1的等差序列
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2  # 计算余弦函数的α值序列
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # 归一化α值序列
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])  # 计算β值序列
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)  # 将β值限制在0和0.999之间
    return torch.tensor(betas_clipped, dtype=dtype)  # 返回β值序列的张量表示


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=2e-2, dtype=torch.float32):
    betas = np.linspace(
        beta_start, beta_end, timesteps
    )  # 在beta_start和beta_end之间生成timesteps个线性等分的β值序列
    return torch.tensor(betas, dtype=dtype)  # 返回β值序列的张量表示


def vp_beta_schedule(timesteps, dtype=torch.float32):
    t = np.arange(1, timesteps + 1)  # 在1到timesteps之间生成等差序列
    T = timesteps
    b_max = 10.
    b_min = 0.1
    alpha = np.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T ** 2)  # 计算α值序列
    betas = 1 - alpha  # 计算β值序列
    return torch.tensor(betas, dtype=dtype)  # 返回β值序列的张量表示

#-----------------------------------------------------------------------------#
#---------------------------------- losses -----------------------------------#
#-----------------------------------------------------------------------------#

class WeightedLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, targ, weights=1.0):
        '''
            pred, targ : tensor [ batch_size x action_dim ]
        '''
        loss = self._loss(pred, targ)  # 计算损失
        weighted_loss = (loss * weights).mean()  # 加权计算损失的均值
        return weighted_loss


class WeightedL1(WeightedLoss):

    def _loss(self, pred, targ):
        return torch.abs(pred - targ)  # 计算带权重的L1损失


class WeightedL2(WeightedLoss):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')  # 计算带权重的L2损失


Losses = {
    'l1': WeightedL1,
    'l2': WeightedL2,
}


class EMA():
    '''
        empirical moving average
    '''

    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)  # 更新滑动平均的参数值

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new  # 计算更新后的滑动平均值