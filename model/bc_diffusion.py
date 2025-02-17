import copy  # 导入copy模块用于复制对象
import numpy as np  # 导入numpy库并使用别名np
import torch  # 导入torch库
import torch.nn as nn  # 导入torch的神经网络模块并使用别名nn
import torch.nn.functional as F  # 导入torch的函数模块并使用别名F
from Utils.logger import logger  # 导入自定义的logger类
from models.diffusion import Diffusion  # 导入自定义的Diffusion类
from models.network_blocks import MLP  # 导入自定义的MLP类


class Diffusion_BC(object):  # 定义名为Diffusion_BC的类
    def __init__(self,  # 类的初始化方法，接收以下参数
                 state_dim,  # 状态维度
                 action_dim,  # 动作维度
                 max_action,  # 最大动作值
                 device,  # 设备（CPU或GPU）
                 discount,  # 折扣因子
                 tau,  # 软更新的目标网络权重参数
                 clip_denoised=True,  # 是否剪切去噪过程中的动作值
                 beta_schedule='linear',  # 扩散过程中的beta值调度方式（线性或恒定）
                 n_timesteps=100,  # 扩散过程的时间步数
                 lr=2e-4,  # 学习率
                 ):

        self.model = MLP(state_dim=state_dim, action_dim=action_dim, device=device, t_dim=n_timesteps)
        # 创建一个MLP对象，用于近似State-Action值函数
        self.actor = Diffusion(state_dim=state_dim, action_dim=action_dim, model=self.model, max_action=max_action,
                               beta_schedule=beta_schedule, n_timesteps=n_timesteps,clip_denoised=clip_denoised
                               ).to(device)
        # 创建Diffusion对象，用于生成扩散过程中的动作值
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        # 创建Adam优化器，用于更新actor的参数
        self.policy_step_lr = torch.optim.lr_scheduler.CosineAnnealingLR(self.actor_optimizer, 1000, 1e-4)
        # 创建学习率调度器，用于调整优化器的学习率
        self.max_action = max_action  # 最大动作值
        self.action_dim = action_dim  # 动作维度
        self.discount = discount  # 折扣因子
        self.tau = tau  # 软更新的目标网络权重参数
        self.device = device  # 设备（CPU或GPU）

    def train(self, replay_buffer, batch_size=100, log_writer=None):
        # 训练方法，接收以下参数
        metric = {'bc_loss': [], 'ql_loss': [], 'actor_loss': [], 'critic_loss': []}
        # 创建一个字典，用于保存各项指标的损失值
        for i in range(len(replay_buffer) // batch_size):  # 对于每个batch
            state, action = replay_buffer.sample(i * batch_size, batch_size)
            # 从replay_buffer中采样获得一个batch的状态和动作
            loss = self.actor.loss(action, state)
            # 计算actor的损失值

            self.actor_optimizer.zero_grad()
            # 清零优化器的梯度
            loss.backward()
            # 反向传播计算梯度
            self.actor_optimizer.step()
            # 使用优化器更新actor的参数

            metric['actor_loss'].append(0.)
            metric['bc_loss'].append(loss.item())
            metric['ql_loss'].append(0.)
            metric['critic_loss'].append(0.)
            # 将损失值添加到指标的损失列表中

        self.policy_step_lr.step()
        # 更新学习率调度器的状态

        return metric
        # 返回指标的损失列表

    def sample_action(self, state):
        # 采样动作方法，接收以下参数
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        # 将状态转换为torch的浮点张量，并发送到指定设备上
        with torch.no_grad():
            action = self.actor.sample(state)
        # 使用actor网络生成一个动作
        return action.cpu().data.numpy().flatten()
        # 返回生成的动作值（将其从张量转换为numpy数组）

    def save_model(self, dir, id=None):
        # 保存模型方法，接收以下参数
        if id is not None:
            torch.save(self.actor.state_dict(), f'{dir}/actor_{id}.pth')
        else:
            torch.save(self.actor.state_dict(), f'{dir}/actor.pth')
        # 根据是否提供id参数，保存actor模型的状态字典

    def load_model(self, dir, id=None):
        # 加载模型方法，接收以下参数
        if id is not None:
            self.actor.load_state_dict(torch.load(f'{dir}/actor_{id}.pth'))
        else:
            self.actor.load_state_dict(torch.load(f'{dir}/actor.pth'))
        # 根据是否提供id参数，加载actor模型的状态字典