import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from Utils.logger import logger

from models.diffusion import Diffusion
from models.network_blocks import MLP
from models.model_utils import EMA


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.q1_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),  # 全连接层，输入为状态和动作的维度，输出维度为隐藏层维度
                                      nn.Mish(),  # Mish激活函数
                                      nn.Linear(hidden_dim, hidden_dim),  # 全连接层，输入维度为隐藏层维度，输出维度为隐藏层维度
                                      nn.Mish(),  # Mish激活函数
                                      nn.Linear(hidden_dim, hidden_dim),  # 全连接层，输入维度为隐藏层维度，输出维度为隐藏层维度
                                      nn.Mish(),  # Mish激活函数
                                      nn.Linear(hidden_dim, 1))  # 输出层，输入维度为隐藏层维度，输出维度为1

        self.q2_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),  # 全连接层，输入为状态和动作的维度，输出维度为隐藏层维度
                                      nn.Mish(),  # Mish激活函数
                                      nn.Linear(hidden_dim, hidden_dim),  # 全连接层，输入维度为隐藏层维度，输出维度为隐藏层维度
                                      nn.Mish(),  # Mish激活函数
                                      nn.Linear(hidden_dim, hidden_dim),  # 全连接层，输入维度为隐藏层维度，输出维度为隐藏层维度
                                      nn.Mish(),  # Mish激活函数
                                      nn.Linear(hidden_dim, 1))  # 输出层，输入维度为隐藏层维度，输出维度为1

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)  # 将状态和动作按照最后一个维度连接起来
        return self.q1_model(x), self.q2_model(x)  # 返回q1_model和q2_model对连接后的输入的结果

    def q1(self, state, action):
        x = torch.cat([state, action], dim=-1)  # 将状态和动作按照最后一个维度连接起来
        return self.q1_model(x)  # 返回q1_model对连接后的输入的结果

    def q_min(self, state, action):
        q1, q2 = self.forward(state, action)  # 调用forward函数得到q1_model和q2_model对连接后的输入的结果
        return torch.min(q1, q2)  # 返回q1和q2中较小的值


class Diffusion_QL(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 device,
                 discount,
                 tau,
                 max_q_backup=False,
                 eta=1.0,
                 beta_schedule='linear',
                 n_timesteps=100,
                 ema_decay=0.995,
                 step_start_ema=1000,
                 update_ema_every=5,
                 lr=3e-4,
                 lr_decay=False,
                 lr_maxt=1000,
                 grad_norm=1.0,
                 ):
        """
        初始化Diffusion_QL类

        参数：
            state_dim(int): 状态空间的维度大小
            action_dim(int): 动作空间的维度大小
            max_action(float): 动作的上限值
            device(str): 设备类型，比如'cpu'或'cuda'
            discount(float): 折扣因子，用于计算累积奖励
            tau(float): 目标网络更新参数
            max_q_backup(bool): 是否使用max Q备份目标
            eta(float): Q-learning权重
            beta_schedule(str): 探索程度的时间表
            n_timesteps(int): 探索程度的时间步数
            ema_decay(float): 指数移动平均的衰减因子
            step_start_ema(int): 开始进行指数移动平均的时间步
            update_ema_every(int): 指数移动平均更新间隔
            lr(float): 学习率
            lr_decay(bool): 是否使用学习率衰减
            lr_maxt(int): 学习率衰减的最大时间步数
            grad_norm(float): 梯度的最大范数
        """
        # 创建MLP模型（actor网络）实例
        self.model = MLP(state_dim=state_dim, action_dim=action_dim, device=device)

        # 创建Diffusion网络（actor网络）实例
        self.actor = Diffusion(state_dim=state_dim, action_dim=action_dim, model=self.model, max_action=max_action,
                               beta_schedule=beta_schedule, n_timesteps=n_timesteps,).to(device)
        # 使用Adam优化器优化actor网络参数
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.lr_decay = lr_decay
        self.grad_norm = grad_norm

        self.step = 0
        self.step_start_ema = step_start_ema
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.actor)
        self.update_ema_every = update_ema_every

        # 创建Critic网络实例
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        # 使用Adam优化器优化critic网络参数
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        if lr_decay:
            # 创建actor学习率的cosine衰减器
            self.actor_lr_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=lr_maxt, eta_min=0.)
            # 创建critic学习率的cosine衰减器
            self.critic_lr_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=lr_maxt, eta_min=0.)

        self.state_dim = state_dim # 状态空间的维度大小
        self.max_action = max_action # 动作的上限值
        self.action_dim = action_dim # 动作空间的维度大小
        self.discount = discount # 折扣因子
        self.tau = tau # 目标网络更新参数
        self.eta = eta  # q_learning权重
        self.device = device # 设备类型
        self.max_q_backup = max_q_backup # 是否使用max Q备份目标

    def step_ema(self):
        # 判断是否需要进行指数移动平均
        if self.step < self.step_start_ema:
            return
        self.ema.update_model_average(self.ema_model, self.actor)

    def train(self, replay_buffer, iterations, batch_size=100, log_writer=None):
        """
        使用训练集训练模型

        参数：
            replay_buffer(object): 经验回放缓冲区对象
            iterations(int): 训练的迭代次数
            batch_size(int): 每个迭代中的批量大小
            log_writer(object): 日志记录器对象

        返回：
            metric(dict): 记录训练过程中的指标
        """
        metric = {'bc_loss': [], 'ql_loss': [], 'actor_loss': [], 'critic_loss': []}
        for _ in range(iterations):
            # 从经验回放缓冲区中取样
            state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

            """ Q Training """
            current_q1, current_q2 = self.critic(state, action)

            if self.max_q_backup:
                next_state_rpt = torch.repeat_interleave(next_state, repeats=10, dim=0)
                next_action_rpt = self.ema_model(next_state_rpt)
                target_q1, target_q2 = self.critic_target(next_state_rpt, next_action_rpt)
                target_q1 = target_q1.view(batch_size, 10).max(dim=1, keepdim=True)[0]
                target_q2 = target_q2.view(batch_size, 10).max(dim=1, keepdim=True)[0]
                target_q = torch.min(target_q1, target_q2)
            else:
                next_action = self.ema_model(next_state)
                target_q1, target_q2 = self.critic_target(next_state, next_action)
                target_q = torch.min(target_q1, target_q2)

            target_q = (reward + not_done * self.discount * target_q).detach()

            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.grad_norm > 0:
                critic_grad_norms = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm, norm_type=2)
            self.critic_optimizer.step()

            """ Policy Training """
            bc_loss = self.actor.loss(action, state)
            new_action = self.actor(state)

            q1_new_action, q2_new_action = self.critic(state, new_action)
            if np.random.uniform() > 0.5:
                q_loss = - q1_new_action.mean() / q2_new_action.abs().mean().detach()
            else:
                q_loss = - q2_new_action.mean() / q1_new_action.abs().mean().detach()
            actor_loss = bc_loss + self.eta * q_loss

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.grad_norm > 0:
                actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm, norm_type=2)
            self.actor_optimizer.step()


            """ Step Target network """
            if self.step % self.update_ema_every == 0:
                self.step_ema()

            # 更新目标网络的参数
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            self.step += 1

            """ Log """
            if log_writer is not None:
                if self.grad_norm > 0:
                    log_writer.add_scalar('Actor Grad Norm', actor_grad_norms.max().item(), self.step)
                    log_writer.add_scalar('Critic Grad Norm', critic_grad_norms.max().item(), self.step)
                log_writer.add_scalar('BC Loss', bc_loss.item(), self.step)
                log_writer.add_scalar('QL Loss', q_loss.item(), self.step)
                log_writer.add_scalar('Critic Loss', critic_loss.item(), self.step)
                log_writer.add_scalar('Target_Q Mean', target_q.mean().item(), self.step)

            metric['actor_loss'].append(actor_loss.item())
            metric['bc_loss'].append(bc_loss.item())
            metric['ql_loss'].append(q_loss.item())
            metric['critic_loss'].append(critic_loss.item())

        if self.lr_decay:
            self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()

        return metric

    def sample_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        state_rpt = torch.repeat_interleave(state, repeats=50, dim=0)
        with torch.no_grad():
            action = self.actor.sample(state_rpt)
            q_value = self.critic_target.q_min(state_rpt, action).flatten()
            idx = torch.multinomial(F.softmax(q_value), 1)
        return action[idx].cpu().data.numpy().flatten()

    def save_model(self, dir, id=None):
        """
        保存模型

        参数：
            dir(str): 保存模型的路径
            id(int): 模型的标识符
        """
        if id is not None:
            torch.save(self.actor.state_dict(), f'{dir}/actor_{id}.pth')
            torch.save(self.critic.state_dict(), f'{dir}/critic_{id}.pth')
        else:
            torch.save(self.actor.state_dict(), f'{dir}/actor.pth')
            torch.save(self.critic.state_dict(), f'{dir}/critic.pth')

    def load_model(self, dir, id=None):
        """
        加载模型

        参数：
            dir(str): 加载模型的路径
            id(int): 模型的标识符
        """
        if id is not None:
            self.actor.load_state_dict(torch.load(f'{dir}/actor_{id}.pth'))
            self.critic.load_state_dict(torch.load(f'{dir}/critic_{id}.pth'))
        else:
            self.actor.load_state_dict(torch.load(f'{dir}/actor.pth'))
            self.critic.load_state_dict(torch.load(f'{dir}/critic.pth'))