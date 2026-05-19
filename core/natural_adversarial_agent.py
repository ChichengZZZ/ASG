import numpy
import os
import sys

import time
import torch
import numpy as np
import math
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import pandas
import Utils
from Utils import ZFilter
from Utils import to_device
from Utils import utils
from Utils import Logger, setup_logger
from Utils import Kalman1D
from Utils import save
from Utils.replay_memory_v2 import Memory

from collections import deque
from Utils.replay_memory_v3 import ReplayBuffer
from core.abstract import Agent
from core.ppo import ppo_step
from core.common import estimate_advantages
from NGSIM_env.envs import NGSIMGAILEnv #, InterActionGAILEnv
import matplotlib.pyplot as plt
# from models.mlp_discriminator import Discriminator
from models.bc_diffusion import Diffusion_BC
from models.mlp_policy import Policy
from models.mlp_critic import Value
from models.mlp_q_net import Q
# import socket
from Utils.server_utils import *
import pickle
import wandb
import pandas as pd
from multiprocessing import cpu_count
from gym import wrappers
from NGSIM_env.envs import HighwayAdEnv #, IntersectionAdEnv
torch.autograd.set_detect_anomaly(True)


def reset_last_layers(network, num_layers):
    """
    重置网络的最后 num_layers 层
    """
    layers = list(network.children())
    for layer in layers[-num_layers:]:
        if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
        elif isinstance(layer, nn.Sequential):
            reset_last_layers(layer, num_layers)
    return network


class NaturalAdversarialAgent(Agent):

    def __init__(self,
                 args,
                 state_dim: int,
                 action_dim: int,
                 max_action: list,
                 train: bool = True,
                 num_threads: int = 4,
                 scenario: str = 'highway',
                 project_dir: str = None,
                 vehicle_dir: str = "NGSIM_env/data/trajectory_set/highway",
                 save_dir: str = "examples/result/data",
                 load_model_id: int = 0,
                 max_iter_num=500,
                 batch_size=2048,
                 lr: float = 1e-3,
                 optim_epochs=10,
                 optim_batch_size=64,
                 discount: float = 0.99,
                 tau: float = 0.95,
                 l2_reg: float = 1e-3,
                 clip_epsilon: float = 0.2,
                 device=torch.device('cpu'),
                 use_running_state=True,
                 seed=0,
                 save_model_iter=1,
                 eval_iter=1,
                 supervise_model_name='Gail_1',
                 supervise_model_id='0',
                 supervise_model_state_dim=0,
                 supervise_model_action_dim=0,
                 supervise_model_T=0,
                 supervise_use_running_state=True,
                 reset_interval=1000, reset_network=3, replay_buffer_capacity=100000
                 ):

        super(NaturalAdversarialAgent, self).__init__()
        self.args = args
        self.exp_name = args.exp_name
        self.seed = seed if self.args.seed is None else self.args.seed
        # torch.set_default_dtype(torch.float32)  # 设置默认张量类型为torch.float32
        torch.set_default_dtype(self.type)  # 设置默认张量类型为torch.float32
        torch.manual_seed(self.seed)  # 设置随机种子
        np.random.seed(self.seed)
        random.seed(self.seed)

        # 初始化当前种子为初始种子
        self.current_seed = self.seed

        self.project_dir = project_dir  # 项目目录
        self.save_dir = save_dir  # 保存目录

        self.state_dim = state_dim  # 状态维度
        self.action_dim = action_dim  # 动作维度
        self.max_action = max_action  # 动作的最大值（用于动作空间的归一化）
        self.scenario = scenario  # 场景

        self.train = train  # 是否为训练模式
        self.start_epoch = 0  # 训练的起始轮次
        self.max_iter_num = max_iter_num  # 最大训练轮次数
        self.batch_size = batch_size  # 批量训练大小
        self.lr = lr  # 学习率
        self.device = device  # 设备（CPU或GPU）
        self.discount = discount  # 折扣因子
        self.tau = tau  # 软更新参数
        self.l2_reg = l2_reg  # L2正则化系数
        self.clip_epsilon = clip_epsilon  # PPO算法中的clip参数
        self.max_grad_norm = 0.5  # 梯度裁剪的最大范数

        # optimization epoch number and batch size for PPO
        self.optim_epochs = optim_epochs  # PPO算法的优化轮次
        self.optim_batch_size = optim_batch_size  # PPO算法的优化批量大小

        # 重置机制参数
        self.reset_interval = reset_interval
        self.reset_network = reset_network
        self.step_count = 0

        # 初始化Replay Buffer
        self.replay_buffer = ReplayBuffer(capacity=replay_buffer_capacity)

        self.init_network()  # 初始化神经网络

        self.use_running_state = use_running_state  # 是否使用运行状态归一化
        self.running_state = ZFilter((self.state_dim, ), clip=5.0) if use_running_state else None

        self.num_threads = min(num_threads, cpu_count())  # 可用线程数
        self.save_model_iter = save_model_iter  # 保存模型的迭代数
        self.eval_iter = eval_iter  # 评估模型的迭代数

        self.expert_traj = None
        self.vehicle_dir = vehicle_dir  # 车辆数据目录

        self.output_path = None
        self.make_dir()  # 创建保存输出的目录

        if not self.train or self.args.resume:
            self.load_model_id = load_model_id  # 要加载的模型ID
            self.load_model(load_model_id)  # 加载模型

        # 加载监督信息
        if self.args.action_type == 'continuous':
            self.supervise_model_name = supervise_model_name  # 监督模型的名称
            self.supervise_model_id = supervise_model_id  # 监督模型的ID
            self.supervise_model_state_dim = supervise_model_state_dim  # 监督模型的状态维度
            self.supervise_model_action_dim = supervise_model_action_dim  # 监督模型的动作维度
            self.supervise_model_T = supervise_model_T  # 监督模型的时间步长
            self.supervise_use_running_state = supervise_use_running_state  # 监督模型是否使用运行状态归一化
            ######再次指定一些
            self.args.supervise_model_update_mode = 'ppo'
            if self.train:
                self.load_supervise_model()  # 加载监督模型

        self.logger = None
        self.exp_log()  # 实验日志记录

    def make_dir(self):
        """创建保存输出的目录。"""
        filename = f"{self.__class__.__name__}"
        self.output_path = os.path.join(self.project_dir, self.save_dir, self.scenario, filename, self.args.model+"_"+self.args.policy_type+"_"+self.args.action_type, self.args.supervise_model,  self.exp_name)
        self.onnx_model_path = os.path.join(self.output_path, 'test.onnx') # 设置ONNX模型的路径

        print(self.onnx_model_path)

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def init_network(self):
        # 根据选择的模型类型进行网络初始化
        if self.args.model in ['ppo']:
            # 初始化策略网络
            if self.args.action_type == 'discrete':
                from models.mlp_policy_disc import DiscretePolicy
                self.policy_net = DiscretePolicy(self.state_dim, self.action_dim)
            else:
                self.policy_net = Policy(self.state_dim, self.action_dim)

            # 初始化价值网络
            self.value_net = Value(self.state_dim)

            # 将网络转移到指定设备
            self.device = torch.device("cpu")
            self.type = torch.float64
            self.policy_net.to(self.device)
            self.value_net.to(self.device)

            # 初始化优化器
            self.optimizer_policy = optim.Adam(self.policy_net.parameters(), lr=self.lr)
            self.optimizer_value = optim.Adam(self.value_net.parameters(), lr=self.lr)
        else:
            raise NotImplementedError(f"Model {self.args.model} is not implemented.")

    # def init_network(self):
    #     # 根据选择的模型类型进行网络初始化
    #     if self.args.model in ['ppo_v2']:
    #         # 如果是ppo_v2模型
    #         # 初始化策略网络
    #         if self.args.action_type == 'discrete':
    #             from models.mlp_policy_disc import DiscretePolicy
    #             self.policy_net = DiscretePolicy(self.state_dim, self.action_dim)
    #         else:
    #             from models.mlp_policy_v2 import Policy as Policy_v2
    #             self.policy_net = Policy_v2(self.state_dim, self.action_dim)
    #
    #         # 初始化价值网络
    #         self.value_net = Value(self.state_dim)
    #
    #         # 将网络转移到指定设备
    #         to_device(self.device, self.policy_net, self.value_net)
    #
    #         # 初始化策略网络和价值网络的优化器和学习率调度器
    #         self.optimizer_value = torch.optim.Adam(self.value_net.parameters(), lr=self.lr)
    #         self.value_step_lr = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_value, self.args.num_epochs, 1e-4)
    #         self.optimizer_policy = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)
    #         self.policy_step_lr = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_policy, self.args.num_epochs, 1e-4)
    #
    #     elif self.args.model in ['ppo']:
    #         # 如果是ppo模型
    #         # 初始化策略网络
    #         if self.args.action_type == 'discrete':
    #             from models.mlp_policy_disc import DiscretePolicy
    #             self.policy_net = DiscretePolicy(self.state_dim, self.action_dim)
    #         else:
    #             self.policy_net = Policy(self.state_dim, self.action_dim)
    #
    #         # 初始化价值网络
    #         self.value_net = Value(self.state_dim)
    #
    #         # 将网络转移到指定设备
    #         to_device(self.device, self.policy_net, self.value_net)
    #
    #         # 初始化策略网络和价值网络的优化器
    #         self.optimizer_policy = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)
    #         self.optimizer_value = torch.optim.Adam(self.value_net.parameters(), lr=self.lr)
    #
    #     elif self.args.model in ['sac']:
    #         # 如果是sac模型
    #         # 设置超参数
    #         self.tau = 0.005
    #         self.discount = 0.99
    #
    #         # 初始化策略网络、价值网络和Q网络
    #         from models.mlp_policy_v3 import Policy as Policy_v3
    #         self.value_net = Value(self.state_dim)
    #         self.policy_net = Policy_v3(self.state_dim, self.action_dim)
    #         self.q_net = Q(self.state_dim, self.action_dim)
    #
    #         # 创建目标价值网络
    #         self.target_value_net = Value(self.state_dim)
    #
    #         # 初始化策略网络、价值网络和Q网络的优化器和学习率调度器
    #         self.optimizer_value = torch.optim.Adam(self.value_net.parameters(), lr=self.lr)
    #         self.value_step_lr = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_value, self.args.num_epochs, 1e-4)
    #
    #         self.optimizer_q_value = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)
    #         self.q_value_step_lr = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_q_value, self.args.num_epochs, 1e-4)
    #
    #         self.optimizer_policy = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)
    #         self.policy_step_lr = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_policy, self.args.num_epochs, 1e-4)
    #
    #         # 将网络转移到指定设备
    #         for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
    #             target_param.data.copy_(param.data)
    #         to_device(self.device, self.policy_net, self.value_net, self.q_net, self.target_value_net)

    def set_env(self, save_video=False):
        # 设置环境
        if self.scenario == 'highway':
            # 如果场景是高速公路
            vehicle_dir = os.path.join(self.project_dir, self.vehicle_dir)
            vehicle_names = os.listdir(vehicle_dir)[:]
            # 获取车辆目录下的所有文件名
            select_vehicle_name = np.random.choice(vehicle_names, size=1, replace=False)
            # 随机选择一个文件名
            path = os.path.join(vehicle_dir, select_vehicle_name[0])
            # 生成文件路径
            vehicle_id = select_vehicle_name[0].split('.')[0].split('_')[-1]
            # 从文件名中提取车辆ID
            period = 0
            env = NGSIMGAILEnv(scene='us-101', path=path, period=period, vehicle_id=vehicle_id, IDM=False, gail=False,
                               action_type=self.args.action_type)
            # 创建NGSIMGAILEnv环境对象
            env.duration = 200
            # 设置环境的持续时间

            if save_video:
                env = wrappers.Monitor(env,
                                       directory="{}/videos/{}_{}/".format(self.output_path,
                                                                           select_vehicle_name[0].split('.')[0],
                                                                           utils.get_current_time()),
                                       force=True, video_callable=lambda e: True)
                # 如果保存视频，则创建wrappers.Monitor环境对象
                env.configure({"simulation_frequency": self.args.fps})
                env.set_monitor(env)

            env.reset(reset_time=0)
            state = env.adv_features()
            gail_state = env.gail_features_v2()

            return env, state, gail_state, select_vehicle_name
            # 返回环境对象、状态和GAIL状态

    def loss_multivariate_normal_WD(self, mu_1, sigmasquare_1, mu_2, sigmasquare_2):
        b, dim = mu_1.shape
        wd = 0

        for bidx in range(b):
            c_mu_1 = mu_1[bidx]
            c_Sigma_1 = torch.diag(sigmasquare_1[bidx]).sqrt()
            c_mu_2 = mu_2[bidx]
            c_Sigma_2 = torch.diag(sigmasquare_2[bidx]).sqrt()

            mean_diff = torch.norm(c_mu_1 - c_mu_2)
            sigma_diff = torch.trace(c_Sigma_1 + c_Sigma_2 - 2 * (c_Sigma_1 @ c_Sigma_2 @ c_Sigma_1).sqrt())

            wd += mean_diff ** 2 + sigma_diff

        wd = wd / b
        return wd

    def loss_multivariate_normal_kl(self, mu_1, sigmasquare_1, mu_2, sigmasquare_2):
        # 多元正态分布的KL散度损失函数
        b, dim = mu_1.shape
        kl = 0

        for bidx in range(b):
            # 遍历每个批次中的样本
            c_mu_1 = mu_1[bidx]
            c_Sigma_1 = torch.diag(sigmasquare_1[bidx])

            c_mu_2 = mu_2[bidx]
            c_Sigma_2 = torch.diag(sigmasquare_2[bidx])

            first = c_Sigma_2.det().log() - c_Sigma_1.det().log()
            # 计算第一部分
            second = -dim
            # 计算第二部分
            third = torch.matmul(c_Sigma_2.inverse(), c_Sigma_1).trace()
            # 计算第三部分
            fourth = torch.matmul(torch.matmul((c_mu_2 - c_mu_1).t(), c_Sigma_2.inverse()), c_mu_2 - c_mu_1)
            # 计算第四部分
            kl = kl + 0.5 * (first + second + third + fourth)

        kl = kl / b
        return kl
        # 返回KL散度损失值

    def expert_reward(self, state, actions, batch=None):
        # 专家奖励
        if batch is not None:
            actions = torch.from_numpy(np.stack(batch.actions)).to(self.type).squeeze().to(self.device)

        with torch.no_grad():
            if self.supervise_model_name.startswith('Gail'):
                pre_actions = self.supervise_policy.select_action(state)
                # 获取预测的动作
                rewards = []

                ############### 引入KL散度
#                for i in range(actions.shape[0]):
#                    kl_div = self.loss_multivariate_normal_kl(actions[i, 1:2], actions[i, 2:3] ** 2,
#                                                               pre_actions[1][i:i + 1], pre_actions[2][i:i + 1] ** 2)
#                    # 计算KL散度损失
#                    kl_reward = kl_div.item()
#                    print(f"kl_reward (before clip): {kl_reward}")  # 打印KL散度
#                    reward = np.clip((25 - kl_reward) / 25, 0, 1)
#                    print(f"reward (after clip): {reward}")  # 打印最终奖励
#                    # 根据KL散度损失计算奖励
#                    rewards.append(reward)

                ################## 引入 Wasserstein Distance
                for i in range(actions.shape[0]):
                    wd_div = self.loss_multivariate_normal_WD(actions[i, 1:2], actions[i, 2:3] ** 2,
                                                              pre_actions[1][i:i + 1], pre_actions[2][i:i + 1] ** 2)
                    # 计算wd散度损失
                    wd_reward = wd_div.item()
                    print(f"kl_reward (before clip): {wd_reward}")  # 打印KL散度
                    reward = np.clip((2.0 - wd_reward) / 2.0, 0, 1)
                    print(f"reward (after clip): {reward}")  # 打印最终奖励
                    # 根据wd散度损失计算奖励
                    rewards.append(reward)

                return rewards
                # 返回奖励值

            elif self.supervise_model_name.startswith('Diffusion'):
                pre_actions = self.supervise_policy.actor.sample(state)
                # 获取预测的动作
                rewards = []
                for i in range(actions.shape[0]):
                    mse = torch.nn.functional.mse_loss(actions[i, 0, :], pre_actions[i, :])
                    # 计算均方误差损失
                    reward = - mse
                    # 根据均方误差损失计算奖励
                    rewards.append(reward.item())
                return rewards
                # 返回奖励值

    def ppo_update_params_v2(self, batch, i_iter):
        state = torch.from_numpy(np.stack(batch.state)).to(self.type).to(self.device)  # 转换状态为PyTorch的张量
        next_state = torch.from_numpy(np.stack(batch.next_state)).to(self.type).to(self.device)  # 转换下一个状态为PyTorch的张量
        action = torch.from_numpy(np.stack(batch.action)).to(self.type).to(self.device)  # 转换动作为PyTorch的张量
        actions = torch.from_numpy(np.stack(batch.actions)).to(self.type).squeeze().to(self.device)  # 转换动作序列为PyTorch的张量
        rewards = torch.from_numpy(np.stack(batch.reward)).to(self.type).to(self.device)  # 转换奖励为PyTorch的张量
        masks = torch.from_numpy(np.stack(batch.mask)).to(self.type).to(self.device)  # 转换掩码为PyTorch的张量

        if self.args.action_type == 'continuous':
            supervise_states = torch.from_numpy(np.stack(batch.supervise_state)).to(self.type).to(
                self.device)  # 转换监督状态为PyTorch的张量
            new_expert_rewards = self.expert_reward(supervise_states, action, batch)  # 计算新的专家奖励
            new_expert_rewards = torch.from_numpy(np.array(new_expert_rewards)).to(self.type).to(
                self.device)  # 转换新的专家奖励为PyTorch的张量
        else:
            new_expert_rewards = torch.from_numpy(np.zeros((10,))).to(self.type).to(
                self.device)  # 如果动作类型不是连续型，则将新的专家奖励初始化为全零

        if self.args.policy_type == 'NAM':  # 如果策略类型是NAM
            rewards = rewards + new_expert_rewards  # 将新的专家奖励添加到原始奖励上
        elif self.args.policy_type == 'AM':  # 如果策略类型是AM
            pass  # 什么都不做
        else:
            raise TypeError(f"can't support policy type of {self.args.policy_type}")  # 如果策略类型是其他未知类型，则抛出异常

        old_action_log_prob = actions[:, -1, :].clone().to(self.device)  # 记录旧的动作的对数概率

        with torch.no_grad():
            values = self.value_net(next_state)  # 计算下一个状态的值函数估计
            target_v = rewards + torch.squeeze(self.discount * values)  # 计算目标值函数

        advantage = (target_v - torch.squeeze(self.value_net(state))).detach()  # 计算优势估计

        # PPO核心部分的迭代优化过程
        for _ in range(self.optim_epochs):  # 迭代ppo_epoch次
            for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.optim_batch_size,
                                      True):  # 使用批次采样器获取索引
                mu, sigma = self.policy_net(state[index])  # 计算策略网络给定状态下的动作均值和标准差
                n = Normal(mu, sigma)  # 根据均值和标准差创建正态分布
                action_log_prob = n.log_prob(action[index])  # 计算动作的对数概率
                action_log_prob = torch.sum(action_log_prob, dim=1)  # 对动作的对数概率进行求和
                ratio = torch.exp(action_log_prob - torch.sum(old_action_log_prob[index], dim=1))  # 计算比值

                L1 = ratio * advantage[index]  # L1代价函数
                L2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantage[index]  # L2代价函数
                action_loss = -torch.min(L1, L2).mean()  # 计算动作损失函数，通过最小化来最大化
                self.optimizer_policy.zero_grad()  # 清空策略网络的梯度
                action_loss.backward()  # 反向传播计算梯度
                nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)  # 对策略网络的梯度进行裁剪
                self.optimizer_policy.step()  # 执行策略网络的优化步骤

                value_loss = F.smooth_l1_loss(torch.squeeze(self.value_net(state[index])),
                                              target_v[index])  # 计算值函数的损失函数
                self.optimizer_value.zero_grad()  # 清空值函数网络的梯度
                value_loss.backward()  # 反向传播计算梯度
                nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)  # 对值函数网络的梯度进行裁剪
                self.optimizer_value.step()  # 执行值函数网络的优化步骤

        self.policy_step_lr.step(i_iter)  # 对策略网络的学习率进行调整
        self.value_step_lr.step(i_iter)  # 对值函数网络的学习率进行调整

        rewards_numpy = rewards.detach().numpy()  # 将奖励转换为NumPy数组
        mean_ = float(np.mean(rewards_numpy))  # 计算奖励的平均值
        std = float(np.std(rewards_numpy, ddof=0))  # 计算奖励的标准差
        std_error = float(std / np.sqrt(rewards_numpy.shape[0]))  # 计算奖励的标准误差

        natural_reward = new_expert_rewards.detach().numpy()  # 将新的专家奖励转换为NumPy数组
        natural_reward_mean = float(np.mean(natural_reward))  # 计算新的专家奖励的平均值
        natural_reward_std = float(np.std(natural_reward, ddof=0))  # 计算新的专家奖励的标准差
        natural_reward_std_error = float(natural_reward_std / np.sqrt(natural_reward.shape[0]))  # 计算新的专家奖励的标准误差

        log = {
            'reward_mean': mean_,
            'reward_std': std,
            'reward_std_error': std_error,
            'natural_reward': natural_reward_mean,
            'natural_reward_std': natural_reward_std,
            'natural_reward_std_error': natural_reward_std_error
        }
        return log


    def update_params(self, batch, i_iter):
        if i_iter > 100:
            for param_group in self.optimizer_policy.param_groups:
                param_group['lr'] = 1.0e-4
            for param_group in self.optimizer_value.param_groups:
                param_group['lr'] = 1.0e-4
        # 将数据转换为Tensor，并放到设备上
        states = torch.from_numpy(np.stack(batch.state)).to(self.type).to(self.device)
        actions = torch.from_numpy(np.stack(batch.action)).to(self.type).to(self.device)
        masks = torch.from_numpy(np.stack(batch.mask)).to(self.type).to(self.device)
        #next_states = torch.from_numpy(np.stack(batch.next_state)).to(self.type).to(self.device)
        rewards = torch.from_numpy(np.stack(batch.reward)).to(self.type).to(self.device)
        supervise_states = torch.from_numpy(np.stack(batch.supervise_state)).to(self.type).to(self.device)

        # 处理专家奖励（根据你的需求调整）
        if self.args.action_type == 'continuous':
            new_expert_rewards = self.expert_reward(supervise_states, actions, batch)
            new_expert_rewards = torch.from_numpy(np.array(new_expert_rewards)).to(self.type).to(self.device)
        else:
            new_expert_rewards = torch.from_numpy(np.zeros((len(batch),))).to(self.type).to(self.device)

        # 根据策略类型进行奖励修正
        if self.args.policy_type == 'NAM':
            rewards = rewards + new_expert_rewards
        elif self.args.policy_type == 'AM':
            pass
        else:
            raise TypeError(f"不支持的策略类型：{self.args.policy_type}")

        # 使用无梯度计算获取值函数和固定的对数概率
        with torch.no_grad():
            values = self.value_net(states)
            fixed_log_probs = self.policy_net.get_log_prob(states, actions)

        # 从轨迹中估计优势和回报
        advantages, returns = estimate_advantages(rewards, masks, values, self.discount, self.tau, self.device)

        # 执行小批量PPO更新
        optim_iter_num = int(math.ceil(states.shape[0] / self.optim_batch_size))
        for _ in range(self.optim_epochs):
            perm = np.arange(states.shape[0])
            np.random.shuffle(perm)
            perm = torch.LongTensor(perm).to(self.device)
            states_shuffled, actions_shuffled, returns_shuffled, advantages_shuffled, fixed_log_probs_shuffled = \
                states[perm].clone(), actions[perm].clone(), returns[perm].clone(), advantages[perm].clone(), \
                fixed_log_probs[perm].clone()
            for i in range(optim_iter_num):
                ind = slice(i * self.optim_batch_size, min((i + 1) * self.optim_batch_size, states.shape[0]))
                states_b, actions_b, advantages_b, returns_b, fixed_log_probs_b = \
                    states_shuffled[ind], actions_shuffled[ind], advantages_shuffled[ind], returns_shuffled[ind], \
                    fixed_log_probs_shuffled[ind]
                ppo_step(
                    self.policy_net,
                    self.value_net,
                    self.optimizer_policy,
                    self.optimizer_value,
                    1,
                    states_b,
                    actions_b,
                    returns_b,
                    advantages_b,
                    fixed_log_probs_b,
                    self.clip_epsilon,
                    self.l2_reg
                )

        # 记录日志
        rewards_numpy = rewards.detach().cpu().numpy()
        mean_ = float(np.mean(rewards_numpy))
        std = float(np.std(rewards_numpy, ddof=0))
        std_error = float(std / np.sqrt(rewards_numpy.shape[0]))
        natural_reward = new_expert_rewards.detach().cpu().numpy()
        natural_reward_mean = float(np.mean(natural_reward))
        natural_reward_std = float(np.std(natural_reward, ddof=0))
        natural_reward_std_error = float(natural_reward_std / np.sqrt(natural_reward.shape[0]))
        log = {
            'replay_reward_mean': mean_,
            'replay_reward_std': std,
            'replay_reward_std_error': std_error,
            'replay_natural_reward_mean': natural_reward_mean,
            'replay_natural_reward_std': natural_reward_std,
            'replay_natural_reward_std_error': natural_reward_std_error
        }
        return log

    def collect_samples(self, pid, env, policy, running_state, min_batch_size, device):
        torch.manual_seed(self.seed)  # 设置随机种子
        memory = Memory()  # 创建存储记忆的对象
        num_steps = 0  # 记录已经走过了多少步
        total_reward = 0  # 记录总的奖励
        min_reward = 1e6  # 记录最小奖励，初始化值为无穷大
        max_reward = -1e6  # 记录最大奖励，初始化值为负无穷大
        num_episodes = 0  # 记录总共进行了多少个episode

        while num_steps < min_batch_size:  # 当已经走过的步数小于最小批次大小时
            try:
                env, adv_state, gail_state, _ = self.set_env()  # 设置环境、adversarial state和gail state
            except:
                continue

            # 针对不同场景的过滤条件
            if self.scenario == 'highway' and env.ego is None:  # 如果是highway的场景且ego车不存在
                continue
            elif self.scenario == 'intersection' and (adv_state is None or not env.open_adversarial):  # 如果是intersection的场景且adversarial state不存在或者不是开启了adversarial training
                continue

            if hasattr(env, 'np_random'):
                env.np_random.seed(self.seed)
            if hasattr(env, 'env') and hasattr(env.env, 'np_random'):
                env.env.np_random.seed(self.seed)

            # 处理状态
            if running_state is not None:
                state = running_state(adv_state)  # 对adv_state进行状态处理
            else:
                state = adv_state

            reward_episode = 0  # 记录当前episode的奖励
            for t in range(10000):  # 在每个episode中进行循环，最多循环10000次
                if self.args.action_type == "continuous":
                    if self.supervise_use_running_state:
                        supervise_state = self.supervise_running_state(gail_state)  # 对gail state进行状态处理
                    else:
                        supervise_state = gail_state
                else:
                    supervise_state = None

                state_var = torch.tensor(state).unsqueeze(0)  # 将状态转化为张量形式
                actions = None
                with torch.no_grad():
                    if self.args.action_type == 'continuous':
                        actions = policy.select_action(state_var.to(device))  # 选择动作
                        actions = to_device(torch.device('cpu'), *actions)
                        action = actions[0][0].numpy()
                    else:
                        action, actions = policy.select_action(state_var.to(device))
                        action = action[0][0].numpy()
                action = action.astype(np.float64) if self.args.action_type == 'continuous' else int(action)  # 将动作转换为浮点数或整数类型
                (next_gail_state, next_adv_state), reward, done, _ = env.step(action)  # 执行动作
                reward_episode += reward  # 累计奖励

                # 处理下一个状态
                if running_state is not None:
                    next_state = running_state(next_adv_state)  # 对下一个state进行状态处理
                else:
                    next_state = next_adv_state

                # 转换动作为numpy数组
                actions = np.array([item.cpu().detach().numpy() for item in actions])  # 将动作转化为numpy数组形式

                gail_state = next_gail_state

                mask = 0 if done else 1  # 设置mask，如果episode结束则为0，否则为1

                # 推送经验到memory，初始化 supervise_reward 为 0
                memory.push(state, action, mask, next_state, reward, 0, supervise_state, actions)

                if done:  # 如果episode结束
                    break

                state = next_state

            # 记录日志
            num_steps += (t + 1)  # 更新已经走过的步数
            num_episodes += 1  # 更新总的episode数
            total_reward += reward_episode  # 更新总的奖励
            min_reward = min(min_reward, reward_episode)  # 更新最小奖励
            max_reward = max(max_reward, reward_episode)  # 更新最大奖励

        log = {
            'num_steps': num_steps,
            'num_episodes': num_episodes,
            'adversarial_reward': total_reward,
            'avg_adversarial_reward': total_reward / num_episodes,
            'avg_step_adversarial_reward': total_reward / num_steps,
            'max_reward': max_reward,
            'min_reward': min_reward,
        }

        return pid, memory, log  # 返回pid, memory和log

    #
    # def collect_samples(self, pid, env, policy, running_state, min_batch_size, device):
    #     torch.manual_seed(self.seed)  # 设置随机种子
    #     memory = Memory()  # 创建存储记忆的对象
    #     num_steps = 0  # 记录已经走过了多少步
    #     total_reward = 0  # 记录总的奖励
    #     min_reward = 1e6  # 记录最小奖励，初始化值为无穷大
    #     max_reward = -1e6  # 记录最大奖励，初始化值为负无穷大
    #     num_episodes = 0  # 记录总共进行了多少个episode
    #
    #     while num_steps < min_batch_size:  # 当已经走过的步数小于最小批次大小时
    #
    #         try:
    #             env, adv_state, gail_state, _ = self.set_env()  # 设置环境、adversarial state和gail state
    #         except:
    #             continue
    #
    #         # if self.scenario == 'highway' and (env.v_sum > 40 or env.spd_mean < 5):
    #         if self.scenario == 'highway' and env.ego is None:  # 如果是highway的场景且ego车不存在
    #             continue
    #
    #         elif self.scenario == 'intersection' and (
    #                 adv_state is None or not env.open_adversarial):  # 如果是intersection的场景且adversarial state不存在或者不是开启了adversarial training
    #             continue
    #
    #         if hasattr(env, 'np_random'):
    #             env.np_random.seed(self.seed)
    #         if hasattr(env, 'env') and hasattr(env.env, 'np_random'):
    #             env.env.np_random.seed(self.seed)
    #
    #         # print(len(state))
    #         if running_state is not None:
    #             state = running_state(adv_state)  # 对adv_state进行状态处理
    #         else:
    #             state = adv_state
    #
    #         reward_episode = 0  # 记录当前episode的奖励
    #         for t in range(10000):  # 在每个episode中进行循环，最多循环10000次
    #
    #             if self.args.action_type == "continuous":
    #                 if self.supervise_use_running_state:
    #                     supervise_state = self.supervise_running_state(gail_state)  # 对gail state进行状态处理
    #                 else:
    #                     supervise_state = gail_state
    #             else:
    #                 supervise_state = None
    #
    #             state_var = torch.tensor(state).unsqueeze(0)  # 将状态转化为张量形式
    #             actions = None
    #             with torch.no_grad():
    #                 if self.args.action_type == 'continuous':
    #                     actions = policy.select_action(state_var.to(device))  # 选择动作
    #                     actions = to_device(torch.device('cpu'), *actions)
    #                     action = actions[0][0].numpy()
    #                 else:
    #                     action, actions = policy.select_action(state_var.to(device))
    #                     action = action[0][0].numpy()
    #             action = action.astype(np.float64) if self.args.action_type == 'continuous' else int(
    #                 action)  # 将动作转换为浮点数或整数类型
    #             (next_gail_state, next_adv_state), reward, done, _ = env.step(action)  # 执行动作
    #             reward_episode += reward  # 累计奖励
    #             if running_state is not None:
    #                 next_state = running_state(next_adv_state)  # 对下一个state进行状态处理
    #             else:
    #                 next_state = next_adv_state
    #             actions = np.array([item.cpu().detach().numpy() for item in actions])  # 将动作转化为numpy数组形式
    #             # actions_device = torch.tensor(actions).permute(1, 0, 2).to(self.device)
    #             # expert_reward = self.expert_reward(torch.unsqueeze(torch.tensor(supervise_state), dim=0), actions_device)
    #             # supervise_reward += expert_reward[0]
    #             gail_state = next_gail_state
    #
    #             mask = 0 if done else 1  # 设置mask，如果episode结束则为0，否则为1
    #
    #             memory.push(state, action, mask, next_state, reward, 0, supervise_state,
    #                         actions)  # 将当前的状态、动作等信息存储到memory中
    #
    #             if done:  # 如果episode结束
    #                 break
    #
    #             state = next_state
    #
    #         # log stats
    #         num_steps += (t + 1)  # 更新已经走过的步数
    #         num_episodes += 1  # 更新总的episode数
    #         total_reward += reward_episode  # 更新总的奖励
    #         min_reward = min(min_reward, reward_episode)  # 更新最小奖励
    #         max_reward = max(max_reward, reward_episode)  # 更新最大奖励
    #
    #     log = {'num_steps': num_steps, 'num_episodes': num_episodes, 'adversarial_reward': total_reward,
    #            'avg_adversarial_reward': total_reward / num_episodes,
    #            'avg_step_adversarial_reward': total_reward / num_steps, 'max_reward': max_reward,
    #            'min_reward': min_reward,
    #            # 'supervise_reward': supervise_rewards, 'avg_supervise_reward': supervise_rewards / num_episodes, 'avg_step_supervise_reward': supervise_rewards / num_steps
    #            }
    #
    #     return pid, memory, log  # 返回pid, memory和log


    def save_log(self, log: dict, iter: int, filename: str):
        # 如果iter为0，表示开始保存日志文件，创建一个新文件
        if iter == 0:
            with open(os.path.join(self.output_path, f'{filename}.csv'), 'w+') as f:
                head = []
                # 遍历字典的键值对，获取所有键（key）
                for k, v in log.items():
                    # 忽略特定的键
                    if k in ['sample_time', 'action_mean', 'action_min', 'action_max']:
                        continue
                    # 将键添加到列表中
                    head.append(k)
                # 将列表中的键转换为逗号分隔的字符串
                head_ = ','.join(head)
                # 将字符串写入文件
                f.write(head_)
                f.write('\n')

        # 打开现有文件或新建一个文件来追加数据
        with open(os.path.join(self.output_path, f'{filename}.csv'), 'a+') as f:
            data = []
            # 遍历字典的键值对，获取所有值（value）
            for k, v in log.items():
                # 忽略特定的键
                if k in ['sample_time', 'action_mean', 'action_min', 'action_max']:
                    continue
                # 将值转换为字符串，并添加到列表中
                data.append(str(v))
            # 将列表中的值转换为逗号分隔的字符串
            data_ = ','.join(data)
            # 将字符串写入文件
            f.write(data_)
            f.write('\n')

    def update_params_from_replay_buffer(self, i_iter):
        if len(self.replay_buffer) < self.batch_size:
            return {}

        batch = self.replay_buffer.sample(self.batch_size)

        # 将数据转换为Tensor，并放到设备上
        states = torch.from_numpy(np.stack(batch.state)).to(self.type).to(self.device)
        actions = torch.from_numpy(np.stack(batch.action)).to(self.type).to(self.device)
        masks = torch.from_numpy(np.stack(batch.mask)).to(self.type).to(self.device)
        rewards = torch.from_numpy(np.stack(batch.reward)).to(self.type).to(self.device)
        supervise_states = torch.from_numpy(np.stack(batch.supervise_state)).to(self.type).to(self.device)

        # 处理专家奖励（根据你的需求调整）
        if self.args.action_type == 'continuous':
            new_expert_rewards = self.expert_reward(supervise_states, actions, batch)
            new_expert_rewards = torch.from_numpy(np.array(new_expert_rewards)).to(self.type).to(self.device)
        else:
            new_expert_rewards = torch.from_numpy(np.zeros((len(batch),))).to(self.type).to(self.device)

        # 根据策略类型进行奖励修正
        if self.args.policy_type == 'NAM':
            rewards = rewards + new_expert_rewards
        elif self.args.policy_type == 'AM':
            pass
        else:
            raise TypeError(f"不支持的策略类型：{self.args.policy_type}")

        # 使用无梯度计算获取值函数和固定的对数概率
        with torch.no_grad():
            values = self.value_net(states)
            fixed_log_probs = self.policy_net.get_log_prob(states, actions)

        # 从轨迹中估计优势和回报
        advantages, returns = estimate_advantages(rewards, masks, values, self.discount, self.tau, self.device)

        # 执行小批量PPO更新
        optim_iter_num = int(math.ceil(states.shape[0] / self.optim_batch_size))
        for _ in range(self.optim_epochs):
            perm = np.arange(states.shape[0])
            np.random.shuffle(perm)
            perm = torch.LongTensor(perm).to(self.device)
            states_shuffled, actions_shuffled, returns_shuffled, advantages_shuffled, fixed_log_probs_shuffled = \
                states[perm].clone(), actions[perm].clone(), returns[perm].clone(), advantages[perm].clone(), \
                fixed_log_probs[perm].clone()
            for i in range(optim_iter_num):
                ind = slice(i * self.optim_batch_size, min((i + 1) * self.optim_batch_size, states.shape[0]))
                states_b, actions_b, advantages_b, returns_b, fixed_log_probs_b = \
                    states_shuffled[ind], actions_shuffled[ind], advantages_shuffled[ind], returns_shuffled[ind], \
                    fixed_log_probs_shuffled[ind]
                ppo_step(
                    self.policy_net,
                    self.value_net,
                    self.optimizer_policy,
                    self.optimizer_value,
                    1,
                    states_b,
                    actions_b,
                    returns_b,
                    advantages_b,
                    fixed_log_probs_b,
                    self.clip_epsilon,
                    self.l2_reg
                )

        # 记录日志
        rewards_numpy = rewards.detach().cpu().numpy()
        mean_ = float(np.mean(rewards_numpy))
        std = float(np.std(rewards_numpy, ddof=0))
        std_error = float(std / np.sqrt(rewards_numpy.shape[0]))
        natural_reward = new_expert_rewards.detach().cpu().numpy()
        natural_reward_mean = float(np.mean(natural_reward))
        natural_reward_std = float(np.std(natural_reward, ddof=0))
        natural_reward_std_error = float(natural_reward_std / np.sqrt(natural_reward.shape[0]))
        log = {
            'replay_reward_mean': mean_,
            'replay_reward_std': std,
            'replay_reward_std_error': std_error,
            'replay_natural_reward_mean': natural_reward_mean,
            'replay_natural_reward_std': natural_reward_std,
            'replay_natural_reward_std_error': natural_reward_std_error
        }
        return log

    def learn(self, debug=False):
        i_iter = self.start_epoch
        update_function = {
            'ppo': self.update_params,
            'ppo_v2': self.ppo_update_params_v2,
        }
        # 如果需要上传数据，则初始化wandb
        if self.args.upload:
            self.init_wandb('V2.0-算法训练验证-对抗模型算法', "Loss曲线变化图", self.job_type, upload=self.args.upload)

        # 初始化 Replay Buffer
        self.replay_buffer = ReplayBuffer(capacity=100000)

        # 循环直到达到最大迭代次数
        while i_iter < self.max_iter_num:
            for j in range(2):
                # 根据j的值判断是进行训练还是评估，如果是偶数则进行训练，否则进行评估
                self.train = True if j % 2 == 0 else False
                if not self.train:
                    continue
                t_start = time.time()
                # 计算每个线程的批次大小
                thread_batch_size = int(math.floor(self.batch_size / self.num_threads))
                # 创建进程池
                ctx = torch.multiprocessing.get_context("spawn")
                p = ctx.Pool(self.num_threads)
                results = []
                if debug:
                    env, _, _, _ = self.set_env()
                    self.collect_samples(0, env, self.policy_net, self.running_state, 256, self.device)
                # 开始多线程采样数据
                for worker_id in range(self.num_threads):
                    env, _, _, _ = self.set_env()
                    worker_args = (worker_id, env, self.policy_net, self.running_state, thread_batch_size, self.device)
                    results.append(p.apply_async(self.collect_samples, worker_args))
                p.close()
                p.join()
                # 创建一个Memory对象用于存储样本数据
                memory = Memory()
                # 存储每个线程的日志和样本数据
                worker_logs = [None] * self.num_threads
                worker_memories = [None] * self.num_threads
                for r in results:
                    try:
                        pid, worker_memory, worker_log = r.get()
                        worker_memories[pid] = worker_memory
                        worker_logs[pid] = worker_log
                    except Exception as e:
                        print(f"Error in worker {r}: {e}")
                        continue
                for worker_memory in worker_memories:
                    if worker_memory is not None:
                        memory.append(worker_memory)
                # 从memory中采样一个batch的样本数据
                batch = memory.sample()
                log_list = worker_logs
                log = self.merge_log(log_list)
                if self.train:
                    # 更新步数
                    self.step_count += self.batch_size
                    # 检查是否需要重置网络
                    if i_iter != 0 and i_iter % self.reset_interval == 0:
                        print(f"Iteration {i_iter}: 达到重置间隔，开始重置网络。")
                        # 重置网络的最后几层
                        self.policy_net = reset_last_layers(self.policy_net, self.reset_network)
                        self.value_net = reset_last_layers(self.value_net, self.reset_network)
                        self.optimizer_policy = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)
                        self.optimizer_value = torch.optim.Adam(self.value_net.parameters(), lr=self.lr)
                        print(f"重置了网络的最后 {self.reset_network} 层")
                        # 更新随机种子为当前训练轮次
                        self.current_seed = i_iter
                        self.seed = self.current_seed  # 同步更新 self.seed
                        print(f"设置随机种子为当前训练轮次: {self.current_seed}")
                        # 使用 Replay Buffer 的经验进行训练
                        replay_log = self.update_params_from_replay_buffer(i_iter)
                        log.update(replay_log)
                        print(f"使用 Replay Buffer 的经验进行训练，迭代次数: {i_iter}")
                    else:
                        # 执行模型更新
                        train_log = update_function[self.args.model](batch, i_iter)
                        log.update(train_log)
                        # 将经验存储到 Replay Buffer
                        for experience in zip(batch.state, batch.action, batch.mask, batch.next_state, batch.reward,
                                              batch.expert_reward, batch.supervise_state, batch.actions):
                            self.replay_buffer.push(*experience)

                    # 记录日志
                    self.logger.record_tabular('Train Epochs', i_iter)
                    log['sample_time'] = time.time() - t_start
                    self.save_log(log, i_iter, 'train')

                    # 如果需要上传数据，则将训练日志数据上传到wandb服务器
                    if self.args.upload:
                        log_ = {k: [v] for k, v in log.items()}
                        df = pandas.DataFrame.from_dict(log_)
                        self.upload_data_to_wandb_server('Result', df)
                else:
                    # 评估部分
                    self.logger.record_tabular('Eval Epochs', i_iter)
                    log['sample_time'] = time.time() - t_start
                    self.save_log(log, i_iter, 'eval')
                # 记录动作统计
                log['action_mean'] = np.mean(np.vstack(batch.action), axis=0)
                log['action_min'] = np.min(np.vstack(batch.action), axis=0)
                log['action_max'] = np.max(np.vstack(batch.action), axis=0)
                # 记录所有日志
                for k, v in log.items():
                    self.logger.record_tabular(k, v)
                self.logger.dump_tabular()
                # 将网络转移到设备
                to_device(self.device, self.policy_net)
                to_device(self.device, self.value_net)
                # 保存模型
                if (i_iter + 1) % self.save_model_iter == 0 and self.train:
                    self.save_model(i_iter)
            i_iter += 1
            torch.cuda.empty_cache()

        if self.args.upload:
            del self.wandb_run


    def merge_log(self, log_list):
        log = dict()
        log['adversarial_reward'] = float(sum([x['adversarial_reward'] for x in log_list]))
        log['num_episodes'] = float(sum([x['num_episodes'] for x in log_list]))
        log['num_steps'] = float(sum([x['num_steps'] for x in log_list]))
        log['avg_adversarial_reward'] = float(log['adversarial_reward'] / log['num_episodes'])
        log['avg_step_adversarial_reward'] = float(log['adversarial_reward'] / log['num_steps'])
        log['max_reward'] = float(max([x['max_reward'] for x in log_list]))
        log['min_reward'] = float(min([x['min_reward'] for x in log_list]))

        # log['supervise_reward'] = float(min([x['supervise_reward'] for x in log_list]))
        # log['avg_supervise_reward'] = float(log['supervise_reward'] / log['num_episodes'])
        # log['avg_step_supervise_reward'] = float(log['supervise_reward'] / log['num_steps'])

        return log

    def sample_action(self, state):
        with torch.no_grad():
            # 使用torch.no_grad()上下文管理器，表示在该上下文中不进行梯度计算
            actions = self.policy_net(state.to(self.device))
            # 将状态(state)传入策略网络(policy_net)得到预测的动作(actions)
            return actions

    def save_model(self, epoch):
        utils.print_banner(f"Save weights to {self.output_path}")
        # 创建一个字典`ckpts_state`用于保存模型和优化器的状态
        ckpts_state = {
            'start_epoch': epoch,
            'running_state': self.running_state,
            'policy_net': self.policy_net.state_dict(),
            'value_net': self.value_net.state_dict(),
            'optimizer_policy': self.optimizer_policy.state_dict(),
            'optimizer_value': self.optimizer_value.state_dict(),
        }

        if self.args.model in ['ppo_v2']:
            # 如果模型类型是'ppo_v2'
            ckpts_state.update(
                {
                    'value_step_lr': self.value_step_lr.state_dict(),
                    'policy_step_lr': self.policy_step_lr.state_dict(),
                }
            )

        # 使用utils.save_checkpoints()函数将ckpts_state字典保存到output_path指定的路径
        utils.save_checkpoints(ckpts_state, self.output_path, model_name=f'{epoch}_{self.args.supervise_model}_model')

    def load_model(self, epoch):
        if self.train: # 如果是训练状态
            model_path = os.path.join(self.output_path, f'latest_{self.args.supervise_model}_model_ckpt.pth')
        else: # 如果是测试状态
            model_path = os.path.join(self.output_path, f'{epoch}_{self.args.supervise_model}_model_ckpt.pth')
        # 断言模型文件是否存在，如果不存在，输出模型文件路径。
        assert os.path.exists(model_path), utils.print_banner(model_path)
        utils.print_banner(f'Load model from {model_path}.')
        # 使用torch.load函数加载模型的参数和优化器的状态字典。
        ckpt = torch.load(model_path, map_location=self.device)

        self.policy_net.load_state_dict(ckpt['policy_net'])
        self.value_net.load_state_dict(ckpt['value_net'])
        self.optimizer_policy.load_state_dict(ckpt['optimizer_policy'])
        self.optimizer_value.load_state_dict(ckpt['optimizer_value'])
        # 根据模型类型的不同，加载不同的网络权重和优化器参数。
        if self.args.model in ['ppo_v2']:
            try:
                self.value_step_lr.load_state_dict(ckpt['value_step_lr'])
                self.policy_step_lr.load_state_dict(ckpt['policy_step_lr'])
            except:
                pass
        if self.train:

            self.start_epoch = ckpt['start_epoch'] + 1

    def load_supervise_model(self):
        """导入监督模型"""

        if self.supervise_model_name.startswith('Gail'):
            model_path = os.path.join(self.project_dir, self.save_dir, self.scenario, self.supervise_model_name.format(self.args.supervise_model_update_mode))
            self.supervise_policy = Policy(self.supervise_model_state_dim, self.supervise_model_action_dim)
            model_path = os.path.join(model_path, f'{self.supervise_model_id}_{self.args.supervise_model_update_mode}_model_ckpt.pth')
            utils.print_banner(f'Load Supervised model from {model_path}.')
            ckpt = torch.load(model_path, map_location=self.device)
            self.supervise_policy.load_state_dict(ckpt['policy_net'])
            if self.supervise_use_running_state:
                self.supervise_running_state = ckpt['running_state']

        elif self.supervise_model_name.startswith('Diffusion'):
            model_path = os.path.join(self.project_dir, self.save_dir, self.scenario, self.supervise_model_name)
            self.supervise_policy = Diffusion_BC(state_dim=self.supervise_model_state_dim, action_dim=self.supervise_model_action_dim,
                                                 max_action=self.max_action, device=self.device,discount=self.discount, tau=self.tau, n_timesteps=self.supervise_model_T)
            model_path = os.path.join(model_path, f'{self.supervise_model_id}_diffusion_model_ckpt.pth')
            utils.print_banner(f'Load Supervised model from {model_path}.')
            ckpt = torch.load(model_path, map_location=self.device)
            try:
                self.supervise_policy.actor.load_state_dict(ckpt['actor'])
            except:
                self.supervise_policy.actor.load_state_dict(ckpt)

            if self.supervise_use_running_state:
                self.supervise_running_state = ckpt['running_state']

    def exp_log(self):
        if self.logger is None:
            self.logger = Logger()

        variant = {
            'Exp Name': "Natural Adversarial Model",
            'Scenario': self.scenario,
            'Train Epochs': self.max_iter_num,
            'Num Threads': self.num_threads,
            'Save Model Iter': self.save_model_iter,
            'Eval Iter': self.eval_iter,
            'State Dim': self.state_dim,
            'Action Dim': self.action_dim,
            'Max Action': self.max_action,
            'Batch Size': self.batch_size,
            'Learning Rate': self.lr,
            'Discount': self.discount,
            'Vehicle Dir': self.vehicle_dir,
            'Output Dir': self.output_path,
        }

        for k, v in variant.items():
            self.logger.record_tabular(k, v)

        self.logger.dump_tabular()

        # setup_logger(os.path.basename(self.output_path), variant=variant, log_dir=self.output_path)
        if self.scenario == "highway":
            self.job_type = "高速公路场景"
        else:
            self.job_type = " 城区十字路口场景"
        if self.args.upload:
            param_dict = {"state_dim": self.state_dim, "action_dim": self.action_dim, "max_action": self.max_action,
                          "batch_size": self.batch_size, "learning_rate": self.lr, 'discount': self.discount, 'tau': self.tau}
            df = pd.DataFrame.from_dict(param_dict)
            self.init_wandb('V2.0-算法训练验证-对抗模型算法', "算法超参数表", self.job_type, upload=self.args.upload)
            self.wandb_run.get_run().log({"Result/Algorithm Hyperparameter Table": wandb.Table(dataframe=df)})
            del self.wandb_run

    # 定义一个函数exp，用于模拟智能体与环境的交互并收集相关数据
    def exp(self, save_video=False, show=False, epochs=1000):

        # 初始化计数变量
        av_crashed = 0  # AV发生碰撞的次数
        bv_crashed = 0  # 其他车辆发生碰撞的次数
        agent_lane_c = 0  # 智能体改变车道的次数
        av_lane_c = 0  # AV改变车道的次数
        skip = 0  # 跳过的次数
        offroad = 0  # 跑出路外的次数

        actions = []  # 记录每个步骤的动作
        TTC_THW = []  # 记录每个步骤的TTC和THW（时间至碰撞和跟驰时间）
        agent_distance = 0.0  # 智能体的行驶距离（公里）
        av_distance = 0.0  # AV的行驶距离（公里）

        # 循环执行训练的迭代次数
        for epoch in range(epochs):
            # 打印进度条
            utils.print_banner('progress:{} %'.format((epoch + 1) / epochs * 100))

            # 设置环境并获取初始状态和GAIL状态
            env, state, gail_state, select_vehicle_name = self.set_env(save_video)

            # 如果是高速公路场景且没有ego车辆，则跳过该次迭代
            if self.scenario == 'highway' and env.ego is None:
                continue

            # 如果是交叉路口场景且无法开启对抗训练，则跳过该次迭代
            if self.scenario == 'intersection' and (state is None or not env.open_adversarial):
                continue

            # 打印场景名称
            utils.print_banner(f"scenario name {select_vehicle_name[0].split('.')[0]}")

            # 如果使用Running State，则对状态进行处理
            if self.use_running_state:
                state = self.running_state(state)

            # 循环执行每一步
            for i in range(200):
                # 获取当前状态
                state_var = torch.tensor(state).unsqueeze(0)

                # 根据动作类型选择采样动作
                if self.args.action_type == 'continuous':
                    # 对连续型动作的转向和加速度进行滤波
                    action = self.sample_action(state_var)[0][0].numpy()
                    steering = Kalman1D(np.array(action.reshape(1, -1))[:, 0], damping=0.1, tc=0.4).squeeze(-1)[-1]
                    acceleration = Kalman1D(np.array(action.reshape(1, -1))[:, 1], damping=0.2, tc=0.5).squeeze(-1)[-1]
                    action = [steering, acceleration]
                else:
                    # 对离散型动作选择最大概率的动作
                    action = self.sample_action(state_var)[0].numpy()
                    action = np.argmax(action)

                # 在show为True的情况下渲染环境
                if show:
                    env.render()

                # 保存当前车道编号
                pre = env.vehicle.lane_index[2]
                av_pre = env.ego.lane_index[2]

                try:
                    # 执行动作并获取下一个状态、奖励等信息
                    (gail_features, adv_fea), reward, done, info = env.step(action)
                except Exception as e:
                    # 保存视频时可能存在底层bug，捕获异常并终止本次迭代
                    break

                # 如果ego车辆为None，则终止本次迭代
                if env.ego is None:
                    break

                # 更新下一个状态
                next_state = adv_fea

                # 保存当前车道编号
                aft = env.vehicle.lane_index[2]
                av_aft = env.ego.lane_index[2]

                # 如果智能体改变车道，则计数加1
                if pre != aft:
                    agent_lane_c += 1

                # 如果AV改变车道，则计数加1
                if av_pre != av_aft:
                    av_lane_c += 1

                # 累加智能体和AV的行驶距离
                agent_distance += info['distance']
                av_distance += info['ego_distance']

                # 如果使用Running State，则对下一个状态进行处理
                if self.use_running_state:
                    next_state = self.running_state(next_state)

                # 更新当前状态为下一个状态
                state = next_state

                # 如果智能体发生碰撞
                if env.vehicle.crashed:
                    if env.ego.crashed:
                        # 如果AV也发生碰撞，则记录碰撞次数及碰撞数据
                        collision_data = info['collision_data']
                        collision_data['file_name']: select_vehicle_name
                        self.save_log(collision_data, av_crashed, f'{self.load_model_id}_collision_data')
                        av_crashed += 1
                    else:
                        # 如果只有其他车辆发生碰撞，则记录碰撞次数
                        bv_crashed += 1

                # 如果智能体跑出路外，则记录跑出次数
                if not env.vehicle.on_road:
                    offroad += 1

                log = {
                    'epoch': epoch, 'iter': i,
                    'AgentSteering': float(env.vehicle.action['steering']),
                    'AgentAcceleration': float(env.vehicle.action['acceleration']),
                    'AVSteering': float(env.ego.action['steering']),
                    'AVAcceleration': float(env.ego.action['acceleration']),
                    'AgentTTC': float(env.TTC_THW[0][0]), 'AgentTHW': float(env.TTC_THW[0][1]),
                    'AVTTC': float(env.TTC_THW[0][2]), 'AVTHW': float(env.TTC_THW[0][3]),
                    'AgentDistance': float(info['distance']), 'AVDistance': float(info['ego_distance']),
                }

                # utils.print_banner('Eval')
                # for k, v in log.items():
                #     self.logger.record_tabular(k, v)
                # else:
                #     self.logger.dump_tabular()

                self.save_log(log, epoch + i, f'{self.load_model_id}_exp')
                if done:
                    break

        summary_log = {
            'av_crashed': av_crashed,
            'bv_crashed': bv_crashed,
            'agent_lane_c': agent_lane_c,
            'av_lane_c': av_lane_c,
            'skip': skip,
            'offroad': offroad,
            'AgentDistance(KM)': agent_distance / 1000,
            'AVDistance(KM)': av_distance / 1000,
        }

        self.save_log(summary_log, 0, f'{self.load_model_id}_summary_exp')

        for k, v in summary_log.items():
            self.logger.record_tabular(k, v)
        else:
            self.logger.dump_tabular()

        if self.args.upload:
            path = os.path.join(self.output_path, f'{self.load_model_id}_exp.csv')
            data = pd.read_csv(path)
            expert_traj_path = f'{self.args.project_dir}/{self.args.expert_traj_path}'
            expert_traj = pickle.load(open(expert_traj_path, "rb"))
            expert_traj = np.array(expert_traj)
            expert_traj = expert_traj[:, -2:]
            nat_steer = expert_traj[:, 0]
            nat_acc = expert_traj[:, 1]
            steer = data['AgentSteering'].to_numpy()
            acc = data['AgentAcceleration'].to_numpy()
            color = ['sandybrown', 'lightsteelblue']

            fig, axs = plt.subplots(1, 2)
            axs[0].hist(steer, bins=50, color=color[0], density=True, alpha=0.5)
            axs[0].set_xlabel('steering', fontsize=15)
            axs[1].hist(acc, bins=50, color=color[0], density=True, alpha=0.5)
            axs[1].set_xlabel('acceleration', fontsize=15)
            plt.tight_layout()
            self.init_wandb('V2.0-算法训练验证-对抗模型算法', "驾驶动作分布图", self.job_type, upload=self.args.upload)
            self.wandb_run.get_run().log({"Result/Driving Action Distribution": wandb.Image(plt)})
            del self.wandb_run

            bins = 10
            # 计算两个分布的交并比以衡量相似性
            def compute_sim(nat_data, vir_data, bins):
                max_v = max(np.max(nat_data), np.max(vir_data))  # 取自然数据与生成数据的最大值
                min_v = min(np.min(nat_data), np.min(vir_data))  # 取自然数据与生成数据的最小值
                right = math.ceil(max_v) # 对最大值向上取整
                left = math.floor(min_v) # 对最小值向下取整

                # 将生成数据划分成bins组,并获取每组所占整体的比例
                virtual_data_hist = np.histogram(vir_data, bins=np.linspace(left, right, (right - left) * bins),
                                                 density=True)
                # 将真实数据划分成bins组,并获取每组所占整体的比例
                nat_data_hist = np.histogram(nat_data, bins=np.linspace(left, right, (right - left) * bins),
                                             density=True)

                intersection = 0.0 # 统计以bin为单位自然数据与生成数据分布面积并集
                union = 0.0 # 统计以bin为单位自然数据与生成数据分布面积交集
                for i in range(virtual_data_hist[0].shape[0]):
                    intersection += min(virtual_data_hist[0][i], nat_data_hist[0][i]) # 交集就是取两者最小值
                    union += max(virtual_data_hist[0][i], nat_data_hist[0][i]) # 并集就是取两者最大值

                return intersection / union

            steer_sim = compute_sim(nat_steer, steer, bins)
            acc_sim = compute_sim(nat_acc, acc)

            sim_data = np.array([steer_sim, acc_sim]).reshape(1, 2)
            df = pd.DataFrame(sim_data, columns=['Steering', 'Acceleration'])
            self.init_wandb('V2.0-算法训练验证-对抗模型算法', "相似度统计表", self.job_type, upload=self.args.upload)
            self.wandb_run.get_run().log({"Table/Similarity Statistics Table": wandb.Table(dataframe=df)})
            del self.wandb_run

            from visualization_module.corner_case_analysis import draw_cluster_result, make_cluster
            data = pd.read_csv(os.path.join(self.output_path, f'{self.load_model_id}_collision_data.csv'))
            try:
                feature = data.drop('file_name').to_numpy()
            except:
                feature = data.to_numpy()
            finally:
                cluster_features, cluster_labels = make_cluster(feature)
                self.init_wandb('V2.0-评价指标', "聚类分析及视频实例", self.job_type, upload=self.args.upload)
                draw_cluster_result(cluster_features, cluster_labels, f'{self.output_path}', wandb_run=self.wandb_run)
                del self.wandb_run

    def set_adv_env(self, save_videos=False, adversarial_config=None):

        if self.scenario == 'highway':
            vehicle_dir = os.path.join(self.project_dir, self.vehicle_dir)
            vehicle_names = os.listdir(vehicle_dir)[:]
            select_vehicle_name = np.random.choice(vehicle_names, size=1, replace=False)
            path = os.path.join(vehicle_dir, select_vehicle_name[0])
            vehicle_id = select_vehicle_name[0].split('.')[0].split('_')[-1]
            period = 0
            env = HighwayAdEnv(scene='us-101', path=path, period=period, vehicle_id=vehicle_id, IDM=False,
                               adversarial_config=adversarial_config)
            env.duration = 200
            length = env.vehicle.ngsim_traj.shape[0]
            timesteps = list(range(30, length - 200, 1))
            starts = np.random.choice(timesteps, size=5, replace=False)
            if save_videos:
                env = wrappers.Monitor(env,
                                       directory="{}/videos/model_{}_seed_{}/{}_{}/".format(self.output_path, self.load_model_id, self.seed, select_vehicle_name[0],
                                                                                     time.strftime('%Y-%m-%d-%H:%M:%S',
                                                                                                   time.localtime())),
                                       force=True, video_callable=lambda e: True)
                # env.configure({"simulation_frequency": 5})  # Higher FPS for rendering
                env.set_monitor(env)

            env.reset(reset_time=starts[0])
            adv_state = env.adv_features()
            return env, adv_state, select_vehicle_name

    def env_set_seed(self, env):
        if hasattr(env, 'np_random'):
            env.np_random.seed(self.seed)
        if hasattr(env, 'env') and hasattr(env.env, 'np_random'):
            env.env.np_random.seed(self.seed)

    def _exp_with_policy(self, save_name, epochs, show, save_videos, adversarial_config=None):
        # 初始化计数器和记录列表
        ego_crashed = 0  # 跟随车辆碰撞计数器
        bv_crashed = 0  # 被跟随车辆碰撞计数器
        lane_c = 0  # 车道变换计数器
        ego_lane_c = 0  # 跟随车辆车道变换计数器
        skip = 0  # 跳过计数器
        offroad = 0  # 路外行驶计数器
        actions = []  # 行动记录列表，包括跟随车辆和前车的方向盘和油门
        TTC_THW = []  # 时距和时间头way记录列表
        km_actions = []  # 公路行驶行动积累列表
        distance = 0.0  # 跟随车辆行驶距离累计
        ego_distance = 0.0  # 前车行驶距离累计

        for epoch in range(epochs):  # 在指定的迭代次数内执行仿真实验
            print('\r', end='')
            print('progress:{} %'.format((epoch + 1) / epochs * 100), end='')

            env, state, name = self.set_adv_env(save_videos=save_videos, adversarial_config=adversarial_config)  # 创建环境
            self.env_set_seed(env)  # 设置环境

            if env.v_sum > 40 or env.spd_mean < 5:  # 如果跟随车辆速度过高或者平均速度过低则跳过
                skip += 1
                continue

            if self.use_running_state:  # 如果使用running state，则处理状态
                state = self.running_state(state)

            for i in range(200):  # 在每个仿真中执行200步
                state_var = torch.tensor(state).unsqueeze(0)  # 将状态转换为tensor形式，增加一维
                with torch.no_grad():
                    action = self.sample_action(state_var)[0][0].numpy()  # 使用策略模型对状态进行采样以获取动作
                action = action.astype(np.float64)  # 将动作转换为浮点数数组

                # 使用Kalman1D滤波对方向盘和油门进行滤波
                steering = Kalman1D(np.array(action.reshape(1, -1))[:, 0], damping=0.1, tc=0.4).squeeze(-1)[-1]
                acceleration = Kalman1D(np.array(action.reshape(1, -1))[:, 1], damping=0.2, tc=0.5).squeeze(-1)[-1]

                action = [steering, acceleration]  # 更新滤波后的动作
                if show:  # 如果需要显示图像，则渲染环境
                    env.render()

                pre = env.ego.lane_index[2]  # 记录之前的车道
                ego_pre = env.vehicle.lane_index[2]  # 记录前车之前的车道

                (gail_features, adv_fea), reward, done, info = env.step(action)  # 执行动作并获取结果
                next_state = adv_fea  # 更新下一个状态

                aft = env.ego.lane_index[2]  # 记录之后的车道
                ego_aft = env.vehicle.lane_index[2]  # 记录前车之后的车道
                if pre != aft:  # 如果车道变化了
                    lane_c += 1
                if ego_pre != ego_aft:  # 如果前车车道变化了
                    ego_lane_c += 1

                actions.append([env.vehicle.action['steering'], env.vehicle.action['acceleration'],
                                env.ego.action['steering'], env.ego.action['acceleration']])  # 将行动记录添加到列表中

                TTC_THW.append(env.TTC_THW)  # 将时距和时间头way添加到列表中
                distance += info['distance']  # 更新跟随车辆行驶距离
                ego_distance += info['ego_distance']  # 更新前车行驶距离

                if self.running_state is not None:
                    next_state = self.running_state(next_state)

                state = next_state

                if env.ego.crashed:
                    if env.vehicle.crashed:
                        ego_crashed += 1  # 跟随车辆和前车都发生碰撞
                    else:
                        bv_crashed += 1  # 仅跟随车辆发生碰撞

                if not env.ego.on_road:
                    offroad += 1  # 跟随车辆离开了道路

                if done:
                    break  # 当前仿真结束

        # 将实验结果记录到日志中
        self.logger.record_tabular('ego_crashed', ego_crashed)
        self.logger.record_tabular('bv_crashed', bv_crashed)
        self.logger.record_tabular('lane_c', lane_c)
        self.logger.record_tabular('skip', skip)
        self.logger.record_tabular('offroad', offroad)
        self.logger.record_tabular('agent_distance', distance / 1000.)
        self.logger.record_tabular('av_distance', ego_distance / 1000.)
        self.logger.dump_tabular()

        # 将实验结果以字典形式返回
        return {
            "save_name": save_name,
            'actions': actions,
            'TTC_THW': TTC_THW,
            'ego_crashed': ego_crashed,
            'bv_crashed': bv_crashed,
            'lane_c': lane_c,
            'distance': distance / 1000.,
            'ego_distance': ego_distance / 1000
        }

    def exp_with_policy(self, show=False, save_videos=False, epochs=1000):
        # 如果show或save_videos为True，直接调用_exp_with_policy方法进行实验，并返回    #######  adversarial_policy_exp.py
        if show or save_videos:
            self._exp_with_policy('', epochs, show, save_videos)
            return

        # 创建进程池
        ctx = torch.multiprocessing.get_context("spawn")
        p = ctx.Pool(self.num_threads)

        save_video = False
        show = False

        # 创建用于保存实验结果的目录
        policy_dir_name = f'model_{self.load_model_id}_seed_{self.seed}'
        if not os.path.exists(os.path.join(self.output_path, policy_dir_name)):
            os.makedirs(os.path.join(self.output_path, policy_dir_name))

        results = []  # 用于保存每个实验的结果

        # 设置不同的对抗性配置进行多个实验
        for i in range(1, 10, 2):
            for j in range(0, 21, 3):
                utils.print_banner(f"natural adversarial| distance_limit_mi:{i}, time_limit_steps: {j} |")
                adversarial_config = {'distance_limit_mi': i, 'time_limit_steps': j}
                save_name = "dis_{}_time_{}".format(i, j)

                worker_args = (save_name, epochs, show, save_videos, adversarial_config)
                results.append(
                    p.apply_async(self._exp_with_policy, worker_args, callback=None, error_callback=None))

        p.close()
        p.join()

        # 保存实验结果到JSON文件
        for r in results:
            statistic_data = r.get()
            save_name = statistic_data['save_name']
            save([{save_name: statistic_data}],
                 '{}/{}/{}.json'.format(self.output_path, policy_dir_name, save_name))

        # 如果设置了上传参数，执行可视化操作
        if self.args.upload:
            from visualization_module.adversarial_model import visualize_distribution, visualize_collision_rate, \
                show_ttc_proportion
            self.init_wandb('V2.0-评价指标', "TTC/THW分布图", self.job_type, upload=self.args.upload)
            path = os.path.join(self.output_path, policy_dir_name)

            # 可视化TTC和THW的分布情况，并将结果上传至wandb
            visualize_distribution(path, title='Distribution of TTC, MAX Value is 10', max_value=10,
                                   wandb_run=self.wandb_run)
            visualize_distribution(path, title='Distribution of TTC, MAX Value is 100', max_value=100,
                                   wandb_run=self.wandb_run)
            visualize_distribution(path, title='Distribution of THW, MAX Value is 10', max_value=10,
                                   type_name='thw', wandb_run=self.wandb_run)

            # 可视化碰撞率的变化情况，并将结果上传至wandb
            self.init_wandb('V2.0-评价指标', "碰撞率变化图", self.job_type, upload=self.args.upload)
            visualize_collision_rate(path, title='Collision Rate', wandb_run=self.wandb_run)

            # 可视化TTC和THW的占比统计表，并将结果上传至wandb
            self.init_wandb('V2.0-评价指标', "TTC/THW占比统计表", self.job_type, upload=self.args.upload)
            show_ttc_proportion(os.path.join(self.output_path, policy_dir_name), self.wandb_run)
            del self.wandb_run

    def export_to_onnx(self):
        """这段代码用于将模型导出为ONNX格式，并使用OpenVINO进行推理。"""
        print(f'save onnx model in {self.onnx_model_path}.')
        dummy_input = torch.randn(1, self.state_dim)  # 先随机一个模型输入的数据
        start = time.time()
        # 这段代码用于在PyTorch中进行推理，计算模型输出actions。
        actions = self.policy_net(torch.tensor(dummy_input).to(self.type).to(self.device))
        print('total inference time is %d ms in torch.' % (int(round(time.time() * 1000)) - int(round(start * 1000))))
        print(actions)
        # 这行代码调用torch.onnx.export()函数将模型导出为ONNX格式，并保存到指定的路径。
        torch.onnx.export(self.policy_net, dummy_input, self.onnx_model_path, verbose=True)
        # 创建了一个IECore对象，读取导出的ONNX模型，并构建一个网络。设置批量大小为1，并加载网络到CPU设备。
        ie = IECore()
        net = ie.read_network(model=self.onnx_model_path)
        input_blob = next(iter(net.input_info))
        out_blob = next(iter(net.outputs))
        net.batch_size = 1  # batchsize
        exec_net = ie.load_network(network=net, device_name="CPU")
        start = time.time()
        res = exec_net.infer(inputs={input_blob: dummy_input})
        print(' total inference time is %d ms in openvino.' % (int(round(time.time() * 1000)) - int(round(start * 1000))))
        print(res[out_blob][0].tolist())

