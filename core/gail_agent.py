import numpy
import os
import sys

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np
import math
import random
from Utils.zfilter import ZFilter
from Utils.replay_memory import Memory
from Utils.math import Kalman1D
from core.abstract import Agent
from NGSIM_env.envs import NGSIMGAILEnv, InterActionGAILEnv
from models.mlp_discriminator import Discriminator
from models.mlp_policy import Policy
from models.mlp_critic import Value
from Utils.torch import to_device
from Utils import utils
from Utils.logger import Logger, setup_logger
import pickle
from core.ppo import ppo_step
from core.common import estimate_advantages
from core.a2c import a2c_step
from core.trpo import trpo_step
from core.ddpg import ddpg_step
from multiprocessing import cpu_count
from models.random_process import *
import socket
from Utils.server_utils import *
import pandas
class Gail(Agent):

    def __init__(self,
                 args,# 输入参数
                 state_dim: int,# 状态维度
                 action_dim: int,# 动作维度
                 max_action: list,# 动作范围
                 train: bool = True,# 是否训练
                 num_threads: int = 4, # 多进程训练数目
                 scenario: str = 'highway',# 场景类型
                 project_dir: str = None,# 项目路径
                 expert_traj_path: str = 'NGSIM_env/data/ngsim_all_trajs_v4.p',# 专家轨迹数据路径
                 vehicle_dir: str = "NGSIM_env/data/trajectory_set/highway",# 场景数据存储文件夹
                 save_dir: str = "examples/result/data",# 数据保存路径
                 load_model_id: int = 0,# 测试时使用模型id数
                 max_iter_num=500,# 训练总次数
                 batch_size=4096, # 批量大小
                 lr: float = 1e-3, # 初始学习率
                 optim_epochs=10,  # 每个迭代模型更新的次数
                 optim_batch_size=64,# 每次更新采样的数据批量大小
                 discount: float = 0.99, # 折扣率
                 tau: float = 0.95,
                 l2_reg: float = 1e-3,
                 clip_epsilon: float = 0.2,
                 device=torch.device('cpu'), # 设备类型
                 use_running_state=True, # 是否使用running state
                 seed=0, # 设置随机种子
                 save_model_iter=1, # 保存模型的频率
                 eval_iter=1,
                 ):

        super(Gail, self).__init__()
        self.args = args
        self.seed = seed
        torch.set_default_dtype(self.type)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        self.project_dir = project_dir
        self.save_dir = save_dir

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.scenario = scenario

        #for trpo
        self.max_kl = 1e-2
        self.damping = 1e-2

        self.train = train
        self.start_epoch = 0
        self.max_iter_num = max_iter_num
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.discount = discount
        self.tau = tau
        self.l2_reg = l2_reg
        self.clip_epsilon = clip_epsilon

        # optimization epoch number and batch size for PPO
        self.optim_epochs = optim_epochs
        self.optim_batch_size = optim_batch_size

        # 创建判别器
        self.Discriminator = Discriminator(state_dim+action_dim)
        # 创建判别器损失函数
        self.discrim_criterion = torch.nn.BCELoss()
        # 将判别器及损失函数参数导入目标设备中
        to_device(device, self.Discriminator, self.discrim_criterion)
        # 创建优化器
        self.optimizer_discrim = torch.optim.Adam(self.Discriminator.parameters(), lr=self.lr)

        # 添加噪音
        self.random_process = OrnsteinUhlenbeckProcess(size=self.action_dim, theta=self.args.ou_theta,
                                                       mu=self.args.ou_mu, sigma=self.args.ou_sigma)
        # 创建策略网络与价值网络
        self.init_network()

        # 是否动态对状态进行标准化
        self.use_running_state = use_running_state
        self.running_state = ZFilter((self.state_dim, ), clip=5.0) if use_running_state else None
        # 确定进程数
        self.num_threads = min(self.args.num_threads, cpu_count())
        self.save_model_iter = save_model_iter
        self.eval_iter = eval_iter

        self.expert_traj = None
        self.expert_traj_path = expert_traj_path
        self.vehicle_dir = vehicle_dir
        # 导入专家数据
        self.load_expert_traj()

        self.output_path = None
        # 创建实验数据保存文件
        self.make_dir()
        # 如果测试或恢复训练,则
        if not self.train or self.args.resume:
            self.load_model_id = load_model_id
            self.load_model(load_model_id)

        self.logger = None
        self.exp_log()

    def make_dir(self):
        # 确定数据保存路径
        filename = f"{self.__class__.__name__}"
        self.output_path = os.path.join(self.project_dir, self.save_dir, self.scenario, filename, self.args.update_mode, self.args.exp_name)
        if not os.path.exists(self.output_path): # 如果路径不存在则进行创建
            os.makedirs(self.output_path)

    def exp_log(self):
        if self.logger is None:
            self.logger = Logger()
        # 打印实验名及超参数
        variant = {
            'Exp Name': "Gail",
            "Update Mode": self.args.update_mode,
            'Scenario': self.scenario,
            'Train Epochs': self.max_iter_num,
            'Save Model Iter': self.save_model_iter,
            'Eval Iter': self.eval_iter,
            'State Dim': self.state_dim,
            'Action Dim': self.action_dim,
            'Max Action': self.max_action,
            'Batch Size': self.batch_size,
            'Learning Rate': self.lr,
            'Discount': self.discount,
            'Expert Trajectory Path': self.expert_traj_path,
            'Vehicle Dir': self.vehicle_dir,
            'Output Dir': self.output_path,
        }

        for k, v in variant.items():
            self.logger.record_tabular(k, v)

        self.logger.dump_tabular()
    def load_expert_traj(self):
        # 读取专家数据
        expert_traj = pickle.load(open(os.path.join(self.project_dir, self.expert_traj_path), "rb"))
        self.expert_traj = np.array(expert_traj)

    def init_network(self):
        if self.args.update_mode in ['ppo_v2']:
            # 如果 update_mode 是 'ppo_v2'，创建一个 MLP 策略网络（Policy_v2）和一个值网络（Value）
            from models.mlp_policy_v2 import Policy as Policy_v2
            self.policy_net = Policy_v2(self.state_dim, self.action_dim)
            self.value_net = Value(self.state_dim)
            to_device(self.device, self.policy_net, self.value_net)  # 将模型移动到指定的设备上
            self.optimizer_value = torch.optim.Adam(self.value_net.parameters(), lr=self.lr)  # 初始化值网络的 Adam 优化器
            self.value_step_lr = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_value, self.args.num_epochs,
                                                                            1e-4)  # 初始化值网络的学习率调度器
            self.optimizer_policy = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)  # 初始化策略网络的 Adam 优化器
            self.policy_step_lr = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_policy,
                                                                             self.args.num_epochs,
                                                                             1e-4)  # 初始化策略网络的学习率调度器
        else:
            # 如果 update_mode 不是 'ppo_v2' 或 'ddpg'，创建一个值网络（Value）和一个策略网络（Policy）
            self.value_net = Value(self.state_dim)  # 创建值网络
            self.policy_net = Policy(self.state_dim, self.action_dim)  # 创建策略网络
            self.optimizer_value = torch.optim.Adam(self.value_net.parameters(), lr=self.lr)  # 初始化值网络的 Adam 优化器
            self.optimizer_policy = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)  # 初始化策略网络的 Adam 优化器

            to_device(self.device, self.policy_net, self.value_net)  # 将模型移动到指定的设备上

    def set_env(self):
        if self.scenario == 'highway':
            # 如果场景是 'highway'
            vehicle_dir = os.path.join(self.project_dir, self.vehicle_dir)  # 车辆数据目录路径
            vehicle_names = os.listdir(vehicle_dir)[:]  # 车辆数据文件名列表
            select_vehicle_name = np.random.choice(vehicle_names, size=1, replace=False)  # 随机选择一个车辆数据文件
            path = os.path.join(vehicle_dir, select_vehicle_name[0])  # 车辆数据文件完整路径
            vehicle_id = select_vehicle_name[0].split('.')[0].split('_')[-1]  # 提取车辆ID
            period = 0
            env = NGSIMGAILEnv(scene='us-101', path=path, period=period, vehicle_id=vehicle_id, IDM=False,
                               gail=True)  # 创建高速公路驾驶场景环境对象
            env.duration = 200
            env.reset(reset_time=0)  # 重置环境
            gail_state = env.gail_features_v2()  # 获取GAIL特征表示的当前状态

            return env, gail_state, select_vehicle_name[0].split('.')[0]  # 返回环境对象、当前环境状态和选中车辆的名称

        elif self.scenario == 'intersection':
            # 如果场景是 'intersection'
            vehicle_dir = os.path.join(self.project_dir, self.vehicle_dir)  # 车辆数据目录路径
            vehicle_names = os.listdir(vehicle_dir)[:]  # 车辆数据文件名列表
            select_vehicle_name = np.random.choice(vehicle_names, size=1, replace=False)  # 随机选择一个车辆数据文件
            path = os.path.join(vehicle_dir, select_vehicle_name[0])  # 车辆数据文件完整路径
            vehicle_id = select_vehicle_name[0].split('.')[0].split('_')[-1]  # 提取车辆ID
            env = InterActionGAILEnv(path=path, vehicle_id=vehicle_id, IDM=False, render=False,
                                     gail=True)  # 创建路口交互场景环境对象
            env.duration = 200
            env.reset(reset_time=0)  # 重置环境
            gail_state = env.gail_features()  # 获取GAIL特征表示的当前状态

            return env, gail_state, select_vehicle_name[0].split('.')[0]  # 返回环境对象、当前环境状态和选中车辆的名称

        else:
            pass

    def update_params(self, batch, i_iter):
        next_states = torch.from_numpy(np.stack(batch.next_state)).to(self.type).to(self.device)
        # 将next_state转换为张量，并移动到指定设备

        states = np.stack(batch.state)
        # 将state堆叠为一个numpy数组

        actions = np.stack(batch.action)
        # 将action堆叠为一个numpy数组

        rewards = self.expert_reward(states, actions, self.device)
        # 使用expert's策略计算给定状态和动作的奖励

        states = torch.from_numpy(states).to(self.type).to(self.device)
        # 将states转换为张量，并移动到指定设备

        actions = torch.from_numpy(actions).to(self.type).to(self.device)
        # 将actions转换为张量，并移动到指定设备

        masks = torch.from_numpy(np.stack(batch.mask)).to(self.type).to(self.device)
        # 将mask转换为张量，并移动到指定设备

        with torch.no_grad():
            values = self.value_net(states)
            fixed_log_probs = self.policy_net.get_log_prob(states, actions)
        # 使用value网络计算状态的值函数，并使用policy网络计算固定动作的对数概率

        """从轨迹中计算优势估计"""
        advantages, returns = estimate_advantages(rewards, masks, values, self.discount, self.tau, self.device)
        # 使用奖励、掩码、值函数、折扣因子和GAE估计优势和回报
        feak_discrim_loss = real_discrim_loss = None
        """更新判别器"""
        for _ in range(1):
            # 将专家数据加载到设备中
            expert_state_actions = torch.from_numpy(self.expert_traj).to(self.type).to(self.device)
            # 判别器对生成数据进行打分
            g_o = self.Discriminator(torch.cat([states, actions], 1))
            # 判别器对真实数据进行打分
            e_o = self.Discriminator(expert_state_actions)
            # 判别器优化器梯度清零
            self.optimizer_discrim.zero_grad()
            # 计算生成数据损失
            feak_discrim_loss = self.discrim_criterion(g_o, torch.ones((states.shape[0], 1), device=self.device))
            # 计算真实数据数据损失
            real_discrim_loss = self.discrim_criterion(e_o,
                                                       torch.zeros((self.expert_traj.shape[0], 1), device=self.device))
            # 将真实数据损失与生成数据损失两者相加,进行梯度反向传播
            discrim_loss = feak_discrim_loss + real_discrim_loss
            discrim_loss.backward()
            self.optimizer_discrim.step()

        """执行小批量PPO更新"""
        # 进行多次优化以执行PPO更新
        optim_iter_num = int(math.ceil(states.shape[0] / self.optim_batch_size))
        for _ in range(self.optim_epochs):
            perm = np.arange(states.shape[0])
            # 对索引进行打乱
            np.random.shuffle(perm)
            # 将perm转换为long类型张量，并移动到指定设备
            perm = torch.LongTensor(perm).to(self.device)
            # 将数据通过索引的方式进行打乱
            states, actions, returns, advantages, fixed_log_probs = \
                states[perm].clone(), actions[perm].clone(), returns[perm].clone(), advantages[perm].clone(), \
                    fixed_log_probs[perm].clone()

            for i in range(optim_iter_num):
                # 从打乱的数据采样小批量数据
                ind = slice(i * self.optim_batch_size, min((i + 1) * self.optim_batch_size, states.shape[0]))
                states_b, actions_b, advantages_b, returns_b, fixed_log_probs_b = \
                    states[ind], actions[ind], advantages[ind], returns[ind], fixed_log_probs[ind]
                # 对PPO的价值网络与策略网络进行更新
                ppo_step(self.policy_net, self.value_net, self.optimizer_policy, self.optimizer_value, 1, states_b,
                         actions_b, returns_b, advantages_b, fixed_log_probs_b, self.clip_epsilon, self.l2_reg)

        # 计算统计量，例如平均奖励、奖励标准差、标准误差和鉴别器损失
        rewards_numpy = rewards.detach().numpy()
        mean_ = float(np.mean(rewards_numpy))
        std = float(np.std(rewards_numpy, ddof=0))
        std_error = float(std / np.sqrt(rewards_numpy.shape[0]))
        log = {
            'reward_mean': mean_,
            'reward_std': std,
            'reward_std_error': std_error,
            'feak_discrim_loss': float(feak_discrim_loss.detach().cpu().numpy()),
            'real_discrim_loss': float(real_discrim_loss.detach().cpu().numpy())
        }

        return log

    def collect_samples(self, pid, env, policy, custom_reward, min_batch_size, device, train, eps, random_process):
        """用于收集训练数据"""
        # 为每个进程设置随机种子
        if pid > 0:
            torch.manual_seed(torch.randint(0, 5000, (1,)) * pid)
            if hasattr(env, 'np_random'):
                env.np_random.seed(env.np_random.randint(5000) * pid)
            if hasattr(env, 'env') and hasattr(env.env, 'np_random'):
                env.env.np_random.seed(env.env.np_random.randint(5000) * pid)

        # 初始化用于跟踪统计信息的变量
        memory = Memory()
        num_steps = 0 #
        total_reward = 0
        min_reward = 1e6
        max_reward = -1e6
        total_c_reward = 0
        min_c_reward = 1e6
        max_c_reward = -1e6
        num_episodes = 0

        # 循环直到步数超过最小批处理大小
        while num_steps < min_batch_size:

            # 获取环境和状态
            env, gail_state, _ = self.set_env()

            # 如果gail_state为None且场景为'highway'或'intersection'，则跳过下一次循环
            if self.scenario == 'highway' and gail_state is None:
                continue
            elif self.scenario == 'intersection' and (gail_state is None):
                continue

            # 如果存在运行状态变换函数，则将其应用于gail_state
            if self.running_state is not None:
                state = self.running_state(gail_state)
            else:
                state = gail_state

            reward_episode = 0

            # 内嵌循环用于与环境进行交互
            for t in range(10000):
                state_var = torch.tensor(state).unsqueeze(0)
                action = None
                with torch.no_grad():
                    if not train:
                        # 在评估模式下使用策略网络生成动作
                        action = policy(state_var.to(device))[0].cpu().numpy()
                    else:
                        # 在训练模式下对其他算法选择动作
                        actions = policy.select_action(state_var.to(device))
                        actions = to_device(torch.device('cpu'), *actions)
                        action = actions[0][0].numpy()

                # 如果动作类型是离散的，则将动作转换为整数，否则将其转换为浮点数
                action = int(action) if self.args.action_type == 'discrete' else action.astype(np.float64)

                # 将动作应用于环境并获取下一个状态、奖励、完成标志和其他信息
                (next_gail_state, next_adv_state), reward, done, _ = env.step(action)

                # 更新该回合的奖励，并将运行状态转换应用于next_gail_state
                reward_episode += reward
                if self.running_state is not None:
                    next_state = self.running_state(next_gail_state)
                else:
                    next_state = next_gail_state

                # 根据完成标志计算掩码值
                mask = 0 if done else 1

                # 将状态、动作、掩码、下一个状态、奖励和动作（如果有）添加到内存对象中
                memory.push(state, action, mask, next_state, reward, actions)

                # 如果该回合结束，则退出内嵌循环
                if done:
                    break

                state = next_state

            # 记录统计信息
            num_steps += (t + 1)
            num_episodes += 1
            total_reward += reward_episode
            min_reward = min(min_reward, reward_episode)
            max_reward = max(max_reward, reward_episode)

        # 创建包含统计信息的字典
        log = {
            'num_steps': num_steps,
            'num_episodes': num_episodes,
            'env_total_reward': total_reward,
            'env_avg_reward': total_reward / num_episodes,
            'env_avg_step_reward': total_reward / num_steps,
            'env_max_reward': max_reward,
            'env_min_reward': min_reward,
        }

        return pid, memory, log

    def expert_reward(self, state, action, device):
        # 将状态与动作进行拼接
        state_action = torch.tensor(np.hstack([state, action]), dtype=self.type)
        with torch.no_grad():
            # 使用判别器输出生成数据真实性评价,然后将输出转换到负对数空间中
            return -torch.log(self.Discriminator(state_action.to(device))).squeeze()

    # 计算epsilon的函数，根据迭代次数进行epsilon的衰减
    def calculate_epsilon(self, iter_i, eps_start=1.0, eps_final=0.01, eps_decay=50.):
        return eps_final + (eps_start - eps_final) * math.exp(-1. * iter_i / eps_decay)

    # 定义训练函数learn，并带有一个可选的debug参数
    def learn(self, debug=False):

        i_iter = self.start_epoch
        update_function = {
            'ppo': self.update_params
        }
        while i_iter < self.max_iter_num:
            # 循环两次，分别为训练和评估阶段
            for j in range(2):
                # 根据j的值设置self.train标志，表示当前是否为训练阶段
                self.train = True if j % 2 == 0 else False
                if not self.train:
                    break

                # 记录开始时间
                t_start = time.time()

                # 设置每个线程的batch大小
                thread_batch_size = int(math.floor(self.batch_size / self.num_threads))
                ctx = torch.multiprocessing.get_context("spawn")
                # 创建一个进程池
                p = ctx.Pool(self.num_threads)
                results = []

                # 计算当前迭代的epsilon值
                eps = self.calculate_epsilon(i_iter, eps_decay=self.max_iter_num * 0.2)

                # 调试模式下，创建环境并收集样本数据，用于测试更新函数的效果并计算时间
                if debug:
                    env, _, _ = self.set_env()
                    pid, worker_memory, worker_log = self.collect_samples(0, env, self.policy_net, self.expert_reward,
                                                                          256, self.device, self.train, eps,
                                                                          self.random_process)
                    memory = Memory()
                    memory.append(worker_memory)
                    print(time.time() - t_start)
                    batch = memory.sample()
                    train_log = update_function[self.args.update_mode](batch, i_iter)

                # 使用进程池创建多个线程，每个线程执行collect_samples函数采集样本数据
                for i in range(self.num_threads):
                    env, _, _ = self.set_env()
                    worker_args = (
                    i, env, self.policy_net, self.expert_reward, thread_batch_size, self.device, self.train, eps,
                    self.random_process)
                    results.append(p.apply_async(self.collect_samples, worker_args, callback=None, error_callback=None))

                # 关闭进程池，并等待所有线程执行完毕
                p.close()
                p.join()

                # 合并所有线程收集到的样本数据
                memory = Memory()
                worker_logs = [None] * self.num_threads
                worker_memories = [None] * self.num_threads
                for r in results:
                    pid, worker_memory, worker_log = r.get()
                    worker_memories[pid] = worker_memory
                    worker_logs[pid] = worker_log
                for worker_memory in worker_memories:
                    memory.append(worker_memory)
                batch = memory.sample()

                # 合并所有线程的日志
                log_list = worker_logs
                log = self.merge_log(log_list)

                # 如果是训练阶段，执行更新函数，并更新训练日志
                if self.train:
                    print(time.time() - t_start)
                    train_log = update_function[self.args.update_mode](batch, i_iter)
                    log.update(train_log)
                    self.logger.record_tabular('Train Epochs', i_iter)
                    log['sample_time'] = time.time() - t_start
                    self.save_log(log, i_iter, 'train')
                else:
                    # 如果是评估阶段，保存评估日志
                    self.logger.record_tabular('Eval Epochs', i_iter)
                    log['sample_time'] = time.time() - t_start
                    self.save_log(log, i_iter, 'eval')

                # 记录日志到logger中
                for k, v in log.items():
                    self.logger.record_tabular(k, v)
                self.logger.dump_tabular()

                # 将策略网络传输到指定设备（通常为GPU）
                to_device(self.device, self.policy_net)

                # 如果迭代次数达到保存模型的间隔，则保存模型
                if (i_iter + 1) % self.save_model_iter == 0 and self.train:
                    self.save_model(i_iter)

            # 迭代次数加1
            i_iter += 1
            # 清空GPU缓存
            torch.cuda.empty_cache()

    def save_log(self, log: dict, iter: int, filename: str):
        if iter == 0:  # 如果迭代次数为0
            with open(os.path.join(self.output_path, f'{filename}.csv'), 'w+') as f:  # 以写入模式打开文件
                head = []
                for k, v in log.items():
                    if k in ['sample_time', 'action_mean', 'action_min', 'action_max']:
                        continue  # 跳过不需要保存的键
                    head.append(k)  # 将需要保存的键添加到头部
                head_ = ','.join(head)
                f.write(head_)  # 将头部写入文件
                f.write('\n')

        with open(os.path.join(self.output_path, f'{filename}.csv'), 'a+') as f:  # 以追加模式打开文件
            data = []
            for k, v in log.items():
                if k in ['sample_time', 'action_mean', 'action_min', 'action_max']:
                    continue  # 跳过不需要保存的键
                data.append(str(v))  # 将需要保存的值加入列表
            data_ = ','.join(data)
            f.write(data_)  # 将数据行写入文件
            f.write('\n')

    # np.savetxt(filename, data, fmt='%d', delimiter=None)

    def merge_log(self, log_list):
        # 数据合并
        log = dict()
        log['env_total_reward'] = float(sum([x['env_total_reward'] for x in log_list]))
        log['num_episodes'] = float(sum([x['num_episodes'] for x in log_list]))
        log['num_steps'] = float(sum([x['num_steps'] for x in log_list]))
        log['env_avg_reward'] = float(log['env_total_reward'] / log['num_episodes'])
        log['env_avg_step_reward'] = float(log['env_total_reward'] / log['num_steps'])
        log['env_max_reward'] = float(max([x['env_max_reward'] for x in log_list]))
        log['env_min_reward'] = float(min([x['env_min_reward'] for x in log_list]))

        return log

    def sample_action(self, state):
        # 根据状态估计动作
        with torch.no_grad():
            actions = self.policy_net(state.to(self.device))
            return actions

    def save_model(self, epoch):
        utils.print_banner(f"保存权重到 {self.output_path}")
        ckpts_state = {
            'start_epoch': epoch,  # 保存当前的epoch
            'running_state': self.running_state,  # 保存运行状态
            'policy_net': self.policy_net.state_dict(),  # 保存策略网络的状态
            'Discriminator': self.Discriminator.state_dict(),  # 保存鉴别器的状态
            'value_net': self.value_net.state_dict(),  # 保存价值网络的状态
            'optimizer_discrim': self.optimizer_discrim.state_dict(),  # 保存鉴别器优化器的状态
            'optimizer_policy': self.optimizer_policy.state_dict(),  # 保存策略网络优化器的状态
            'optimizer_value': self.optimizer_value.state_dict()  # 保存价值网络优化器的状态
        }
        if self.args.update_mode in ['ddpg']:
            ckpts_state.update(
                {
                    'value_net_target': self.value_net_target.state_dict(),  # 保存目标价值网络的状态
                    'policy_net_target': self.policy_net_target.state_dict(),  # 保存目标策略网络的状态
                }
            )
        utils.save_checkpoints(ckpts_state, self.output_path,
                               model_name=f'{epoch}_{self.args.update_mode}_model')  # 保存模型的状态

    def load_model(self, epoch):
        if self.train:
            model_path = os.path.join(self.output_path,
                                      f'latest_{self.args.supervise_model}_model_ckpt.pth')  # 设置模型路径，用于训练模式
        else:
            model_path = os.path.join(self.output_path, f'{epoch}_ppo_model_ckpt.pth')  # 设置模型路径，用于测试模式
        assert os.path.exists(model_path)  # 断言模型路径存在
        utils.print_banner(f'从 {model_path} 加载模型.')  # 打印加载模型的提示信息
        ckpt = torch.load(model_path, map_location=self.device)  # 加载模型的checkpoint
        self.policy_net.load_state_dict(ckpt['policy_net'])  # 从checkpoint中加载策略网络的状态
        self.value_net.load_state_dict(ckpt['value_net'])  # 从checkpoint中加载价值网络的状态
        self.Discriminator.load_state_dict(ckpt['Discriminator'])  # 从checkpoint中加载鉴别器的状态
        self.optimizer_discrim.load_state_dict(ckpt['optimizer_discrim'])  # 从checkpoint中加载鉴别器优化器的状态
        self.optimizer_policy.load_state_dict(ckpt['optimizer_policy'])  # 从checkpoint中加载策略网络优化器的状态
        self.optimizer_value.load_state_dict(ckpt['optimizer_value'])  # 从checkpoint中加载价值网络优化器的状态

        if self.args.update_mode in ['ddpg']:
            self.value_net_target.load_state_dict(ckpt['value_net_target'])  # 从checkpoint中加载目标价值网络的状态
            self.policy_net_target.load_state_dict(ckpt['policy_net_target'])  # 从checkpoint中加载目标策略网络的状态

        if self.train:
            self.start_epoch = ckpt['start_epoch'] + 1  # 如果是训练模式，设置开始的epoch为加载的epoch加1

    def exp(self, save_video=False, show=False, epochs=1000):
        ego_crashed = 0  # 自车碰撞次数
        lane_c = 0  # 车道变更次数
        skip = 0  # 跳过的次数
        offroad = 0  # 离开道路的次数
        actions = []  # 动作列表
        TTC_THW = []  # 时间-碰撞距离和时间-车头距离列表
        distance = 0.0  # 行驶距离
        for epoch in range(epochs):  # 迭代每个epoch
            utils.print_banner('progress:{} %'.format((epoch + 1) / epochs * 100))  # 打印进度
            env, state, select_name = self.set_env()  # 设置环境并获取初始状态

            if state is None:  # 如果初始状态为None，则跳过当前epoch
                skip += 1
                continue

            for i in range(500):  # 迭代每个epoch的每个步骤
                state_var = torch.tensor(state).unsqueeze(0)  # 将状态转换为张量
                action = self.sample_action(state_var)[0][0].detach().numpy()  # 根据当前状态获取动作

                if show:  # 如果show为True，则渲染环境以可视化进度
                    env.render()

                pre = env.vehicle.lane_index[2]  # 获取之前的车道索引

                (gail_features, adv_fea), reward, done, info = env.step(action)  # 执行动作并获取环境返回的结果
                next_state = gail_features  # 更新下一个状态为GAIL特征
                aft = env.vehicle.lane_index[2]  # 获取之后的车道索引
                if pre != aft:  # 如果车道索引发生变化，则增加车道变更次数
                    lane_c += 1

                actions.append([env.vehicle.action['steering'], env.vehicle.action['acceleration']])  # 将动作添加到动作列表中

                TTC_THW.append(env.TTC_THW)  # 将时间-碰撞距离和时间-车头距离添加到TTC_THW列表中
                distance += info['distance']  # 增加行驶距离

                state = next_state  # 更新当前状态为下一个状态

                if env.vehicle.crashed:  # 如果自车碰撞，则增加自车碰撞次数
                    ego_crashed += 1

                if not env.vehicle.on_road:  # 如果离开道路，则增加离开道路次数
                    offroad += 1

                log = {  # 创建日志字典
                    'epoch': epoch, 'iter': i,
                    'steering': float(env.vehicle.action['steering']),
                    'acceleration': float(env.vehicle.action['acceleration']),
                    'TTC': float(env.TTC_THW[0][0]), 'THW': float(env.TTC_THW[0][1]),
                    'distance': float(info['distance']), 'ego_crashed': int(env.vehicle.crashed),
                    'offroad': int(not env.vehicle.on_road)
                }

                self.save_log(log, epoch + i, f'{self.load_model_id}_exp')  # 保存日志

                if done:  # 如果episode完成，则跳出循环
                    break

        summary_log = {  # 创建总结日志字典
            'ego_crashed': ego_crashed,
            'lane_c': lane_c,
            'skip': skip,
            'offroad': offroad,
            'vehicle_distance(KM)': distance / 1000,
        }

        self.save_log(summary_log, 0, f'{self.load_model_id}_summary_exp')  # 保存总结日志

        self.logger.record_tabular('ego_crashed', ego_crashed)  # 更新记录的统计数据
        self.logger.record_tabular('lane_c', lane_c)
        self.logger.record_tabular('skip', skip)
        self.logger.record_tabular('offroad', offroad)
        self.logger.record_tabular('vehicle_distance', distance / 1000.)
        self.logger.dump_tabular()  # 打印和清空记录的统计数据