import os
import random
import sys
import argparse
import pickle
import numpy as np
import pandas
import torch

import Utils
from models.bc_diffusion import Diffusion_BC as Agent
from Utils import utils
from Utils.logger import logger, setup_logger
import json
from Utils.zfilter import ZFilter
from Utils.replay_memory import Memory
from NGSIM_env.envs import NGSIMGAILEnv, InterActionGAILEnv
# from NGSIM_env.envs import HighwayEnv
from models.ql_diffusion import Diffusion_QL
from models.bc_diffusion import Diffusion_BC
from gym import wrappers
from core.abstract import Agent
from Utils.logger import Logger, setup_logger
from Utils.math import Kalman1D
import pickle
from multiprocessing import cpu_count
# 导入套接字模块
import socket
# 导入线程模块
import threading
# 导入时间
import time
import pygame
from Utils.server_utils import *
dtype = torch.float64
torch.set_default_dtype(dtype)
import matplotlib.pyplot as plt
import wandb
import pandas as pd
import math


class DataSample(object):
    """从人类专家数据中采样数据"""
    def __init__(self, path='/NGSIM_env/data/ngsim_all_trajs_v4.p', device=torch.device('cpu')):
        utils.print_banner(f'Load data from {path}')
        assert os.path.exists(path), print(path) # 判断文件是否存在
        self.expert_traj = pickle.load(open(path, "rb")) # 读取专家数据
        self.expert_traj = np.array(self.expert_traj)
        self.state = torch.from_numpy(self.expert_traj[:, :-2]) # 获取状态数据
        self.action = torch.from_numpy(self.expert_traj[:, -2:]) # 获取动作数据
        self.device = device # 数据存储的设备

    def sample(self, start, batch_size=1024):
        # ind = torch.range(start, start + batch_size - 1).to(torch.int64)
        ind = torch.randint(0, self.expert_traj.shape[0], size=(batch_size,)) # 随即生成batch_size个索引
        # 返回索引对应的数据
        return (
            self.state[ind].to(self.device),
            self.action[ind].to(self.device),
        )

    def __len__(self):
        return self.expert_traj.shape[0] # 返回数据集大小


class DiffusionAgent(Agent):

    def __init__(self,
                 args, # 输入参数
                 state_dim: int, # 状态维度
                 action_dim: int, # 动作维度
                 max_action: list, # 动作范围
                 train: bool = True, # 是否训练
                 num_steps_per_epoch=20, # 每个迭代训练多少个轮次
                 batch_size=256, # 批量大小
                 lr_decay=True, # 是否使用学习率衰减
                 discount=0.99, # 奖励的折扣率
                 tau=0.005,
                 T=16, # 扩散模型扩散步长
                 beta_schedule='linear', # 学习率衰减类型
                 algo='bc', # 算法类型,默认行为克隆
                 num_epochs=2000, # 总的训练次数
                 eval_freq=20,
                 eval_episodes=20,
                 lr=2e-3, # 初始学习率
                 eta=1.0,
                 max_q_backup=True, # 奖励训练方法
                 top_k=1,
                 gn=1,
                 clip_denoised=True,
                 scenario: str = 'highway', # 场景类型
                 project_dir: str = None, # 项目路径
                 expert_traj_path: str = 'NGSIM_env/data/ngsim_all_trajs_v4.p', # 专家轨迹数据路径
                 vehicle_dir: str = "NGSIM_env/data/trajectory_set/highway", # 场景数据存储文件夹
                 save_dir: str = "examples/result/data", # 数据保存路径
                 load_model_id: int = 0, # 测试时使用模型id数
                 device=torch.device('cpu'), # 设备类型
                 seed=0, # 随机种子
                 save_model_iter=1, # 保存模型的频率
                 eval_iter=1,
                 ):
        super(DiffusionAgent, self).__init__()
        self.args = args
        self.device = device
        self.seed = seed
        self.algo = algo
        torch.set_default_dtype(self.type)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        # 数据采样器
        self.data_sampler = DataSample(path=os.path.join(project_dir, expert_traj_path), device=device)
        self.project_dir = project_dir
        self.save_dir = save_dir

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.scenario = scenario

        self.train = train
        self.start_epoch = 0
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.discount = discount
        self.tau = tau
        self.eta = eta
        self.max_q_backup = max_q_backup
        self.top_k = top_k
        self.num_steps_per_epoch = num_steps_per_epoch
        self.lr_decay = lr_decay
        self.T = T
        self.gn = gn
        self.clip_denoised = clip_denoised
        self.beta_schedule = beta_schedule

        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.save_model_iter = save_model_iter
        self.eval_iter = eval_iter

        self.expert_traj = None
        self.expert_traj_path = expert_traj_path
        self.vehicle_dir = vehicle_dir

        self.output_path = None
        # 创建文件夹
        self.make_dir()
        # 创建模型
        self.create_model()
        # 测试时或恢复训练需要重新加载模型
        if not self.train or self.args.resume:
            self.load_model_id = load_model_id
            self.load_model(load_model_id)

        self.logger = None # 日志对象
        self.exp_log() # 打印实验类型

    def make_dir(self):
        """设置数据保存路径"""
        filename = f"{self.__class__.__name__}"
        self.output_path = os.path.join(self.project_dir, self.save_dir, self.scenario, filename, self.args.exp_name)
        if not os.path.exists(self.output_path): # 如果路径不存在需要创建
            os.makedirs(self.output_path)

    def set_env(self):
        """创建仿真环境"""
        if self.scenario == 'highway':
            # 获取所有场景文件名
            vehicle_dir = os.path.join(self.project_dir, self.vehicle_dir)
            vehicle_names = os.listdir(vehicle_dir)[:]
            # 随机挑选场景文件
            select_vehicle_name = np.random.choice(vehicle_names, size=1, replace=False)
            path = os.path.join(vehicle_dir, select_vehicle_name[0])
            vehicle_id = select_vehicle_name[0].split('.')[0].split('_')[-1]
            period = 0
            # 创建基于场景数据的仿真环境
            env = NGSIMGAILEnv(scene='us-101', path=path, period=period, vehicle_id=vehicle_id, IDM=False, gail=True)
            env.reset(reset_time=0)
            # 获取交通流模型输入状态数据
            gail_state = env.gail_features_v2()
            return env, gail_state, select_vehicle_name
        elif self.scenario == 'intersection':
            # 获取所有场景文件名
            vehicle_dir = os.path.join(self.project_dir, self.vehicle_dir)
            vehicle_names = os.listdir(vehicle_dir)[:]
            # 随机挑选场景文件
            select_vehicle_name = np.random.choice(vehicle_names, size=1, replace=False)
            path = os.path.join(vehicle_dir, select_vehicle_name[0])
            vehicle_id = select_vehicle_name[0].split('.')[0].split('_')[-1]
            # 创建基于场景数据的仿真环境
            env = InterActionGAILEnv(path=path, vehicle_id=vehicle_id, IDM=False, render=False, gail=True)
            env.duration = 200
            env.reset(reset_time=0)
            # 获取交通流模型输入状态数据
            gail_state = env.gail_features()
            return env, gail_state, select_vehicle_name
        else:
            pass

    def create_model(self):
        if self.algo == "bc":
            # 创建基于扩散模型的行为克隆模型
            self.agent = Diffusion_BC(state_dim=self.state_dim, # 状态维度
                          action_dim=self.action_dim, # 动作维度
                          max_action=self.max_action, # 动作范围
                          device=self.device, # 设备类型
                          discount=self.discount, # 折扣率
                          tau=self.tau,
                          clip_denoised=self.clip_denoised,
                          beta_schedule=self.beta_schedule,
                          n_timesteps=self.T,
                          lr=self.lr)

        elif self.algo == 'ql':
            pass
            # self.agent = Diffusion_QL(state_dim=self.state_dim,
            #               action_dim=self.action_dim,
            #               max_action=self.max_action,
            #               device=self.device,
            #               discount=self.discount,
            #               tau=self.tau,
            #               max_q_backup=self.max_q_backup,
            #               beta_schedule=self.beta_schedule,
            #               n_timesteps=self.T,
            #               eta=self.eta,
            #               lr=self.lr,
            #               lr_decay=self.lr_decay,
            #               lr_maxt=self.num_epochs,
            #               grad_norm=self.gn)

    def learn(self):
        """算法训练"""
        # 如果需要将实验数据上传wandb,则需要创建wandb对象
        if self.args.upload:
            self.init_wandb('V2.0-算法训练验证-交通流算法', "Loss曲线变化图", self.job_type, upload=self.args.upload)

        early_stop = False
        writer = None

        evaluations = []

        metric = 100.
        utils.print_banner(f"Training Start", separator="*", num_star=90)
        while (self.start_epoch < self.num_epochs) and (not early_stop):
            # 训练扩散模型
            loss_metric = self.agent.train(self.data_sampler,
                                      batch_size=self.batch_size,
                                      log_writer=writer)
            self.save_model(self.start_epoch)
            # Logging
            logger.record_tabular('Trained Epochs', self.start_epoch)
            logger.record_tabular('Batch Size', self.batch_size)
            logger.record_tabular('Learning Rate', self.agent.policy_step_lr.get_lr())
            logger.record_tabular('BC Loss', np.mean(loss_metric['bc_loss']))
            logger.record_tabular('QL Loss', np.mean(loss_metric['ql_loss']))
            logger.record_tabular('Actor Loss', np.mean(loss_metric['actor_loss']))
            logger.record_tabular('Critic Loss', np.mean(loss_metric['critic_loss']))
            logger.dump_tabular()
            train_log = {'BC Loss': float(np.mean(loss_metric['bc_loss'])), 'QL Loss': float(np.mean(loss_metric['ql_loss'])),
                         'Actor Loss': float(np.mean(loss_metric['actor_loss'])), 'Critic Loss': float(np.mean(loss_metric['critic_loss']))}
            self.save_log(train_log, self.start_epoch, 'train')
            for k,v in train_log.copy().items():
                train_log[k] = [v]

            # 上传训练数据到wandb
            if self.args.upload:
                df = pandas.DataFrame.from_dict(train_log)
                df = df.drop(columns=['QL Loss', 'Actor Loss', 'Critic Loss'])
                self.upload_data_to_wandb_server('Result', df)

            self.start_epoch += 1

        if self.args.upload:
            del self.wandb_run

    def save_log(self, log: dict, iter: int, filename: str):
        """保存实验过程中的数据"""
        if iter == 0:
            # 初始时需要保存数据列名
            with open(os.path.join(self.output_path, f'{filename}.csv'), 'w+') as f:
                head = []
                for k, v in log.items():
                    head.append(k)
                head_ = ','.join(head)
                f.write(head_)
                f.write('\n')
        # 以追加的方式保存实验数据
        with open(os.path.join(self.output_path, f'{filename}.csv'), 'a+') as f:
            data = []
            for k, v in log.items():
                data.append(str(v))
            data_ = ','.join(data)
            f.write(data_)
            f.write('\n')

    def eval_policy(self, policy, seed, eval_episodes=100):
        """对模型进行测试"""
        scores = []
        traj_lengths = []
        for _ in range(self.eval_episodes):
            # 创建仿真环境
            eval_env, state, _ = self.set_env()

            traj_return = 0.
            traj_len = 0
            done = False
            while not done:
                # 根据当前状态获取模型预测动作
                action = policy.sample_action(np.array(state))
                # 将动作作用于环境以获得奖励及下个环境状态
                (state, _), reward, done, _ = eval_env.step(action)
                traj_return += reward
                traj_len += 1
            scores.append(traj_return)
            traj_lengths.append(traj_len)

        avg_reward = np.mean(scores)
        std_reward = np.std(scores)
        avg_traj_len = np.mean(traj_lengths)
        std_traj_len = np.std(traj_lengths)

        utils.print_banner(f"Evaluation over {eval_episodes} episodes: avg_reward {avg_reward:.2f}, std_reward{std_reward:.2f}, "
                           f"avg_traj_len {avg_traj_len:.2f}, std_traj_len {std_traj_len:.2f}.")

        return avg_reward, std_reward, avg_traj_len, std_traj_len

    def sample_action(self, state):
        """根据状态获取模型输出"""
        with torch.no_grad():# 不记录梯度
            action = self.agent.actor.sample(state) # 获取动作
            return action.cpu().data.numpy().flatten() # 将动作从设备中导出到cpu,并转化为numpy数据

    def save_model(self, epoch):
        """模型保存"""
        ckpts_state = {
            'epoch': epoch, # 当前训练轮次
            'actor': self.agent.actor.state_dict(), # 获取策略网络的权重
            'actor_optimizer': self.agent.actor_optimizer.state_dict(), # 获取优化器的权重
            'policy_step_lr': self.agent.policy_step_lr.state_dict(), # 获取学习率衰减权重
        }
        # 保存以上数据
        utils.save_checkpoints(ckpts_state, self.output_path, model_name=f'{epoch}_diffusion_model')

    def load_model(self, epoch):
        """模型参数加载"""
        if not self.train:
            model_path = os.path.join(self.output_path, f'{epoch}_diffusion_model_ckpt.pth')
        else:
            model_path = os.path.join(self.output_path, f'latest_diffusion_model_ckpt.pth')
        # 判断模型文件是否存在
        assert os.path.exists(model_path), print(model_path)
        utils.print_banner(f'Load model from {model_path}.')
        # 加载模型文件
        ckpt = torch.load(model_path, map_location=self.device)
        self.agent.actor.load_state_dict(ckpt['actor']) # 将策略权重加载到策略网络中
        if self.train:
            self.agent.actor_optimizer.load_state_dict(ckpt['actor_optimizer']) # 加载优化器权重
            self.agent.policy_step_lr.load_state_dict(ckpt['policy_step_lr']) # 加载学习率衰减器权重
            self.start_epoch = ckpt['epoch'] # 加载模型轮次

    def exp_log(self):
        if self.logger is None:
            self.logger = Logger()

        variant = {
            'Exp Name': "Diffusion BC",
            'Scenario': self.scenario,
            'Train Epochs': self.num_epochs,
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
        # 将变量以表的形式打印
        for k, v in variant.items():
            self.logger.record_tabular(k, v)

        self.logger.dump_tabular()
        # 将模型超参数上传wandb
        param_dict = {"state_dim": self.state_dim, "action_dim": self.action_dim, "max_action": self.max_action,
                      "batch_size": self.batch_size,
                      "learning_rate": self.lr, 'discount': self.discount, 'tau': self.tau, 'eta': self.eta,
                      "T": self.T, 'gn': self.gn}

        if self.scenario == "highway":
            self.job_type = "高速公路场景"
        else:
            self.job_type = " 城区十字路口场景"

        if self.args.upload:
            df = pd.DataFrame.from_dict(param_dict)
            self.init_wandb('V2.0-算法训练验证-交通流算法', "算法超参数表", self.job_type, upload=self.args.upload)
            self.wandb_run.get_run().log({"Table/Algorithm Hyperparameter Table": wandb.Table(dataframe=df)}) # 将模型超参数保存到表中

    def exp(self, save_video=False, show=False, epochs=1000):
        ego_crashed = 0 # 碰撞次数统计
        lane_c = 0 # 变道次数统计
        skip = 0 # 场景跳过次数统计
        offroad = 0 # 跑出道路次数统计
        actions = [] # 动作
        TTC_THW = [] # ttc, thw统计
        distance = 0.0 # 行使距离
        for epoch in range(epochs):
            utils.print_banner('progress:{} %'.format((epoch + 1) / epochs * 100))
            # 创建仿真环境
            env, state, select_name = self.set_env()

            # if self.scenario == 'highway' and env.spd_mean < 5 or state is None:
            if state is None:
                skip += 1
                # print('len(self.road.vehicles)', env.v_sum, 'spd_mean', env.spd_mean)
                continue

            # if self.use_running_state:
            #     state = self.running_state(state)

            for i in range(500):
                # print(i)
                state_var = torch.tensor(state).unsqueeze(0) # 将状态转化为tensor
                action = self.sample_action(state_var) # 根据状态获取模型预测动作
                if show:
                    env.render() # 场景渲染

                pre = env.vehicle.lane_index[2] # 获取当前道路id

                (gail_features, adv_fea), reward, done, info = env.step(action) # 将动作作用于仿真环境
                next_state = gail_features # 获取交通流模型输入
                aft = env.vehicle.lane_index[2] # 获取主车当前所在车道
                if pre != aft: # 判断是否发生变道
                    lane_c += 1

                # 实验数据收集
                actions.append([env.vehicle.action['steering'], env.vehicle.action['acceleration']])
                TTC_THW.append(env.TTC_THW)
                distance += info['distance']

                state = next_state
                # 判断是否碰撞
                if env.vehicle.crashed:
                    ego_crashed += 1
                # 判断是否跑出道路
                if not env.vehicle.on_road:
                    offroad += 1


                log = {
                    'epoch': epoch, 'iter': i,
                    'steering': float(env.vehicle.action['steering']),
                    'acceleration': float(env.vehicle.action['acceleration']),
                    'TTC': float(env.TTC_THW[0][0]), 'THW': float(env.TTC_THW[0][1]),
                    'distance': float(info['distance']), 'ego_crashed':
                        int(env.vehicle.crashed), 'offroad': int(not env.vehicle.on_road)
                }
                # 保存实验数据
                self.save_log(log, epoch + i, f'{self.load_model_id}_exp')

                if done:
                    break

        summary_log = {
            'ego_crashed': ego_crashed,
            'lane_c': lane_c,
            'skip': skip,
            'offroad': offroad,
            'vehicle_distance(KM)': distance / 1000,
        }
        # 保存实验数据总结
        self.save_log(summary_log, 0, f'{self.load_model_id}_summary_exp')
        # 打印实验数据总结
        self.logger.record_tabular('ego_crashed', ego_crashed)
        self.logger.record_tabular('lane_c', lane_c)
        self.logger.record_tabular('skip', skip)
        self.logger.record_tabular('offroad', offroad)
        self.logger.record_tabular('vehicle_distance', distance / 1000.)
        self.logger.dump_tabular()
        if self.args.upload:
            # 加载专家轨迹数据
            expert_traj_path = f'{self.args.project_dir}/{self.expert_traj_path}'
            expert_traj = pickle.load(open(expert_traj_path, "rb"))
            expert_traj = np.array(expert_traj)
            expert_traj = expert_traj[:, -2:]
            nat_steer = expert_traj[:, 0] # 获取人类驾驶方向盘转向角数据
            nat_acc = expert_traj[:, 1] # 获取人类驾驶加速度数据

            df = pandas.read_csv(os.path.join(self.output_path, f'{self.load_model_id}_exp.csv'))
            steer = df['steering'].to_numpy()
            acc = df['acceleration'].to_numpy()
            color = ['sandybrown', 'lightsteelblue']
            # 对比人类数据与模型预测数据分布
            fig, axs = plt.subplots(1, 2)
            # 绘制概率分布统计图
            axs[0].hist(steer, bins=50, color=color[0], label='virtual data', density=True, alpha=0.5)
            axs[0].hist(nat_steer, bins=50, color=color[1], label='natural data', density=True, alpha=0.5)
            # 设置第一个子图x轴标签名称
            axs[0].set_xlabel('steering', fontsize=15)
            axs[1].hist(acc, bins=50, color=color[0], density=True, alpha=0.5)
            axs[1].hist(nat_acc, bins=50, color=color[1], density=True, alpha=0.5)
            # 设置第二个字图x轴标签名称
            axs[1].set_xlabel('acceleration', fontsize=15)
            plt.tight_layout()
            handles, labels = axs[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper right')
            self.init_wandb('V2.0-算法训练验证-交通流算法', "驾驶动作分布图", self.job_type, upload=self.args.upload)
            self.wandb_run.get_run().log({"Result/Driving Action Distribution": wandb.Image(plt)})
            del self.wandb_run

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

            bins = 5
            steer_sim = compute_sim(nat_steer, steer, bins)
            acc_sim = compute_sim(nat_acc, acc)

            sim_data = np.array([steer_sim, acc_sim]).reshape(1, 2)
            df = pd.DataFrame(sim_data, columns=['Steering', 'Acceleration'])
            self.init_wandb('V2.0-算法训练验证-交通流算法', "相似度统计表", self.job_type, upload=self.args.upload)

            self.wandb_run.get_run().log({"Table/Similarity Statistics Table": wandb.Table(dataframe=df)})