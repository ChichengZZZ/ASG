import abc
import os
import sys

import pandas
import torch
import numpy as np
from abc import ABC
from Utils.logger import logger
from Utils.wandb_run_provider import create_wandb_runner


class Agent(ABC):
    def __init__(self):
        self.policy_net = None # 策略网络
        self.logger = logger # 日志记录器
        self.type = torch.float64 # 数据类型
        self.wandb_run = None # 用于wandb运行时

    @abc.abstractmethod
    def set_env(self):
        """设置环境"""
        raise NotImplementedError

    @abc.abstractmethod
    def learn(self):
        """实现算法训练整个过程"""
        raise NotImplementedError

    @abc.abstractmethod
    def sample_action(self, state):
        """根据状态获取模型输出的动作"""
        raise NotImplementedError

    @abc.abstractmethod
    def load_model(self, epoch):
        """实现模型加载过程"""
        raise NotImplementedError

    @abc.abstractmethod
    def save_model(self, epoch):
        """实现模型保存过程"""
        raise NotImplementedError

    def init_wandb(self, project, name_zh, job_type, upload=True):
        """创建wandb对象"""
        self.wandb_run = create_wandb_runner(project, name_zh, job_type, upload)

    def upload_data_to_wandb_server(self, name: str, df: pandas.DataFrame):
        """将DataFrame格式数据上传wandb服务中"""
        col_names = df.columns
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                self.wandb_run.get_run().log({f"{name}/"+col_names[j]: float(df.loc[i, col_names[j]])})



