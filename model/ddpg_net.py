import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import random

def fanin_init(size, fanin=None):
    # 初始化权重
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class Actor(nn.Module):
    def __init__(self, nb_states, nb_actions, max_action, hidden1=256, hidden2=256, init_w=3e-3):
        # 初始化 Actor 类
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, nb_actions)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.init_weights(init_w)
        self.max_action = max_action
        self.action_dim = nb_actions
        self.state_dim = nb_states

    def init_weights(self, init_w):
        # 初始化权重，使用 fanin_init 函数
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, x):
        # 前向传播过程，使用 ReLU 和 Tanh 激活函数
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.max_action * self.tanh(out)
        return out

    def select_action(self, state, epision, noise):
        # 根据给定的状态选择动作
        action = self.forward(state)
        if random.random() < epision:
            action += noise
        return action, torch.zeros((1, self.action_dim)), torch.zeros((1, self.action_dim)), torch.zeros((1, self.action_dim))

class Critic(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=256, hidden2=256, init_w=3e-3):
        # 初始化 Critic 类
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1 + nb_actions, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
        self.init_weights(init_w)

    def init_weights(self, init_w):
        # 初始化权重，使用 fanin_init 函数
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, x, a):
        # 前向传播过程，连接状态和动作
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(torch.cat([out, a], 1))
        out = self.relu(out)
        out = self.fc3(out)
        return out

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_layer=(256, 256)):
        # 初始化 Actor 类
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, hidden_layer[0])
        self.l2 = nn.Linear(hidden_layer[0], hidden_layer[1])
        self.l3 = nn.Linear(hidden_layer[1], action_dim)
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.max_action = max_action

    def forward(self, x):
        # 前向传播过程，使用 ReLU 和 Tanh 激活函数
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x

    def select_action(self, state):
        # 根据给定的状态选择动作
        return self.forward(state), torch.zeros((1, self.action_dim)), torch.zeros((1, self.action_dim)), torch.zeros((1, self.action_dim))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layer=(256, 256)):
        # 初始化 Critic 类
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, hidden_layer[0])
        self.l2 = nn.Linear(hidden_layer[0], hidden_layer[1])
        self.l3 = nn.Linear(hidden_layer[1], 1)

    def forward(self, x, u):
        # 前向传播过程，连接状态和动作
        x = F.relu(self.l1(torch.cat([x, u], 1)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x