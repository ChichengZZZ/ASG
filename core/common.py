import torch
from Utils import to_device


def estimate_advantages(rewards, masks, values, gamma, tau, device):
    """这段代码是用来估计强化学习中的优势函数（advantage function）和回报（returns）的函数。"""
    rewards, masks, values = to_device(torch.device('cpu'), rewards, masks, values) # 使用to_device函数将rewards、masks和values转移到了cpu设备上。
    tensor_type = type(rewards) # 获取了rewards张量的类型，并将其赋值给tensor_type。
    deltas = tensor_type(rewards.size(0), 1) # 创建了一个与rewards张量相同大小的deltas张量，用于存储每个时间步的增益差异（delta）。
    advantages = tensor_type(rewards.size(0), 1) # 创建了一个与rewards张量相同大小的advantages张量，用于存储每个时间步的优势函数值。

    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))): # 逆序的循环，循环变量i从rewards.size(0) - 1到0，递减1。
        deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values[i] # 计算deltas[i]，表示当前时间步的增益差异。
        advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i] # 计算advantages[i]，表示当前时间步的优势函数值。

        prev_value = values[i, 0]
        prev_advantage = advantages[i, 0]

    returns = values + advantages # 计算returns，将values和advantages相加。
    advantages = (advantages - advantages.mean()) / advantages.std() # 对advantages进行标准化处理，即减去均值并除以标准差。

    advantages, returns = to_device(device, advantages, returns) # 使用to_device函数将advantages和returns转移到指定的device设备上。
    return advantages, returns
