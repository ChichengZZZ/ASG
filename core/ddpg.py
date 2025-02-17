import torch


def ddpg_step(policy_net, policy_net_target, value_net, value_net_target, optimizer_policy, optimizer_value,
              states, next_states, actions, rewards, masks, discount, l2_reg, tau):
    """这段代码是用于执行DDPG算法中的一个训练步骤，包括更新值网络（value network）和策略网络（policy network）。"""

    """更新价值网络"""
    # 计算目标Q值（target Q）
    target_Q = value_net_target(next_states, policy_net_target(next_states)).squeeze() # 生成下一个状态的动作策略
    target_Q = rewards + (masks * discount * target_Q).detach() # 将其乘以掩码，折扣率，以及加上环境奖励，并使用detach()方法截断梯度流,以获得target_Q。

    # 计算当前的Q值（current Q）
    current_Q = value_net(states, actions).squeeze()
    # 计算价值网络的损失函数。通过计算(current_Q - target_Q)的平方，再取平均值得到损失函数的值。同时还。整体上，该损失函数的意义在于将当前预测的Q值与目标Q值尽可能地靠近。
    value_loss = (current_Q - target_Q).pow(2).mean()
    # 权重衰减, 对价值网络每个参数进行平方和乘以一个系数的累加添加到损失中
    for param in value_net.parameters():
        value_loss += param.pow(2).sum() * l2_reg
    optimizer_value.zero_grad() # 梯度缓存清零
    value_loss.backward() # 反向传播计算梯度
    optimizer_value.step() # 根据梯度进行参数更新

    """策略更新"""
    actor_loss = -value_net(states, policy_net(states)).mean() # 对负的价值进行梯度下降
    optimizer_policy.zero_grad()
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 40) # 对梯度进行裁减
    optimizer_policy.step()

    # 将策略网络权重与目标策略网络权重加权求和作为目标策略网络权重
    for param, target_param in zip(policy_net.parameters(), policy_net_target.parameters()):
        target_param.data.copy_(((1 - tau) * target_param.data) + tau * param.data)
    # 将价值网络权重与目标价值网络权重加权求和作为目标价值网络权重
    for param, target_param in zip(value_net.parameters(), value_net_target.parameters()):
        target_param.data.copy_(((1 - tau) * target_param.data) + tau * param.data)

    vl = value_loss.detach().cpu().numpy() # 获取价值损失
    pl = actor_loss.detach().cpu().numpy() # 获取策略损失

    train_log = {
        'value_loss': float(vl),
        'policy_loss': float(pl),
    }
    return train_log
