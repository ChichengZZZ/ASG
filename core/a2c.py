import torch


def a2c_step(policy_net, value_net, optimizer_policy, optimizer_value, states, actions, returns, advantages, l2_reg):
    """
    A2C（Advantage Actor-Critic）算法的一步更新
    """
    """对值函数（critic）进行更新"""
    values_pred = value_net(states)  # 使用价值网络（value_net）对给定的状态（states）进行预测，得到对当前状态的价值的预测。
    value_loss = (values_pred - returns).pow(2).mean() # 计算价值函数损失，使用预测的价值减去实际的回报值（returns），然后取平方并求均值，得到损失值。
    # weight decay
    for param in value_net.parameters(): # 对价值网络的参数进行迭代
        value_loss += param.pow(2).sum() * l2_reg # 将每个参数的平方和乘以一个正则化系数（l2_reg），然后将其添加到值函数损失中。
    optimizer_value.zero_grad() # 将值函数优化器（optimizer_value）的梯度缓冲区置零。
    value_loss.backward() # 计算价值函数损失的梯度。
    optimizer_value.step() # 使用值函数优化器更新价值函数的参数。

    """对策略（policy）进行更新"""
    log_probs = policy_net.get_log_prob(states, actions) # 使用策略网络（policy_net）计算给定状态和动作的对数概率。
    policy_loss = -(log_probs * advantages).mean() # 计算策略损失，对数概率乘以优势函数（advantages），并取其负值，然后求均值。
    optimizer_policy.zero_grad() # 将策略优化器（optimizer_policy）的梯度缓冲区置零。
    policy_loss.backward() # 计算策略损失的梯度。
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 40) # 对策略网络的梯度进行裁剪，限制在指定范围内。
    optimizer_policy.step() # 使用策略优化器更新策略网络的参数。

    vl = value_loss.detach().cpu().numpy() # 将价值函数损失的数值转换为numpy数据
    pl = policy_loss.detach().cpu().numpy() # 将策略损失的数值转换为numpy数据

    train_log = {
        'value_loss': float(vl),
        'policy_loss': float(pl),
    }
    return train_log