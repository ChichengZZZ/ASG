import torch


def ppo_step(policy_net, value_net, optimizer_policy, optimizer_value, optim_value_iternum, states, actions,
             returns, advantages, fixed_log_probs, clip_epsilon, l2_reg):
    """我们定义一个ppo_step函数，这个函数用于执行PPO算法的一个步骤。函数的入参包括
    policy_net（策略网络）、value_net（值函数网络）、optimizer_policy（策略网络的优化器）、optimizer_value（值函数网络的优化器）、
    optim_value_iternum（值函数网络的迭代次数）、states（状态）、actions（动作）、returns（回报）、advantages（优势函数）、
    fixed_log_probs（固定的对数概率）、clip_epsilon（裁剪范围）、l2_reg（L2正则化参数）。"""
    """update critic"""
    for _ in range(optim_value_iternum):
        values_pred = value_net(states)  # 预测的值函数值
        value_loss = (values_pred - returns).pow(2).mean()  # 计算值函数损失值
        # 权重衰减
        for param in value_net.parameters():
            value_loss += param.pow(2).sum() * l2_reg
        optimizer_value.zero_grad()  # 清空值函数网络的梯度
        value_loss.backward()  # 反向传播计算梯度
        optimizer_value.step()  # 更新值函数网络的参数

    """update policy"""
    log_probs = policy_net.get_log_prob(states, actions)  # 计算当前状态下选择动作的对数概率
    ratio = torch.exp(log_probs - fixed_log_probs)  # 计算比值ratio
    surr1 = ratio * advantages  # 第一个surrogate loss
    
    ###################沒有leaky
#    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages  # 第二个surrogate loss

    ######################  leaky ppo 的修改
    alpha = 0.01  # 0<= alpha <1
    l_sa = alpha * ratio + (1 - alpha) * (1.0 - clip_epsilon)
    u_sa = alpha * ratio + (1 - alpha) * (1.0 + clip_epsilon)
    surr2 = torch.clamp(ratio, l_sa, u_sa) * advantages

    policy_surr = -torch.min(surr1, surr2).mean()  # 最小化两个surrogate loss的平均值作为策略损失
    optimizer_policy.zero_grad()  # 清空策略网络的梯度
    policy_surr.backward()  # 反向传播计算梯度
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 40)  # 对策略网络的梯度进行裁剪
    optimizer_policy.step()  # 更新策略网络的参数