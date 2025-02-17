import numpy as np
import scipy.optimize
from Utils import *


def conjugate_gradients(Avp_f, b, nsteps, rdotr_tol=1e-10):
    ''' 函数功能：使用共轭梯度法求解线性方程组Ax=b的解x
        参数说明：
        - Avp_f：一个函数，用于计算Avp，其中A是系数矩阵，v是向量p的乘积，该函数返回一个向量
        - b: 方程组Ax=b中的向量b
        - nsteps: 最大迭代步数
        - rdotr_tol: 终止条件，当残差向量的内积小于该值时停止迭代
        返回值说明：
        - x: 方程组Ax=b的解向量
    '''
    x = zeros(b.size(), device=b.device)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for i in range(nsteps):
        Avp = Avp_f(p)
        # 计算步长alpha
        alpha = rdotr / torch.dot(p, Avp)
        # 更新解向量x
        x += alpha * p
        # 更新残差向量r
        r -= alpha * Avp
        # 计算新的残差向量的内积
        new_rdotr = torch.dot(r, r)
        # 计算修正因子betta
        betta = new_rdotr / rdotr
        # 更新搜索方向向量p
        p = r + betta * p
        # 更新残差向量的内积
        rdotr = new_rdotr
        # 判断是否满足终止条件，若满足则停止迭代
        if rdotr < rdotr_tol:
            break
    return x


def line_search(model, f, x, fullstep, expected_improve_full, max_backtracks=10, accept_ratio=0.1):
    '''
        函数功能：线性搜索算法，用于确定步长，以在给定搜索方向上找到满足一定条件的目标函数值的减少量
        参数说明：
        - model：模型
        - f：目标函数，输入一个布尔值，返回目标函数的值
        - x：当前参数向量 - fullstep：完整步长，指定搜索方向的步长向量
        - expected_improve_full：预期变化量的比例，用于调整步长
        - max_backtracks：最大回溯次数，用于控制线性搜索的迭代次数
        - accept_ratio：接受阈值，当目标函数的实际改善比例大于此阈值时，线性搜索停止
        返回值说明：
        - 如果找到满足接受阈值的步长，则返回 True 和更新后的参数向量 x_new
        - 如果所有尝试的步长都不满足接受阈值，则返回 False 和原始的参数向量 x
    '''
    fval = f(True).item() # 计算当前目标函数值

    for stepfrac in [.5**x for x in range(max_backtracks)]:
        x_new = x + stepfrac * fullstep # 根据步长因子计算新的参数向量
        set_flat_params_to(model, x_new) # 更新模型的参数
        fval_new = f(True).item() # 计算更新后的目标函数值
        actual_improve = fval - fval_new # 计算实际改善量
        expected_improve = expected_improve_full * stepfrac # 计算预期改善量
        ratio = actual_improve / expected_improve  # 计算实际改善量与预期改善量的比例

        if ratio > accept_ratio: # 如果比例大于接受阈值，则停止线性搜索
            return True, x_new
    # 如果所有尝试的步长都不满足接受阈值，则返回 False 和原始的参数向量
    return False, x


def trpo_step(policy_net, value_net, states, actions, returns, advantages, max_kl, damping, l2_reg, use_fim=True):
    '''
        函数功能：使用 TRPO 算法进行优化步骤
        参数说明：
        - policy_net：策略网络
        - value_net：值函数网络
        - states：状态数据
        - actions：动作数据
        - returns：回报数据
        - advantages：优势函数数据
        - max_kl：最大 KL 散度
        - damping：阻尼系数
        - l2_reg：L2 正则化系数
        - use_fim：是否使用 FIM 进行 Hessian-向量乘法计算
        返回值说明：
        - 如果线性搜索成功找到满足条件的步长，则返回 True - 否则返回 False
    '''
    """更新值函数"""

    def get_value_loss(flat_params):
        set_flat_params_to(value_net, tensor(flat_params))
        for param in value_net.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)
        values_pred = value_net(states)
        value_loss = (values_pred - returns).pow(2).mean()

        # 权重衰减
        for param in value_net.parameters():
            value_loss += param.pow(2).sum() * l2_reg
        value_loss.backward()
        return value_loss.item(), get_flat_grad_from(value_net.parameters()).cpu().numpy()

    # 使用 L-BFGS 算法进行值函数的优化
    flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_value_loss,
                                                            get_flat_params_from(value_net).detach().cpu().numpy(),
                                                            maxiter=25)
    set_flat_params_to(value_net, tensor(flat_params))

    """更新策略"""
    with torch.no_grad():
        fixed_log_probs = policy_net.get_log_prob(states, actions)
    """定义 TRPO 的损失函数"""
    def get_loss(volatile=False):
        with torch.set_grad_enabled(not volatile):
            log_probs = policy_net.get_log_prob(states, actions)
            action_loss = -advantages * torch.exp(log_probs - fixed_log_probs)
            return action_loss.mean()

    """使用 Fisher 信息矩阵计算 Hessian-向量乘法"""
    def Fvp_fim(v):
        M, mu, info = policy_net.get_fim(states)
        mu = mu.view(-1)
        filter_input_ids = set() if policy_net.is_disc_action else set([info['std_id']])

        t = ones(mu.size(), requires_grad=True, device=mu.device)
        mu_t = (mu * t).sum()
        Jt = compute_flat_grad(mu_t, policy_net.parameters(), filter_input_ids=filter_input_ids, create_graph=True)
        Jtv = (Jt * v).sum()
        Jv = torch.autograd.grad(Jtv, t)[0]
        MJv = M * Jv.detach()
        mu_MJv = (MJv * mu).sum()
        JTMJv = compute_flat_grad(mu_MJv, policy_net.parameters(), filter_input_ids=filter_input_ids).detach()
        JTMJv /= states.shape[0]
        if not policy_net.is_disc_action:
            std_index = info['std_index']
            JTMJv[std_index: std_index + M.shape[0]] += 2 * v[std_index: std_index + M.shape[0]]
        return JTMJv + v * damping

    """直接从 KL 散度计算 Hessian-向量乘法"""
    def Fvp_direct(v):
        kl = policy_net.get_kl(states)
        kl = kl.mean()

        grads = torch.autograd.grad(kl, policy_net.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

        kl_v = (flat_grad_kl * v).sum()
        grads = torch.autograd.grad(kl_v, policy_net.parameters())
        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).detach()

        return flat_grad_grad_kl + v * damping

    Fvp = Fvp_fim if use_fim else Fvp_direct

    loss = get_loss()
    grads = torch.autograd.grad(loss, policy_net.parameters())
    loss_grad = torch.cat([grad.view(-1) for grad in grads]).detach()
    stepdir = conjugate_gradients(Fvp, -loss_grad, 10)

    shs = 0.5 * (stepdir.dot(Fvp(stepdir)))
    lm = math.sqrt(max_kl / shs)
    fullstep = stepdir * lm
    expected_improve = -loss_grad.dot(fullstep)

    prev_params = get_flat_params_from(policy_net)
    success, new_params = line_search(policy_net, get_loss, prev_params, fullstep, expected_improve)
    set_flat_params_to(policy_net, new_params)

    return success
