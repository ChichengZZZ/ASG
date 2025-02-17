from __future__ import division, print_function

import importlib
from functools import partial
import numpy as np

from NGSIM_env import utils


def finite_mdp(env, time_quantization=1., horizon=10.):
    """
        状态的时间-碰撞时间（Time-To-Collision，TTC）表示。

        状态的奖励是根据不同TTC和车道上的占用栅格进行定义的。栅格单元格编码了在给定持续时间内，如果自车位于给定车道上，
        每辆被观察到的车辆（包括自车）都将保持恒定速度且不换道（不包括自车）的假设下，自车与另一辆车碰撞的概率。

        例如，在一个三车道道路上，左车道上有一辆车，在5秒内预测到将发生碰撞的情况下，栅格将为：
        [0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0] TTC状态是这个栅格中的一个坐标（车道，时间）。

        如果自车有能力改变速度，一个额外的层被添加到占用栅格中，以遍历可用的不同速度选择。

        最后，为了与FiniteMDPEnv环境兼容，该状态被展平。

        :param AbstractEnv env: 一个环境
        :param time_quantization: 状态表示中使用的时间量化[s]
        :param horizon: 预测碰撞的时间范围[s]

    """
    # 计算TTC栅格
    grid = compute_ttc_grid(env, time_quantization, horizon)

    # 计算当前状态
    grid_state = (env.vehicle.speed_index(), env.vehicle.lane_index[2], 0)
    state = np.ravel_multi_index(grid_state, grid.shape)

    # 计算过渡函数
    transition_model_with_grid = partial(transition_model, grid=grid)
    transition = np.fromfunction(transition_model_with_grid, grid.shape + (env.action_space.n,), dtype=int)
    transition = np.reshape(transition, (np.size(grid), env.action_space.n))

    # 计算奖励函数
    v, l, t = grid.shape
    lanes = np.arange(l)/max(l - 1, 1)
    velocities = np.arange(v)/max(v - 1, 1)
    state_reward = \
        + env.COLLISION_REWARD * grid \
        + env.RIGHT_LANE_REWARD * np.tile(lanes[np.newaxis, :, np.newaxis], (v, 1, t)) \
        + env.HIGH_VELOCITY_REWARD * np.tile(velocities[:, np.newaxis, np.newaxis], (1, l, t))
    state_reward = np.ravel(state_reward)
    action_reward = [env.LANE_CHANGE_REWARD, 0, env.LANE_CHANGE_REWARD, 0, 0]
    reward = np.fromfunction(np.vectorize(lambda s, a: state_reward[s] + action_reward[a]),
                             (np.size(state_reward), np.size(action_reward)),  dtype=int)

    # 计算终止状态
    collision = grid == 1
    end_of_horizon = np.fromfunction(lambda h, i, j: j == grid.shape[2] - 1, grid.shape, dtype=int)
    terminal = np.ravel(collision | end_of_horizon)

    # 创建一个新的有限MDP（马尔可夫决策过程）
    try:
        module = importlib.import_module("finite_mdp.mdp")
        mdp = module.DeterministicMDP(transition, reward, terminal, state=state)
        mdp.original_shape = grid.shape
        return mdp
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("The finite_mdp module is required for conversion. {}".format(e))


def compute_ttc_grid(env, time_quantization, horizon, considered_lanes="all"):
    """
    对于每个自车速度和车道，计算车道内每辆车到碰撞的预测时间，并将结果存储在占用栅格中。
    """
    road_lanes = env.road.network.all_side_lanes(env.vehicle.lane_index)
    grid = np.zeros((env.vehicle.SPEED_COUNT, len(road_lanes), int(horizon / time_quantization)))

    for velocity_index in range(grid.shape[0]):
        ego_velocity = env.vehicle.index_to_speed(velocity_index)

        for other in env.road.vehicles:
            if (other is env.vehicle) or (ego_velocity == other.velocity):
                continue
            margin = other.LENGTH / 2 + env.vehicle.LENGTH / 2
            collision_points = [(0, 1), (-margin, 0.5), (margin, 0.5)]

            for m, cost in collision_points:
                distance = env.vehicle.lane_distance_to(other) + m
                other_projected_velocity = other.velocity * np.dot(other.direction, env.vehicle.direction)
                time_to_collision = distance / utils.not_zero(ego_velocity - other_projected_velocity)
                if time_to_collision < 0:
                    continue
                if env.road.network.is_connected_road(env.vehicle.lane_index, other.lane_index, route=env.vehicle.route, depth=3):
                    # 相同的道路，或具有相同车道数的连接道路
                    if len(env.road.network.all_side_lanes(other.lane_index)) == len(env.road.network.all_side_lanes(env.vehicle.lane_index)):
                        lane = [other.lane_index[2]]
                    # 不同车道数的不同道路：未来车道的不确定性，使用全部车道
                    else:
                        lane = range(grid.shape[1])
                    
                    # 将时间-碰撞（TTC）定量化为上限和下限值。
                    for time in [int(time_to_collision / time_quantization), int(np.ceil(time_to_collision / time_quantization))]:
                        if 0 <= time < grid.shape[2]:
                            # TODO: check lane overflow (e.g. vehicle with higher lane id than current road capacity)
                            grid[velocity_index, lane, time] = np.maximum(grid[velocity_index, lane, time], cost)
                            
    return grid


def transition_model(h, i, j, a, grid):
    """
        根据格子中的位置进行确定性转换到下一位置。

        :param h: 速度的索引
        :param i: 车道的索引
        :param j: 时间的索引
        :param a: 动作的索引
        :param grid: TTC 格子，指定速度、车道、时间和动作的限制
    """
    # 默认转换动作为静止动作（1）
    next_state = clip_position(h, i, j + 1, grid)
    left = a == 0
    right = a == 2
    faster = (a == 3) & (j == 0)
    slower = (a == 4) & (j == 0)
    next_state[left] = clip_position(h[left], i[left] - 1, j[left] + 1, grid)
    next_state[right] = clip_position(h[right], i[right] + 1, j[right] + 1, grid)
    next_state[faster] = clip_position(h[faster] + 1, i[faster], j[faster] + 1, grid)
    next_state[slower] = clip_position(h[slower] - 1, i[slower], j[slower] + 1, grid)
    return next_state


def clip_position(h, i, j, grid):
    """
        Clip a position in the TTC grid, so that it stays within bounds.

    :param h: velocity index
    :param i: lane index
    :param j: time index
    :param grid: the ttc grid
    :return: The raveled index of the clipped position
    """
    h = np.clip(h, 0, grid.shape[0] - 1)
    i = np.clip(i, 0, grid.shape[1] - 1)
    j = np.clip(j, 0, grid.shape[2] - 1)
    indexes = np.ravel_multi_index((h, i, j), grid.shape)
    return indexes
