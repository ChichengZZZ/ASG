from __future__ import division, print_function, absolute_import
import copy
import os
import gym
from gym import spaces
from gym.utils import seeding

from NGSIM_env import utils
from NGSIM_env.envs.common.observation import observation_factory
from NGSIM_env.envs.common.finite_mdp import finite_mdp
from NGSIM_env.envs.common.graphics import EnvViewer
from NGSIM_env.vehicle.behavior import IDMVehicle
from NGSIM_env.vehicle.control import MDPVehicle
from NGSIM_env.vehicle.dynamics import Obstacle


class AbstractEnv(gym.Env):
    """
    A generic environment for various tasks involving a vehicle driving on a road.

    The environment contains a road populated with vehicles, and a controlled ego-vehicle that can change lane and
    velocity. The action space is fixed, but the observation space and reward function must be defined in the
    environment implementations.
    一个通用的环境，用于进行与车辆在道路上行驶相关的各种任务。
    该环境包含了一条有其他车辆的道路，以及一个可以改变车道和速度的控制车辆。动作空间是固定的，但观察空间和奖励函数必须在环境实现中定义。

    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    # 将动作索引映射到动作标签。
    ACTIONS = {0: 'LANE_LEFT',
               1: 'IDLE',
               2: 'LANE_RIGHT',
               3: 'FASTER',
               4: 'SLOWER'}

    # 将动作标签映射到动作索引。
    ACTIONS_INDEXES = {v: k for k, v in ACTIONS.items()}

    # 系统动力学模拟的频率 [Hz]
    SIMULATION_FREQUENCY = 10

    # 观测中存在的任何车辆的最大距离 [米]
    PERCEPTION_DISTANCE = 3.0 * MDPVehicle.SPEED_MAX


    def __init__(self, config=None):
        # Configuration
        self.config = self.default_config()
        if config:
            self.config.update(config)

        # Seeding
        self.np_random = None
        self.seed()

        # Scene
        self.road = None
        self.vehicle = None

        # Spaces
        self.observation = None
        self.action_space = None
        self.observation_space = None
        self.define_spaces()

        # Running
        self.time = 0  # Simulation time
        self.steps = 0  # Actions performed
        self.done = False

        # Rendering
        self.viewer = None
        self.automatic_rendering_callback = None
        self.should_update_rendering = True
        self.rendering_mode = 'human'
        self.offscreen = self.config.get("offscreen_rendering", False)
        self.enable_auto_render = False

        self.reset()

    @classmethod
    def default_config(cls):
        """
        默认的环境配置。

        可以在环境实现中进行重载，或通过调用configure()进行重载。
        :return: 一个配置字典
        """
        return {
            "observation": {"type": "Kinematics"},
            "policy_frequency": 1,  # [Hz]
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
            "screen_width": 640,  # [px]
            "screen_height": 320,  # [px]
            "centering_position": [0.5, 0.5],
            "show_trajectories": False,
            "simulation_frequency": 15,  # [Hz]
            "scaling": 5.5,
            "render_agent": True,
            "offscreen_rendering": os.environ.get("OFFSCREEN_RENDERING", "0") == "1",
            "manual_control": False,
            "real_time_rendering": False
        }

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def configure(self, config):
        if config:
            self.config.update(config)

    def define_spaces(self):
        self.action_space = spaces.Discrete(len(self.ACTIONS))

        if "observation" not in self.config:
            raise ValueError("The observation configuration must be defined")
        self.observation = observation_factory(self, self.config["observation"])
        self.observation_space = self.observation.space()

    def _reward(self, action):
        """
        返回与执行给定动作并到达当前状态相关联的奖励。

        :param action: 最后执行的动作
        :return: 奖励
        """
        raise NotImplementedError

    def _is_terminal(self):
        """
        检查当前状态是否为终止状态
        :return:是否是终止状态
        """
        raise NotImplementedError

    def _cost(self, action):
        """
        用于有受限的MDP的约束度量。

        如果定义了一个约束条件，必须使用一个不包含约束条件作为惩罚的替代奖励。
        :param action: 最后执行的动作
        :return: 约束信号，替代（无约束）奖励
        """
        raise NotImplementedError

    def reset(self):
        """
        将环境重置为初始配置
        :return: 重置状态的观测值
        """
        self.time = 0
        self.done = False
        self.define_spaces()

        return self.observation.observe()

    def step(self, action):
        """
        执行一个动作并推进环境动态。

        这个动作由自动驾驶车辆执行，道路上的所有其他车辆在下一个决策步骤之前执行它们的默认行为，持续几个模拟时间步长。
        :param int action: 自动驾驶车辆执行的动作
        :return: 一个元组 (观测值, 奖励, 终止状态, 信息)

        """
        if self.road is None or self.vehicle is None:
            raise NotImplementedError("The road and vehicle must be initialized in the environment implementation")

        self._simulate(action)

        obs = self.observation.observe()
        reward = self._reward(action)
        terminal = self._is_terminal()

        info = {
            "velocity": self.vehicle.velocity,
            "crashed": self.vehicle.crashed,
            "action": action,
        }

        try:
            info["cost"] = self._cost(action)
        except NotImplementedError:
            pass

        return obs, reward, terminal, info

    def _simulate(self, action=None):
        """
        执行几个恒定动作的模拟步骤
        """
        for _ in range(int(self.SIMULATION_FREQUENCY // self.config["policy_frequency"])):
            if action is not None and self.time % int(self.SIMULATION_FREQUENCY // self.config["policy_frequency"]) == 0:
                # Forward action to the vehicle
                self.vehicle.act(self.ACTIONS[action])

            self.road.act()
            self.road.step(1/self.SIMULATION_FREQUENCY)
            self.time += 1

            # 如果已经启动了一个查看器，则自动渲染中间的模拟步骤
            # 如果渲染是在屏幕外进行的，则忽略
            self._automatic_rendering()

            # Stop at terminal states
            if self.done or self._is_terminal():
                break

        self.enable_auto_render = False

    def render(self, mode='human'):
        """
        渲染环境。

        如果没有查看器存在，则创建一个，并使用它来渲染图像。
        :param mode: 渲染模式

        """
        self.rendering_mode = mode

        if self.viewer is None:
            self.viewer = EnvViewer(self, offscreen=self.offscreen)

        self.enable_auto_render = not self.offscreen

        # 如果帧已经被渲染过了，则不做任何操作
        if self.should_update_rendering:
            self.viewer.display()

        if mode == 'rgb_array':
            image = self.viewer.get_image()
            #if not self.viewer.offscreen:
            #    self.viewer.handle_events()
            #self.viewer.handle_events()
            return image
        #elif mode == 'human':
        #    if not self.viewer.offscreen:
            #   self.viewer.handle_events()
                

        self.should_update_rendering = False

    def update_metadata(self, video_real_time_ratio=2):
        frames_freq = self.config["simulation_frequency"] \
            if self._monitor else self.config["policy_frequency"]
        self.metadata['video.frames_per_second'] = video_real_time_ratio * frames_freq

    def set_monitor(self, monitor: gym.wrappers.Monitor):
        self._monitor = monitor
        self.update_metadata()

    def close(self):
        """
        关闭环境。

        如果存在环境查看器，则关闭它。
        """
        self.done = True
        if self.viewer is not None:
            self.viewer.close()

        self.viewer = None

    def get_available_actions(self):
        """
        获取当前可用的动作列表。
        在道路边界上不可用车道变换，而在最大或最小速度下不可用速度变换。
        :return: 可用动作的列表
        """
        actions = [self.ACTIONS_INDEXES['IDLE']]
        
        for l_index in self.road.network.side_lanes(self.vehicle.lane_index):
            if l_index[2] < self.vehicle.lane_index[2] and self.road.network.get_lane(l_index).is_reachable_from(self.vehicle.position):
                actions.append(self.ACTIONS_INDEXES['LANE_LEFT'])
            if l_index[2] > self.vehicle.lane_index[2] and self.road.network.get_lane(l_index).is_reachable_from(self.vehicle.position):
                actions.append(self.ACTIONS_INDEXES['LANE_RIGHT'])

        if self.vehicle.velocity_index < self.vehicle.SPEED_COUNT - 1:
            actions.append(self.ACTIONS_INDEXES['FASTER'])
        if self.vehicle.velocity_index > 0:
            actions.append(self.ACTIONS_INDEXES['SLOWER'])

        return actions

    def _automatic_rendering(self):
        """
        在动作仍在进行时自动渲染中间帧。 这样可以渲染整个视频，而不仅仅是与智能体决策对应的单个步骤。

        如果已经设置了回调函数，则使用它执行渲染。这对于需要访问这些中间渲染的环境包装器（例如视频记录监视器）非常有用。

        """
        if self.viewer is not None and self.enable_auto_render:
            self.should_update_rendering = True

            if self.automatic_rendering_callback:
                self.automatic_rendering_callback()
            else:
                self.render(self.rendering_mode)

    def simplify(self):
        """
            返回一个简化的环境副本，其中远处的车辆已经从道路上消失。

            这旨在降低策略计算负载，同时保留最优动作集。

            :return: 一个简化的环境状态

        """
        state_copy = copy.deepcopy(self)
        state_copy.road.vehicles = [state_copy.vehicle] + state_copy.road.close_vehicles_to(state_copy.vehicle, self.PERCEPTION_DISTANCE)

        return state_copy

    def change_vehicles(self, vehicle_class_path):
        """
            改变道路上所有车辆的类型
            :param vehicle_class_path: 其他车辆行为类的路径 示例: "highway_env.vehicle.behavior.IDMVehicle"
            :return: 具有修改后的其他车辆行为模型的新环境
        """
        vehicle_class = utils.class_from_path(vehicle_class_path)

        env_copy = copy.deepcopy(self)
        vehicles = env_copy.road.vehicles
        for i, v in enumerate(vehicles):
            if v is not env_copy.vehicle and not isinstance(v, Obstacle):
                vehicles[i] = vehicle_class.create_from(v)
                
        return env_copy

    def set_preferred_lane(self, preferred_lane=None):
        env_copy = copy.deepcopy(self)
        if preferred_lane:
            for v in env_copy.road.vehicles:
                if isinstance(v, IDMVehicle):
                    raise NotImplementedError
                else:
                    raise NotImplementedError
                    # Vehicle with lane preference are also less cautious
                    v.LANE_CHANGE_MAX_BRAKING_IMPOSED = 1000

        return env_copy

    def set_route_at_intersection(self, _to):
        env_copy = copy.deepcopy(self)
        for v in env_copy.road.vehicles:
            if isinstance(v, IDMVehicle):
                v.set_route_at_intersection(_to)

        return env_copy

    def randomize_behaviour(self):
        env_copy = copy.deepcopy(self)
        for v in env_copy.road.vehicles:
            if isinstance(v, IDMVehicle):
                v.randomize_behavior()

        return env_copy

    def to_finite_mdp(self):
        return finite_mdp(self, time_quantization=1/self.config["policy_frequency"])

    def __deepcopy__(self, memo):
        """
        执行深拷贝，但不复制环境查看器。
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k not in ['viewer', 'automatic_rendering_callback']:
                setattr(result, k, copy.deepcopy(v, memo))
            else:
                setattr(result, k, None)

        return result
