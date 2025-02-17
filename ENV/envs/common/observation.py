from __future__ import division, print_function, absolute_import
import pandas
from gym import spaces
import numpy as np

from NGSIM_env import utils
from NGSIM_env.envs.common.finite_mdp import compute_ttc_grid
from NGSIM_env.road.lane import AbstractLane
from NGSIM_env.vehicle.control import MDPVehicle
from NGSIM_env.road.graphics import WorldSurface


class ObservationType(object):
    def space(self):
        raise NotImplementedError()

    def observe(self):
        raise NotImplementedError()


class GrayscaleObservation(ObservationType):
    """
    一个观察类，直接收集仿真器渲染的内容作为输入，并像Nature DQN中一样堆叠收集到的帧。在传递的配置字典中，预计会有特定的键。

    环境配置中观察字典的示例：
    "observation":
    {
        "type": "GrayscaleObservation",
        "weights": [0.2989, 0.5870, 0.1140], #用于RGB转换的权重，
        "stack_size": 4,
        "observation_shape": (84, 84)
    }

    此外，环境的screen_height和screen_width应与预期的observation_shape匹配。

    """
    def __init__(self, env, **config):
        self.env = env
        self.config = config
        self.observation_shape = config["observation_shape"]
        self.shape = self.observation_shape + (config["stack_size"], )
        self.state = np.zeros(self.shape)

    def space(self):
        try:
            return spaces.Box(shape=self.shape, low=0, high=1, dtype=np.float32)
        except AttributeError:
            return None

    def observe(self):
        new_obs = self._record_to_grayscale()
        new_obs = np.reshape(new_obs, self.observation_shape)
        self.state = np.roll(self.state, -1, axis=-1)
        self.state[:, :, -1] = new_obs
        return self.state

    def _record_to_grayscale(self):
        raw_rgb = self.env.render('rgb_array')
        return np.dot(raw_rgb[..., :3], self.config['weights'])


class TimeToCollisionObservation(ObservationType):
    def __init__(self, env, horizon=10, **kwargs):
        self.env = env
        self.horizon = horizon

    def space(self):
        try:
            return spaces.Box(shape=self.observe().shape, low=0, high=1, dtype=np.float32)
        except AttributeError:
            return None

    def observe(self):
        grid = compute_ttc_grid(self.env, time_quantization=1/self.env.config["policy_frequency"], horizon=self.horizon)
        padding = np.ones(np.shape(grid))
        padded_grid = np.concatenate([padding, grid, padding], axis=1)
        obs_lanes = 3
        l0 = grid.shape[1] + self.env.vehicle.lane_index[2] - obs_lanes // 2
        lf = grid.shape[1] + self.env.vehicle.lane_index[2] + obs_lanes // 2
        clamped_grid = padded_grid[:, l0:lf+1, :]
        repeats = np.ones(clamped_grid.shape[0])
        repeats[np.array([0, -1])] += clamped_grid.shape[0]
        padded_grid = np.repeat(clamped_grid, repeats.astype(int), axis=0)
        obs_velocities = 3
        v0 = grid.shape[0] + self.env.vehicle.velocity_index - obs_velocities // 2
        vf = grid.shape[0] + self.env.vehicle.velocity_index + obs_velocities // 2
        clamped_grid = padded_grid[v0:vf + 1, :, :]

        return clamped_grid


class KinematicObservation(ObservationType):
    """
    观察附近车辆的运动学。
    """
    FEATURES = ['presence', 'x', 'y', 'vx', 'vy']# 'ax', 'ay', 'lane']

    def __init__(self, env,
                 features=FEATURES,
                 vehicles_count=6,
                 features_range=None,
                 absolute=False,
                 normalize=True,
                 clip=True,
                 order="sorted",
                 observe_intentions=False,
                 **kwargs):
        """
            :param env: 要观察的环境
            :param features: 观察中使用的特征的名称
            :param vehicles_count: 观察的车辆数量
            :param absolute: 使用绝对坐标
            :param normalize: 观察是否应被归一化
            :param clip: 值是否应被裁剪到所需范围内
            :param order: 观察车辆的顺序。可选值：按顺序排列，随机重排
            :param observe_intentions: 观察其他车辆的目的地
        """
        self.env = env
        self.features = features
        self.vehicles_count = vehicles_count
        self.features_range = features_range
        self.absolute = absolute
        self.order = order
        self.normalize = normalize
        self.clip = clip
        self.observe_intentions = observe_intentions

    def space(self):
        return spaces.Box(shape=(self.vehicles_count*len(self.features),), low=-1, high=1, dtype=np.float32)

    def normalize_obs(self, df):
        """
        对观察值进行归一化。

        目前，假设道路沿x轴是直线。
        :param Dataframe df: 观察数据

        """
        if not self.features_range:
            side_lanes = self.env.road.network.all_side_lanes(self.env.vehicle.lane_index)
            self.features_range = {
                "x": [-5.0 * MDPVehicle.SPEED_MAX, 5.0*MDPVehicle.SPEED_MAX],
                "y": [-(12 / 3.281)*len(side_lanes), (12 / 3.281)*len(side_lanes)],
                "vx": [-2*MDPVehicle.SPEED_MAX, 2*MDPVehicle.SPEED_MAX],
                "vy": [-2*MDPVehicle.SPEED_MAX, 2*MDPVehicle.SPEED_MAX]
            }

        for feature, f_range in self.features_range.items():
            if feature in df:
                df[feature] = utils.remap(df[feature], [f_range[0], f_range[1]], [-1, 1])
                if self.clip:
                    df[feature] = np.clip(df[feature], -1, 1)

        return df

    def observe(self, vehicle=None):
        # Add ego-vehicle
        if vehicle is None:
            vehicle = self.env.vehicle
        if vehicle is None:
            return

        df = pandas.DataFrame.from_records([vehicle.to_dict()])[self.features]
        # Add nearby traffic
        sort, see_behind = (True, False) if self.order == "sorted" else (False, True)
        close_vehicles = self.env.road.close_vehicles_to(vehicle,
                                                         self.env.PERCEPTION_DISTANCE,
                                                         count=self.vehicles_count-1,
                                                         sort=sort,
                                                         see_behind=see_behind)
        if close_vehicles:
            origin = vehicle if not self.absolute else None
            df = pandas.concat([df, pandas.DataFrame.from_records(
                [v.to_dict(origin, observe_intentions=self.observe_intentions)
                 for v in close_vehicles[-self.vehicles_count + 1:]])[self.features]])
        
        # Normalize
        if self.normalize:
            df = self.normalize_obs(df)

        # Fill missing rows
        if df.shape[0] < self.vehicles_count:
            rows = np.zeros((self.vehicles_count - df.shape[0], len(self.features)))
            df = pandas.concat([df, pandas.DataFrame(data=rows, columns=self.features)])
        
        # Reorder
        df = df[self.features]
        obs = df.values.copy()
        
        if self.order == "shuffled":
            self.env.np_random.shuffle(obs[1:])
        
        # Flatten
        return obs


class OccupancyGridObservation(ObservationType):
    """
    观察附近车辆的占用网格。
    """
    FEATURES = ['presence', 'vx', 'vy']
    GRID_SIZE = [[-5.5*5, 5.5*5], [-5.5*5, 5.5*5]]
    GRID_STEP = [5, 5]

    def __init__(self,
                 env,
                 features=FEATURES,
                 grid_size=GRID_SIZE,
                 grid_step=GRID_STEP,
                 features_range=None,
                 absolute=False,
                 **kwargs):
        """
            :param env: 要观察的环境
            :param features: 观察中使用的特征的名称
            :param vehicles_count: 观察的车辆数量
        """
        self.env = env
        self.features = features
        self.grid_size = np.array(grid_size)
        self.grid_step = np.array(grid_step)
        grid_shape = np.asarray(np.floor((self.grid_size[:, 1] - self.grid_size[:, 0]) / grid_step), dtype=int)
        self.grid = np.zeros((len(self.features), *grid_shape))
        self.features_range = features_range
        self.absolute = absolute

    def space(self):
        return spaces.Box(shape=self.grid.shape, low=-1, high=1, dtype=np.float32)

    def normalize(self, df):
        """
        归一化观察值。

        目前，假设道路沿着x轴是直线。
        :param Dataframe df: 观察数据

        """
        if not self.features_range:
            self.features_range = {
                "vx": [-2*MDPVehicle.SPEED_MAX, 2*MDPVehicle.SPEED_MAX],
                "vy": [-2*MDPVehicle.SPEED_MAX, 2*MDPVehicle.SPEED_MAX]
            }
        for feature, f_range in self.features_range.items():
            if feature in df:
                df[feature] = utils.remap(df[feature], [f_range[0], f_range[1]], [-1, 1])
        return df

    def observe(self):
        if self.absolute:
            raise NotImplementedError()
        else:
            # Add nearby traffic
            self.grid.fill(0)
            df = pandas.DataFrame.from_records([v.to_dict(self.env.vehicle) for v in self.env.road.vehicles])
            # Normalize
            df = self.normalize(df)
            # Fill-in features
            for layer, feature in enumerate(self.features):
                for _, vehicle in df.iterrows():
                    x, y = vehicle["x"], vehicle["y"]
                    # Recover unnormalized coordinates for cell index
                    if "x" in self.features_range:
                        x = utils.remap(x, [-1, 1], [self.features_range["x"][0], self.features_range["x"][1]])
                    if "y" in self.features_range:
                        y = utils.remap(y, [-1, 1], [self.features_range["y"][0], self.features_range["y"][1]])
                    cell = (int((x - self.grid_size[0, 0]) / self.grid_step[0]),
                            int((y - self.grid_size[1, 0]) / self.grid_step[1]))
                    if 0 <= cell[1] < self.grid.shape[-2] and 0 <= cell[0] < self.grid.shape[-1]:
                        self.grid[layer, cell[1], cell[0]] = vehicle[feature]
            # Clip
            obs = np.clip(self.grid, -1, 1)
            
            return obs


class KinematicsGoalObservation(KinematicObservation):
    def __init__(self, env, scales, **kwargs):
        self.scales = np.array(scales)
        super(KinematicsGoalObservation, self).__init__(env, **kwargs)

    def space(self):
        try:
            obs = self.observe()
            return spaces.Dict(dict(
                desired_goal=spaces.Box(-np.inf, np.inf, shape=obs["desired_goal"].shape, dtype=np.float32),
                achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype=np.float32),
                observation=spaces.Box(-np.inf, np.inf, shape=obs["observation"].shape, dtype=np.float32),
            ))
        except AttributeError:
            return None

    def observe(self):
        obs = np.ravel(pandas.DataFrame.from_records([self.env.vehicle.to_dict()])[self.features])
        goal = np.ravel(pandas.DataFrame.from_records([self.env.goal.to_dict()])[self.features])
        obs = {
            "observation": obs / self.scales,
            "achieved_goal": obs / self.scales,
            "desired_goal": goal / self.scales
        }
        return obs

class AttributesObservation(ObservationType):
    def __init__(self, env, attributes, **kwargs):
        self.env = env
        self.attributes = attributes

    def space(self) -> spaces.Space:
        try:
            obs = self.observe()
            return spaces.Dict({
                attribute: spaces.Box(-np.inf, np.inf, shape=obs[attribute].shape, dtype=np.float32)
                for attribute in self.attributes
            })
        except AttributeError:
            return spaces.Space()

    def observe(self):
        return {
            attribute: getattr(self.env, attribute) for attribute in self.attributes
        }


def observation_factory(env, config):
    if config["type"] == "TimeToCollision":
        return TimeToCollisionObservation(env, **config)
    elif config["type"] == "Kinematics":
        return KinematicObservation(env, **config)
    elif config["type"] == "OccupancyGrid":
        return OccupancyGridObservation(env, **config)
    elif config["type"] == "KinematicsGoal":
        return KinematicsGoalObservation(env, **config)
    elif config["type"] == "GrayscaleObservation":
        return GrayscaleObservation(env, **config)
    else:
        raise ValueError("Unknown observation type")
