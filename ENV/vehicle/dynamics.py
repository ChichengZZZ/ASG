from __future__ import division, print_function

import copy

import numpy as np
import pandas as pd
from collections import deque

from NGSIM_env import utils
from NGSIM_env.logger import Loggable


class Vehicle(Loggable):
    """
    一辆在道路上移动的车辆及其动力学。

    车辆通过一个动力学系统来表示：一个修改过的自行车模型。 它的状态根据转向和加速度的动作来传播。
    """
    # Enable collision detection between vehicles 
    COLLISIONS_ENABLED = True
    # Vehicle length [m]
    LENGTH = 5.0
    # Vehicle width [m]
    WIDTH = 2.0
    # Range for random initial velocities [m/s]
    DEFAULT_VELOCITIES = [23, 25]
    # Maximum reachable velocity [m/s]
    MAX_VELOCITY = 40

    def __init__(self, road, position, heading=0, velocity=0):
        self.road = road
        self.position = np.array(position).astype('float')
        self.heading = heading
        self.velocity = velocity
        self.lane_index = self.road.network.get_closest_lane_index(self.position) if self.road else np.nan
        self.lane = self.road.network.get_lane(self.lane_index) if self.road else None
        self.action = {'steering': 0, 'acceleration': 0}
        self.crashed = False
        self.log = []
        self.history = deque(maxlen=50)
        self.is_ego = False
        self.controlled_by_model = False
        self.linear = None

    @classmethod
    def make_on_lane(cls, road, lane_index, longitudinal, velocity=0):
        """
        在指定的车道上创建一个车辆，并指定纵向位置。

        :param road: 车辆行驶的道路
        :param lane_index: 车辆所在车道的索引
        :param longitudinal: 沿车道的纵向位置
        :param velocity: 初始速度，单位为[m/s]
        :return: 在指定位置的车辆

        """
        lane = road.network.get_lane(lane_index)

        if velocity is None:
            velocity = lane.speed_limit

        return cls(road, lane.position(longitudinal, 0), lane.heading_at(longitudinal), velocity)

    @classmethod
    def create_random(cls, road, velocity=None, spacing=1):
        """
        在道路上创建一个随机的车辆。

        车辆的车道和/或速度是随机选择的，而纵向位置是在道路上最后一辆车的后方选择的，其密度基于车道数。

        :param road: 车辆行驶的道路
        :param velocity: 初始速度，单位为[m/s]。如果为None，则随机选择
        :param spacing: 前方车辆的间距比例，1为默认值
        :return: 具有随机位置和/或速度的车辆

        """
        if velocity is None:
            velocity = road.np_random.uniform(Vehicle.DEFAULT_VELOCITIES[0], Vehicle.DEFAULT_VELOCITIES[1])

        default_spacing = 1.5*velocity

        _from = road.np_random.choice(list(road.network.graph.keys()))
        _to = road.np_random.choice(list(road.network.graph[_from].keys()))
        _id = road.np_random.choice(len(road.network.graph[_from][_to]))

        offset = spacing * default_spacing * np.exp(-5 / 30 * len(road.network.graph[_from][_to]))
        x0 = np.max([v.position[0] for v in road.vehicles]) if len(road.vehicles) else 3*offset
        x0 += offset * road.np_random.uniform(0.9, 1.1)
        
        v = cls(road,
                road.network.get_lane((_from, _to, _id)).position(x0, 0),
                road.network.get_lane((_from, _to, _id)).heading_at(x0),
                velocity)

        return v

    @classmethod
    def create_from(cls, vehicle):
        """
        从现有车辆创建一个新的车辆。 只复制车辆的动力学，其他属性将使用默认值。

        :param vehicle: 车辆
        :return: 一个在相同动力学状态下的新车辆
        """

        v = cls(vehicle.road, vehicle.position, vehicle.heading, vehicle.velocity)

        return v

    def act(self, action=None):
        """
        存储一个待重复的动作。

        :param action: 输入的动作
        """

        if action:
            self.action = action

    def step(self, dt):
        """
        根据车辆的动作来传播车辆状态。

        使用带有一阶响应的车轮动力学的修改过的自行车模型进行积分。 如果车辆发生碰撞，动作会被替换为随机转向和制动，直至完全停止。 更新车辆当前的车道。

        :param dt: 模型积分的时间步长 [秒]
        """
        if self.crashed:
            self.action['steering'] = 0
            self.action['acceleration'] = -1.0*self.velocity

        if self.velocity < 0:
            self.velocity = 0

        self.action['steering'] = float(self.action['steering'])
        self.action['acceleration'] = float(self.action['acceleration'])

        if self.velocity > self.MAX_VELOCITY:
            self.action['acceleration'] = min(self.action['acceleration'], 1.0*(self.MAX_VELOCITY - self.velocity))
        elif self.velocity < -self.MAX_VELOCITY:
            self.action['acceleration'] = max(self.action['acceleration'], 1.0*(self.MAX_VELOCITY - self.velocity))

        v = self.velocity * np.array([np.cos(self.heading), np.sin(self.heading)])
        self.position += v * dt
        self.heading += self.velocity * np.tan(self.action['steering']) / self.LENGTH * dt
        self.velocity += self.action['acceleration'] * dt

        if self.road:
            self.lane_index = self.road.network.get_closest_lane_index(self.position)
            self.lane = self.road.network.get_lane(self.lane_index)
            if self.road.record_history:
                self.history.appendleft(self.create_from(self))

    def lane_distance_to(self, vehicle):
        """
        计算与当前车道上另一辆车的有向距离。

        :param vehicle: 另一辆车辆
        :return: 与另一辆车辆的距离 [米]

        """
        if not vehicle:
            return np.nan
        
        return self.lane.local_coordinates(vehicle.position)[0] - self.lane.local_coordinates(self.position)[0]

    def check_collision(self, other):
        """
        检查与另一辆车辆的碰撞。

        :param other: 另一辆车辆
        """
        if not self.COLLISIONS_ENABLED or not other.COLLISIONS_ENABLED or self.crashed or other is self:
            return

        # Fast spherical pre-check
        if np.linalg.norm(other.position - self.position) > self.LENGTH:
            return

        # Accurate rectangular check
        if utils.rotated_rectangles_intersect((self.position, 0.8*self.LENGTH, 0.8*self.WIDTH, self.heading),
                                              (other.position, 0.8*other.LENGTH, 0.8*other.WIDTH, other.heading)):
            self.velocity = other.velocity = min([self.velocity, other.velocity], key=abs)
            self.crashed = other.crashed = True

    @property
    def direction(self):
        return np.array([np.cos(self.heading), np.sin(self.heading)])

    @property
    def destination(self):
        if getattr(self, "route", None):
            last_lane = self.road.network.get_lane(self.route[-1])
            return last_lane.position(last_lane.length, 0)
        else:
            return self.position

    @property
    def destination_direction(self):
        if (self.destination != self.position).any():
            return (self.destination - self.position) / np.linalg.norm(self.destination - self.position)
        else:
            return np.zeros((2,))

    @property
    def on_road(self):
        """ Is the vehicle on its current lane, or off-road ? """
        return self.lane.on_lane(self.position)

    def front_distance_to(self, other):
        return self.direction.dot(other.position - self.position)

    def to_dict(self, origin_vehicle=None, observe_intentions=True):
        #print(self.position)
        d = {
            'presence': 1,
            'x': self.position[0],
            'y': self.position[1],
            'vx': self.velocity * self.direction[0],
            'vy': self.velocity * self.direction[1],
            'heading': self.heading,
            'w': self.WIDTH,
            'h': self.LENGTH,
            'cos_h': self.direction[0],
            'sin_h': self.direction[1],
            'cos_d': self.destination_direction[0],
            'sin_d': self.destination_direction[1],
            'vehicle_id': id(self) % 1000
        }
        if not observe_intentions:
            d["cos_d"] = d["sin_d"] = 0
        if origin_vehicle:
            origin_dict = origin_vehicle.to_dict()
            for key in ['x', 'y', 'vx', 'vy']:
                d[key] -= origin_dict[key]

        return d

    def dump(self):
        """
        更新车辆的内部日志，包括：
        - 其运动学信息；
        - 相对于相邻车辆的一些指标。
        """
        data = {
            'x': self.position[0],
            'y': self.position[1],
            'psi': self.heading,
            'vx': self.velocity * np.cos(self.heading),
            'vy': self.velocity * np.sin(self.heading),
            'v': self.velocity,
            'acceleration': self.action['acceleration'],
            'steering': self.action['steering']}

        if self.road:
            for lane_index in self.road.network.side_lanes(self.lane_index):
                lane_coords = self.road.network.get_lane(lane_index).local_coordinates(self.position)
                data.update({
                    'dy_lane_{}'.format(lane_index): lane_coords[1],
                    'psi_lane_{}'.format(lane_index): self.road.network.get_lane(lane_index).heading_at(lane_coords[0])
                })
            front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self)
            if front_vehicle:
                data.update({
                    'front_v': front_vehicle.velocity,
                    'front_distance': self.lane_distance_to(front_vehicle)
                })
            if rear_vehicle:
                data.update({
                    'rear_v': rear_vehicle.velocity,
                    'rear_distance': rear_vehicle.lane_distance_to(self)
                })

        self.log.append(data)

    def get_log(self):
        """
        将内部日志转换为DataFrame。

        :return: 车辆日志的DataFrame。

        """
        return pd.DataFrame(self.log)

    def __str__(self):
        return "{} #{}: {}".format(self.__class__.__name__, id(self) % 1000, self.position)

    def __repr__(self):
        return self.__str__()


class Obstacle(Vehicle):
    """
    一个静止的障碍物在给定的位置。
    """

    def __init__(self, road, position, heading=0):
        super(Obstacle, self).__init__(road, position, velocity=0, heading=heading)
        self.target_velocity = 0
        self.LENGTH = self.WIDTH
