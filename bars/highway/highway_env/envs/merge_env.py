import math
from typing import List, Tuple, Optional, Callable, TypeVar, Generic, Union, Dict, Text

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import action_factory, Action, DiscreteMetaAction, ActionType
from highway_env.envs.common.observation import observation_factory, ObservationType
from highway_env.road.lane import LineType, StraightLane, SineLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle, MDPVehicle
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.objects import Obstacle
import random
Observation = TypeVar("Observation")

class MergeEnv(AbstractEnv):

    """
    A highway merge negotiation environment.

    The ego-vehicle is driving on a highway and approached a merge, with some vehicles incoming on the access ramp.
    It is rewarded for maintaining a high speed and avoiding collisions, but also making room for merging
    vehicles.
    """

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "observation": {
                "type": "TrafficFlowV1",
                # "type": "Kinematics",
                "vehicles_count": 6,
                "features": ["x", "y", "vx", "vy"],
                # "features": ["presence", "x", "y", "vx", "vy", "heading", "lat_off", "ang_off"],
                "features_range": {
                    "x": [-100, 100],
                    "y": [-100, 100],
                    "vx": [-20, 20],
                    "vy": [-20, 20],
                },
                "absolute": False,
                "flatten": False,
                "observe_intentions": False
            },
            # "observation": {
            #     "type": "Kinematics",
            #     # "type": "TrafficFlow"
            # },
            "action": {
                # "type": "MultiDiscreteMetaAction",
                "type": "DiscreteMetaAction"
            },
            "duration": 500,
            "simulation_frequency": 10,  # [Hz]
            "policy_frequency": 10,  # [Hz]
            "screen_width": 600,  # [px]
            "screen_height": 300,  # [px]
            "collision_reward": -1,
            "right_lane_reward": 0.0,
            "high_speed_reward": 0.4,
            "duration_time_reward": 0.2,
            "merging_speed_reward": -0.5,
            "lane_change_reward": 0.1,
            "lane_change_in_merge_lane_reward": -1,
            'ttc_reward': -1,
            'adv_front_ttc_coefficient': 1,
            'adv_rear_ttc_coefficient': 2,
            "decelerate_reward": 0.1,
            "reward_version": 0,
            "max_ttc_value": 5,
        })
        return cfg

    def _reward(self, action: int) -> float:
        """
        对抗奖励设计的核心是主车与后车TTC越小奖励越大,但如果主车与前方车辆即将发生碰撞则奖励只考虑主车与前车TTC以避免放生碰撞

        :param action: the action performed
        :return: the reward of the state-action transition
        返回状态-动作转换的奖励
        """

        reward = 0.0
        distance_th = 40
        # 定义距离阈值
        front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self.vehicle, self.vehicle.lane_index)
        # 获取前车和后车信息

        if front_vehicle is not None and abs(
                np.linalg.norm(front_vehicle.position - self.vehicle.position)) > distance_th:
            # 如果前车存在，并且与本车的距离超过了距离阈值
            front_vehicle = None
            # 将前车置为空

        if rear_vehicle is not None and abs(
                np.linalg.norm(rear_vehicle.position - self.vehicle.position)) > distance_th:
            # 如果后车存在，并且与本车的距离超过了距离阈值
            rear_vehicle = None
            # 将后车置为空

        front_ttc, rear_ttc = 50, 50
        # 初始化前车和后车的时间间隔
        vehicle_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        # 计算本车速度在道路方向上的分量

        if rear_vehicle is not None:
            # 如果后车存在
            v_speed = rear_vehicle.speed * np.cos(rear_vehicle.heading)
            # 计算后车速度在道路方向上的分量
            if v_speed > vehicle_speed:
                # 如果后车速度大于本车速度
                rear_ttc = min(rear_ttc, (self.vehicle.position[0] - rear_vehicle.position[0]) / utils.not_zero(
                    v_speed - vehicle_speed))
                # 计算后车与本车的时间间隔

        if front_vehicle is not None:
            # 如果前车存在
            v_speed = front_vehicle.speed * np.cos(front_vehicle.heading)
            # 计算前车速度在道路方向上的分量
            if v_speed < vehicle_speed:
                # 如果前车速度小于本车速度
                front_ttc = min(front_ttc, (front_vehicle.position[0] - self.vehicle.position[0]) / utils.not_zero(
                    vehicle_speed - v_speed))
                # 计算前车与本车的时间间隔

        if rear_vehicle is None:
            # 如果后车不存在
            if self.vehicle.speed < 3:
                # 如果本车速度小于3
                reward += -1
                # 奖励减1
            else:
                if self.vehicle.lane_index[2] in [1, 2, 3] and action in [0, 2]:
                    # 如果本车所在的车道索引在[1, 2, 3]内，并且执行动作为[0, 2]之一
                    reward += self.config['lane_change_reward']
                    # 奖励加上车道变换的奖励
                elif self.vehicle.lane_index not in [("j", "k", 0), ("k", "b", 0), ('d', 'q', 0), ('q', 'r', 0)]:
                    # 如果本车所在的车道索引不是[("j", "k", 0), ("k", "b", 0), ('d', 'q', 0), ('q', 'r', 0)]之一
                    if self.vehicle.lane_index[2] == 0 and action == 2 or self.vehicle.lane_index[
                        2] == 4 and action == 0:
                        # 如果本车所在的车道索引的第三位为0，并且执行动作为2；或者车道索引的第三位为4，并且执行动作为0
                        reward += self.config['lane_change_reward']
                        # 奖励加上车道变换的奖励
        else:
            # 如果后车存在
            # ttc_reward = (1 - min_ttc - 0.5) * 2 + 0.1 + 1 / min_ttc
            reward += math.exp(-rear_ttc / 5) * self.config['adv_rear_ttc_coefficient'] + 0.1
            # 计算后车与本车之间的时间间隔奖励
            # 避免主车停止不动
            if self.vehicle.speed < 1:
                # 如果本车速度小于1
                reward += -0.1 + -math.exp(-5 / 5) * self.config['adv_rear_ttc_coefficient']
                # 奖励减去0.1和与后车之间的时间间隔奖励的乘积

        if front_ttc <= 5:
            # 如果前车与本车之间的时间间隔小于等于5
            reward = -math.exp(-front_ttc / 5) * self.config['adv_front_ttc_coefficient']
            # 奖励设为前车与本车之间的时间间隔奖励的相反数
            if self.vehicle.crashed:
                # 如果发生碰撞
                reward += self.config["collision_reward"]
                # 奖励加上碰撞奖励
            return reward

        if self.vehicle.crashed:
            # 如果发生碰撞
            reward = self.config["collision_reward"]
            # 奖励设为碰撞奖励

        return reward

    def _rewards(self, action: int) -> Dict[Text, float]:
        # min_dis = 20
        # close_vehicles = self.road.close_objects_to(self.vehicle,
        #                                                 20,
        #                                                 count=10,
        #                                                 see_behind=True,
        #                                                 sort=True,
        #                                                 vehicles_only=True)
        # ob_vehicle = None
        # for v in close_vehicles:
        #     min_dis = min(np.linalg.norm(v.position - self.vehicle.position), min_dis)

        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        forward_speed = self.vehicle.speed
        return {
            # "collision_reward": self.vehicle.crashed,
            # "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": forward_speed / 20,
            # "lane_change_reward": action in [0, 2],
            # "merging_speed_reward": sum(  # Altruistic penalty
            #     (vehicle.target_speed - vehicle.speed) / vehicle.target_speed
            #     for vehicle in self.road.vehicles
            #     if vehicle.lane_index == ("b", "c", 4) and isinstance(vehicle, ControlledVehicle)
            # ),
            # "duration_time_reward": self.run_steps / self.config['duration'],
            # "ttc_reward": (20 - min_dis) / 20,
        }

    def _is_terminated(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return self.vehicle.crashed or bool(self.vehicle.position[0] > 500) or self.run_steps >= self.config['duration']

    def _is_truncated(self) -> bool:
        return False

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()
        self.run_steps = 0
        # for rewards
        self.last_step_have_bv = False
        self.last_step_min_ttc = 50


    def _make_road(self) -> None:
        """
        Make a road composed of a straight highway and a merging lane.

        :return: the road
        """
        net = RoadNetwork()
        w = 4
        # Highway lanes
        ends = [150, 80, 40, 40, 80, 150]  # Before, converging, merge, after
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        y = [w * i for i in range(5)]
        line_type = [[c, s], [n, s], [n, s], [n, s], [n, c]]
        for i in range(5):
            net.add_lane("a", "b", StraightLane([0, y[i]], [sum(ends[:2]), y[i]], line_types=line_type[i], width=w))
            net.add_lane("b", "c",
                         StraightLane([sum(ends[:2]), y[i]], [sum(ends[:3]), y[i]], line_types=line_type[i], width=w))
            net.add_lane("c", "d",
                         StraightLane([sum(ends[:3]), y[i]], [sum(ends[:4]), y[i]], line_types=line_type[i], width=w))
            net.add_lane("d", "e",
                         StraightLane([sum(ends[:4]), y[i]], [sum(ends[:5]), y[i]], line_types=line_type[i], width=w))
            net.add_lane("e", "f",
                         StraightLane([sum(ends[:5]), y[i]], [sum(ends), y[i]], line_types=line_type[i], width=w))

        # Merging lane
        amplitude = 6.25
        ljk = StraightLane([0, 0.5 + 4 * 7], [ends[0], 0.5 + 4 * 7], line_types=[c, c], forbidden=True)
        lkb = SineLane(ljk.position(ends[0], -amplitude), ljk.position(sum(ends[:2]), -amplitude),
                       amplitude, 2 * np.pi / (2 * ends[1]), np.pi / 2, line_types=[c, c], forbidden=True)

        lpq = SineLane(ljk.position(sum(ends[:4]), -amplitude), ljk.position(sum(ends[:5]), -amplitude),
                       amplitude, 2 * np.pi / (2 * ends[4]), -np.pi / 2, line_types=[c, c], forbidden=True)

        lqr = StraightLane([sum(ends[:5]), 0.5 + 4 * 7], [sum(ends), 0.5 + 4 * 7], line_types=[c, c], forbidden=True)

        net.add_lane("j", "k", ljk)
        net.add_lane("k", "b", lkb)

        net.add_lane('d', 'q', lpq)
        net.add_lane('q', 'r', lqr)

        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        # road.objects.append(Obstacle(road, lbc.position(ends[2], 0)))
        self.road = road

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.

        :return: the ego-vehicle
        """

        if random.random() > 0.7:
            self.vehicle = MDPVehicle.make_on_lane(self.road, ("j", "k", 0), random.choice(range(40, 100)), speed=random.choice(range(5, 10)))
        else:
            idx = random.choice(range(5))
            self.vehicle = MDPVehicle.make_on_lane(self.road, ("a", "b", idx), random.choice(range(40, 100)), speed=random.choice(range(5, 10)))

        if random.random() > 0.3:
            self.vehicle.plan_route_to('f')
        else:
            self.vehicle.plan_route_to('r')

        self.road.vehicles.append(self.vehicle)

        n_vehicles = random.choice(range(10, 50))
        for t in range(n_vehicles - 1):
            self._spawn_vehicle()
    
    def _spawn_vehicle(self, spawn_probability: float = 0.7) -> None:
        vehicle_type = IDMVehicle
        vehicle = None

        if random.random() > spawn_probability:
            vehicle = vehicle_type.make_on_lane(self.road, ("j", "k", 0), random.choice(range(150)), speed=random.choice(range(5, 10)))
            # vehicle.plan_route_to(("p", "q", 0))
        else:
            vehicle = vehicle_type.make_on_lane(self.road, ("a", "b", random.choice(range(5))), random.choice(range(150)), speed=random.choice(range(5, 10)))

        vehicle1 = vehicle_type.make_on_lane(self.road, ("b", "c", random.choice(range(5))), random.choice(range(80)),
                                            speed=random.choice(range(5, 10)))
        vehicle2 = vehicle_type.make_on_lane(self.road, ("c", "d", random.choice(range(5))), random.choice(range(40)),
                                            speed=random.choice(range(5, 10)))

        vehicles = [vehicle, vehicle1, vehicle2]
        for i in range(len(vehicles)):
            if random.random() > 0.9 or vehicles[i].lane_index[2] < 4:
                vehicles[i].plan_route_to('f')
            else:
                vehicles[i].plan_route_to('r')

            for v in self.road.vehicles:
                if np.linalg.norm(v.position - vehicles[i].position) < 10:
                    break
            else:
                self.road.vehicles.append(vehicles[i])

    def _simulate(self, action: Optional[Action] = None) -> None:
        """Perform several steps of simulation with constant action."""
        frames = int(self.config["simulation_frequency"] // self.config["policy_frequency"])
        for frame in range(frames):
            # Forward action to the vehicle
            if action is not None \
                    and not self.config["manual_control"] \
                    and self.steps % int(self.config["simulation_frequency"] // self.config["policy_frequency"]) == 0:
                self.action_type.act(action)

            self.road.act()
            self.road.step(1 / self.config["simulation_frequency"])
            self.run_steps += 1

            # Automatically render intermediate simulation steps if a viewer has been launched
            # Ignored if the rendering is done offscreen
            if frame < frames - 1:  # Last frame will be rendered through env.render() as usual
                self._automatic_rendering()

        self.enable_auto_render = False