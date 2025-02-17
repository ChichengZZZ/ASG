import random
from typing import Tuple, Dict, Text

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane, CircularLane, SineLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import MDPVehicle
import math

class RoundaboutEnv(AbstractEnv):

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "TrafficFlowV2",
                "vehicles_count": 11,
                "features": ["x", "y", "vx", "vy"],
                "features_range": {
                    "x": [-100, 100],
                    "y": [-100, 100],
                    "vx": [-20, 20],
                    "vy": [-20, 20],
                },
                "absolute": True,
                "flatten": False,
                "observe_intentions": False
            },
            "action": {
                "type": "DiscreteMetaAction",
                "longitudinal": True,
                "lateral": True,
                "target_speeds": np.linspace(0, 10, 5)
            },
            "simulation_frequency": 10,  # [Hz]
            "policy_frequency": 10,  # [Hz]
            "incoming_vehicle_destination": None,
            "collision_reward": -5,
            "high_speed_reward": 0.2,
            "right_lane_reward": 0,
            "lane_change_reward": 0.2,
            "screen_width": 600,
            "screen_height": 600,
            "centering_position": [0.5, 0.6],
            "duration": 50,
            "normalize_reward": True,
            "reward_version": 0,
            'adv_front_ttc_coefficient': 2,
            'adv_rear_ttc_coefficient': 2,
            "adv_roundabout_ttc_coefficient": 2,
        })
        return config

    def _reward(self, action: int) -> float:
        # 删除障碍物车辆的颜色属性，如果存在
        if self.delete_ob_vehicle is not None and hasattr(self.delete_ob_vehicle, 'color'):
            delattr(self.delete_ob_vehicle, 'color')

        # 设置距离阈值，离自车距离小于该阈值的车辆被认为是近邻车辆
        distance_th = 20
        ob_vehicle = None
        # 获取近邻车辆列表
        close_vehicles = self.observation_type.close_objects_to(self.vehicle,
                                                                distance_th,
                                                                count=10,
                                                                see_behind=True,
                                                                sort=True,
                                                                vehicles_only=True)
        # 遍历近邻车辆列表，找到横向上与自车不在同一条道路但纵向距离较近的车辆作为目标车辆
        for v in close_vehicles:
            if self.vehicle.lane_index[0] != v.lane_index[0] and self.vehicle.lane_index[1] == v.lane_index[1]:
                ob_vehicle = v
                break

        reward = 0.0
        front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self.vehicle, self.vehicle.lane_index)
        # 如果与前车的距离大于设定的距离阈值，则将前车置为None
        if front_vehicle is not None and abs(
                np.linalg.norm(front_vehicle.position - self.vehicle.position)) > distance_th:
            front_vehicle = None
        # 如果与后车的距离大于设定的距离阈值，则将后车置为None
        if rear_vehicle is not None and abs(
                np.linalg.norm(rear_vehicle.position - self.vehicle.position)) > distance_th:
            rear_vehicle = None
        rear_ttc = 50

        if self.last_ob_vehicle is not None:
            bv_front_vehicle, _ = self.road.neighbour_vehicles(self.last_ob_vehicle, self.last_ob_vehicle.lane_index)
            av_front_vehicle, _ = self.road.neighbour_vehicles(self.vehicle, self.vehicle.lane_index)
            dis_av_with_bv = np.linalg.norm(self.vehicle.position - self.last_ob_vehicle.position)
            # 如果自车在目标车辆正前方且距离较近，则增加奖励
            if bv_front_vehicle == self.vehicle and dis_av_with_bv < 8:
                reward += 0.5
                reward += math.exp(-dis_av_with_bv / 8) * 2
            # 如果目标车辆在自车正前方且距离较近，则减少奖励
            elif av_front_vehicle == self.last_ob_vehicle and dis_av_with_bv < 8:
                reward += -0.5
                reward += -math.exp(-dis_av_with_bv / 8) * 2
            # 如果自车和目标车辆间距离较近，则减少奖励
            elif dis_av_with_bv < 5:
                reward += -math.exp(-dis_av_with_bv / 4) * 2
            # 如果自车和目标车辆间距离较远，则增加奖励
            else:
                reward += math.exp(-dis_av_with_bv / 4) * 2

        elif rear_vehicle is not None:
            # 计算与后车的ttc
            av_lane = self.road.network.get_lane(self.vehicle.lane_index)
            av_lon, _ = av_lane.local_coordinates(self.vehicle.position)
            rear_veh_lon, _ = av_lane.local_coordinates(rear_vehicle.position)
            av_angle = av_lane.local_angle(self.vehicle.heading, av_lon)
            rear_veh_angle = av_lane.local_angle(rear_vehicle.heading, rear_veh_lon)

            bv_speed = rear_vehicle.speed * np.cos(rear_veh_angle)
            av_speed = self.vehicle.speed * np.cos(av_angle)

            if bv_speed > av_speed:
                rear_ttc = min(rear_ttc, (av_lon - rear_veh_lon) / utils.not_zero(bv_speed - av_speed))

            reward += math.exp(-rear_ttc / 5) * 2 + self.config['lane_change_reward']
            # 避免主车停止不动
            if self.vehicle.speed < 1:
                reward += -self.config['lane_change_reward']

        else:
            if self.vehicle.speed < 1:
                reward += -1
            elif len(self.vehicle.lane_index[0]) == len(self.vehicle.lane_index[1]) == 2:
                if (self.vehicle.lane_index[2] == 0 and action in [2]) or (
                        self.vehicle.lane_index[2] == 2 and action in [0]) or \
                        (self.vehicle.lane_index[2] == 1 and action in [0, 2]):
                    reward += self.config['lane_change_reward']

        # 更新目标车辆信息
        if self.last_ob_vehicle is not None:
            self.delete_ob_vehicle = self.last_ob_vehicle
        self.last_ob_vehicle = ob_vehicle
        if self.last_ob_vehicle is not None:
            setattr(self.last_ob_vehicle, 'color', (0, 0, 255))
        # 当即将撞上前车时，只考虑与前车产生的奖励
        front_ttc = 3
        if front_vehicle is not None:
            # 计算与前车的ttc
            av_lane = self.road.network.get_lane(self.vehicle.lane_index)
            av_lon, _ = av_lane.local_coordinates(self.vehicle.position)
            front_veh_lon, _ = av_lane.local_coordinates(front_vehicle.position)
            av_angle = av_lane.local_angle(self.vehicle.heading, av_lon)
            front_veh_angle = av_lane.local_angle(front_vehicle.heading, front_veh_lon)

            bv_speed = front_vehicle.speed * np.cos(front_veh_angle)
            av_speed = self.vehicle.speed * np.cos(av_angle)

            if bv_speed < av_speed:
                front_ttc = min(front_ttc, (front_veh_lon - av_lon) / utils.not_zero(av_speed - bv_speed))

        if front_ttc < 3:
            reward = -math.exp(-front_ttc / 5) * 2
            if self.vehicle.crashed:
                reward += self.config["collision_reward"]
            return reward
        else:
            if self.vehicle.crashed:
                reward = self.config["collision_reward"]
            return reward

    def _rewards(self, action: int) -> Dict[Text, float]:
        return {
            "collision_reward": self.vehicle.crashed,
            "high_speed_reward":
                 MDPVehicle.get_speed_index(self.vehicle) / (MDPVehicle.DEFAULT_TARGET_SPEEDS.size - 1),
            "lane_change_reward": action in [0, 2],
            "on_road_reward": self.vehicle.on_road
        }

    def _is_terminated(self) -> bool:
        return self.vehicle.crashed

    def _is_truncated(self) -> bool:
        destinations = ["exr", "sxr", "nxr", "wxr"]
        return self.time >= self.config["duration"] or (self.vehicle.lane_index[1] in destinations and self.vehicle.lane.local_coordinates(self.vehicle.position)[0] >= self.vehicle.lane.length - 4 * self.vehicle.LENGTH)

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()
        self.last_ob_vehicle = None
        self.delete_ob_vehicle = None

    def _make_road(self) -> None:
        # Circle lanes: (s)outh/(e)ast/(n)orth/(w)est (e)ntry/e(x)it.
        center = [0, 0]  # [m]
        radius = 30  # [m]
        alpha = 24  # [deg]

        net = RoadNetwork()
        radii = [radius-4, radius, radius+4]
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        line = [[c, s], [n, s], [n, c]]
        for lane in [0, 1, 2]:
            net.add_lane("se", "ex",
                         CircularLane(center, radii[lane], np.deg2rad(90 - alpha), np.deg2rad(alpha),
                                      clockwise=False, line_types=line[lane], priority=lane+1))
            net.add_lane("ex", "ee",
                         CircularLane(center, radii[lane], np.deg2rad(alpha), np.deg2rad(-alpha),
                                      clockwise=False, line_types=line[lane], priority=lane+1))
            net.add_lane("ee", "nx",
                         CircularLane(center, radii[lane], np.deg2rad(-alpha), np.deg2rad(-90 + alpha),
                                      clockwise=False, line_types=line[lane], priority=lane+1))
            net.add_lane("nx", "ne",
                         CircularLane(center, radii[lane], np.deg2rad(-90 + alpha), np.deg2rad(-90 - alpha),
                                      clockwise=False, line_types=line[lane], priority=lane+1))
            net.add_lane("ne", "wx",
                         CircularLane(center, radii[lane], np.deg2rad(-90 - alpha), np.deg2rad(-180 + alpha),
                                      clockwise=False, line_types=line[lane], priority=lane+1))
            net.add_lane("wx", "we",
                         CircularLane(center, radii[lane], np.deg2rad(-180 + alpha), np.deg2rad(-180 - alpha),
                                      clockwise=False, line_types=line[lane], priority=lane+1))
            net.add_lane("we", "sx",
                         CircularLane(center, radii[lane], np.deg2rad(180 - alpha), np.deg2rad(90 + alpha),
                                      clockwise=False, line_types=line[lane], priority=lane+1))
            net.add_lane("sx", "se",
                         CircularLane(center, radii[lane], np.deg2rad(90 + alpha), np.deg2rad(90 - alpha),
                                      clockwise=False, line_types=line[lane], priority=lane+1))

        # Access lanes: (r)oad/(s)ine
        access = 200  # [m]
        dev = 115  # [m]
        a = 5  # [m]
        delta_st = 0.2*dev  # [m]

        delta_en = dev-delta_st*0.95
        w = 2*np.pi/dev
        net.add_lane("ser", "ses", StraightLane([2, access], [2, dev/2], line_types=(s, c), speed_limit=10, priority=0))
        net.add_lane("ses", "se", SineLane([2+a, dev/2], [2+a, dev/2-delta_st], a, w, -np.pi/2, line_types=(c, c), speed_limit=10, priority=0))
        net.add_lane("sx", "sxs", SineLane([-2-a, -dev/2+delta_en], [-2-a, dev/2], a, w, -np.pi/2+w*delta_en, line_types=(c, c), speed_limit=10, priority=0))
        net.add_lane("sxs", "sxr", StraightLane([-2, dev / 2], [-2, access], line_types=(n, c), speed_limit=10, priority=0))

        net.add_lane("eer", "ees", StraightLane([access, -2], [dev / 2, -2], line_types=(s, c), speed_limit=10, priority=0))
        net.add_lane("ees", "ee", SineLane([dev / 2, -2-a], [dev / 2 - delta_st, -2-a], a, w, -np.pi / 2, line_types=(c, c), speed_limit=10, priority=0))
        net.add_lane("ex", "exs", SineLane([-dev / 2 + delta_en, 2+a], [dev / 2, 2+a], a, w, -np.pi / 2 + w * delta_en, line_types=(c, c), speed_limit=10, priority=0))
        net.add_lane("exs", "exr", StraightLane([dev / 2, 2], [access, 2], line_types=(n, c), speed_limit=10, priority=0))

        net.add_lane("ner", "nes", StraightLane([-2, -access], [-2, -dev / 2], line_types=(s, c), speed_limit=10, priority=0))
        net.add_lane("nes", "ne", SineLane([-2 - a, -dev / 2], [-2 - a, -dev / 2 + delta_st], a, w, -np.pi / 2, line_types=(c, c), speed_limit=10, priority=0))
        net.add_lane("nx", "nxs", SineLane([2 + a, dev / 2 - delta_en], [2 + a, -dev / 2], a, w, -np.pi / 2 + w * delta_en, line_types=(c, c), speed_limit=10, priority=0))
        net.add_lane("nxs", "nxr", StraightLane([2, -dev / 2], [2, -access], line_types=(n, c), speed_limit=10, priority=0))

        net.add_lane("wer", "wes", StraightLane([-access, 2], [-dev / 2, 2], line_types=(s, c), speed_limit=10, priority=0))
        net.add_lane("wes", "we", SineLane([-dev / 2, 2+a], [-dev / 2 + delta_st, 2+a], a, w, -np.pi / 2, line_types=(c, c), speed_limit=10, priority=0))
        net.add_lane("wx", "wxs", SineLane([dev / 2 - delta_en, -2-a], [-dev / 2, -2-a], a, w, -np.pi / 2 + w * delta_en, line_types=(c, c), speed_limit=10, priority=0))
        net.add_lane("wxs", "wxr", StraightLane([-dev / 2, -2], [-access, -2], line_types=(n, c), speed_limit=10, priority=0))

        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        self.road = road

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.

        :return: the ego-vehicle
        """
        speed_deviation = 2
        start_lane = self.np_random.choice([("ser", "ses"), ("eer", "ees"), ("ner", "nes"), ("wer", "wes")], size=1, replace=False)
        # Ego-vehicle
        ego_lane = self.road.network.get_lane((start_lane[0][0], start_lane[0][1], 0))
        ego_vehicle = self.action_type.vehicle_class(self.road,
                                                     ego_lane.position(125, 0),
                                                     speed=random.choice(range(4, 10)),
                                                     heading=ego_lane.heading_at(140))
        try:
            end_lane = self.np_random.choice(["exr", "sxr", "nxr", "wxr"])
            ego_vehicle.plan_route_to(end_lane)
        except AttributeError:
            pass
        self.road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle
        destinations = ["exr", "sxr", "nxr", "wxr"]
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        for k, v in self.road.network.graph.items():
            lane_start = k
            for k1, v1 in v.items():
                lane_end = k1
                lane_idx = random.choice(range(len(v1)))
                lane = self.road.network.get_lane((lane_start, lane_end, lane_idx))
                vehicle = other_vehicles_type.make_on_lane(self.road,
                                                           (lane_start, lane_end, lane_idx),
                                                           longitudinal=random.choice(range(round(lane.length))),
                                                           speed=5 + self.np_random.normal() * speed_deviation)
                if len(lane_start) == len(lane_end) == 2 and lane_idx == 2:
                    destination = self.np_random.choice(destinations)
                    vehicle.plan_route_to(destination)
                    vehicle.randomize_behavior()

                for v in self.road.vehicles:  # Prevent early collisions
                    if np.linalg.norm(v.position - vehicle.position) < 5:
                        break
                else:
                    self.road.vehicles.append(vehicle)

        for v in self.road.vehicles:  # Prevent early collisions
            if v is not self.vehicle and np.linalg.norm(v.position - self.vehicle.position) < 10:
                self.road.vehicles.remove(v)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = super().step(action)
        self._clear_vehicles()
        # self._spawn_vehicle()
        return obs, reward, terminated, truncated, info

    def _spawn_vehicle(self):
        destinations = ["exr", "sxr", "nxr", "wxr"]
        n_vehicle = random.choice(range(5))
        for i in range(n_vehicle):
            route = self.np_random.choice([("ser", "ses"), ("eer", "ees"), ("ner", "nes"), ("wer", "wes")], size=1, replace=False)
            vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
            vehicle = vehicle_type.make_on_lane(self.road, (route[0][0], route[0][1], 0),
                                                longitudinal=(random.choice(range(200))),
                                                speed=random.choice(range(1, 10)))
            for v in self.road.vehicles:
                if np.linalg.norm(v.position - vehicle.position) < 10:
                    break
            vehicle.plan_route_to(random.choice(destinations))
            vehicle.randomize_behavior()
            self.road.vehicles.append(vehicle)

    def _clear_vehicles(self) -> None:
        destinations = ["exr", "sxr", "nxr", "wxr"]
        is_leaving = lambda vehicle: vehicle.lane_index[1] in destinations and vehicle.lane.local_coordinates(vehicle.position)[0] >= vehicle.lane.length - 4 * vehicle.LENGTH
        self.road.vehicles = [vehicle for vehicle in self.road.vehicles if
                              vehicle in self.controlled_vehicles or not is_leaving(vehicle)]

