import random
from typing import Dict, Tuple, Text, Optional
import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv, MultiAgentWrapper
from highway_env.road.lane import LineType, StraightLane, CircularLane, AbstractLane
from highway_env.road.regulation import RegulatedRoad
from highway_env.road.road import RoadNetwork
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.controller import ControlledVehicle, MDPVehicle
from highway_env.vehicle.behavior import IDMVehicle
import math

class IntersectionEnv(AbstractEnv):

    # ACTIONS: Dict[int, str] = {
    #     0: 'SLOWER',
    #     1: 'IDLE',
    #     2: 'FASTER'
    # }
    # ACTIONS_INDEXES = {v: k for k, v in ACTIONS.items()}
    PERCEPTION_DISTANCE = 100
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
            "duration": 50,  # [s]
            "controlled_vehicles": 1,
            "initial_vehicle_count": 10,
            "spawn_probability": 0.6,
            "screen_width": 600,
            "screen_height": 600,
            "centering_position": [0.5, 0.6],
            "scaling": 5.5 * 1.3,
            "collision_reward": -5,
            "arrived_reward": 1,
            "reward_speed_range": [7.0, 9.0],
            "normalize_reward": False,
            "offroad_terminal": False,
            "reward_version": 1,
            "simulation_frequency": 10,  # [Hz]
            "policy_frequency": 10,  # [Hz]
            "high_speed_reward": 0.4,
            "lane_change_reward": 0.1,
            'adv_front_ttc_coefficient': 2,
            'adv_rear_ttc_coefficient': 2,
            "adv_interaction_ttc_coefficient": 2,
            "time_difference": 0.2,
            "deceleration_reward": 0.5,
        })
        return config

    # 定义一个私有方法_reward，参数为动作action，返回值为浮点数
    def _reward(self, action: int) -> float:
        """Aggregated reward, for cooperative agents."""

        # 初始化奖励为0
        reward = 0.0

        # 如果车辆所在车道没有冲突
        if 'o' in self.vehicle.lane_index[0] or 'o' in self.vehicle.lane_index[1]:
            # 定义距离阈值为20
            distance_th = 20

            # 获取前车和后车
            front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self.vehicle, self.vehicle.lane_index)

            # 如果前车不为空并且与前车的距离大于距离阈值，则将前车置为空
            if front_vehicle is not None and abs(
                    np.linalg.norm(front_vehicle.position - self.vehicle.position)) > distance_th:
                front_vehicle = None

            # 如果后车不为空并且与后车的距离大于距离阈值，则将后车置为空
            if rear_vehicle is not None and abs(
                    np.linalg.norm(rear_vehicle.position - self.vehicle.position)) > distance_th:
                rear_vehicle = None

            # 初始化前车和后车的时间间隔
            front_ttc, rear_ttc = 50, 50

            # 如果前车不为空，则计算与前车的时间间隔
            if front_vehicle is not None:
                # 获取车道对象
                av_lane = self.road.network.get_lane(self.vehicle.lane_index)

                # 获取车辆在车道上的局部坐标和角度
                av_lon, _ = av_lane.local_coordinates(self.vehicle.position)
                front_veh_lon, _ = av_lane.local_coordinates(front_vehicle.position)
                av_angle = av_lane.local_angle(self.vehicle.heading, av_lon)
                front_veh_angle = av_lane.local_angle(front_vehicle.heading, front_veh_lon)

                # 计算车辆速度在车道上的分量
                bv_speed = front_vehicle.speed * np.cos(front_veh_angle)
                av_speed = self.vehicle.speed * np.cos(av_angle)

                # 如果后车速度小于自车速度，则计算前车的时间间隔
                if bv_speed < av_speed:
                    front_ttc = min(front_ttc, (front_veh_lon - av_lon) / utils.not_zero(av_speed - bv_speed))

            # 当即将撞上前车时，只考虑与前车产生的奖励
            if front_ttc <= 5:
                reward += -math.exp(-front_ttc / 5) * self.config['adv_front_ttc_coefficient']

            # 如果车辆碰撞了，则奖励为碰撞奖励值
            if self.vehicle.crashed:
                reward = self.config["collision_reward"]

            # 返回奖励值
            return reward

        # 如果车辆所在车道有冲突
        else:
            # 获取与车辆最近的10辆车辆
            close_vehicles = self.observation_type.close_objects_to(self.vehicle,
                                                                    60,
                                                                    count=10,
                                                                    see_behind=True,
                                                                    sort=True,
                                                                    vehicles_only=True)
            # 设置最小交互时间差为5
            min_interaction_time_diff = 5

            # 初始化障碍车辆
            ob_vehicle = None

            # 初始化时间间隔及道路权重
            av_to_p_time = bv_to_p_time = 0.0
            bv_to_av_pos_time = 10
            av_to_p_lon_diff = 0.0
            max_av_to_p_lon_diff = 20

            # 遍历所有与车辆最近的车辆
            for v in close_vehicles:
                if hasattr(v, 'color'):
                    delattr(v, 'color')

                # 如果车辆所在车道与其他车辆所在车道不同且不与目标车辆有冲突
                if 'o' not in v.lane_index[0] and 'o' not in v.lane_index[1] and self.vehicle.lane_index[
                                                                                 :-1] != v.lane_index[:-1]:
                    # 获取交互时间差
                    time_diff, av_to_p_time_, bv_to_p_time_, bv_to_av_pos_time_, av_to_p_lon_diff_ = self.calculate_interaction_time_difference(
                        self.vehicle, v)

                    # 如果交互时间差小于最小交互时间差，则更新最小交互时间差和障碍车辆
                    if min_interaction_time_diff > time_diff:
                        ob_vehicle = v
                        min_interaction_time_diff = time_diff
                        bv_to_av_pos_time = bv_to_av_pos_time_
                        av_to_p_lon_diff = min(max_av_to_p_lon_diff, av_to_p_lon_diff_)

            # 如果没有障碍车辆或者与障碍车辆的距离大于允许的最大距离
            if ob_vehicle is None or np.linalg.norm(ob_vehicle.position - self.vehicle.position) > max_av_to_p_lon_diff:
                # 如果车速小于1，则奖励值减1，否则根据动作和车道判断是否为变道操作，来给予相应奖励值
                if self.vehicle.speed < 1:
                    reward += -1
                else:
                    # 获取前车和后车
                    front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self.vehicle, self.vehicle.lane_index)

                    # 如果后车为空或者与后车的距离大于等于10，则根据动作和车道判断是否为变道操作，来给予相应奖励值
                    if rear_vehicle == None or np.linalg.norm(rear_vehicle.position - self.vehicle.position) >= 10:
                        if self.vehicle.lane_index[2] == 0 and action in [2] or self.vehicle.lane_index[
                            2] == 1 and action in [0]:
                            reward += self.config['lane_change_reward']
                    else:
                        # 获取车道对象
                        av_lane = self.road.network.get_lane(self.vehicle.lane_index)

                        # 获取车辆在车道上的局部坐标和角度
                        av_lon, _ = av_lane.local_coordinates(self.vehicle.position)
                        rear_veh_lon, _ = av_lane.local_coordinates(rear_vehicle.position)
                        av_angle = av_lane.local_angle(self.vehicle.heading, av_lon)
                        rear_veh_angle = av_lane.local_angle(rear_vehicle.heading, rear_veh_lon)

                        # 计算车辆速度在车道上的分量
                        bv_speed = rear_vehicle.speed * np.cos(rear_veh_angle)
                        av_speed = self.vehicle.speed * np.cos(av_angle)

                        # 初始化后车的时间间隔
                        rear_ttc = 10.0

                        # 如果后车速度大于自车速度，则计算后车的时间间隔
                        if bv_speed > av_speed:
                            rear_ttc = min(rear_ttc, (av_lon - rear_veh_lon) / utils.not_zero(bv_speed - av_speed))

                        # 根据后车的时间间隔计算奖励值
                        reward += math.exp(-rear_ttc / 5) * self.config['adv_rear_ttc_coefficient'] + 0.1
            else:
                # 将障碍车辆颜色设为蓝色
                setattr(ob_vehicle, 'color', (0, 0, 255))

                # 获取障碍车辆前方的车辆
                bv_front_vehicle, _ = self.road.neighbour_vehicles(ob_vehicle, ob_vehicle.lane_index)

                # 获取车辆自身前方的车辆
                av_front_vehicle, _ = self.road.neighbour_vehicles(self.vehicle, self.vehicle.lane_index)

                # 计算车辆与障碍车辆的距离
                dis_av_with_bv = np.linalg.norm(self.vehicle.position - ob_vehicle.position)

                # 如果障碍车辆前方的车辆是自车，并且与障碍车辆的距离小于8，则奖励值加0.5
                if bv_front_vehicle == self.vehicle and dis_av_with_bv < 8:
                    reward += 0.5

                    # 根据与障碍车辆的距离计算奖励值
                    reward += math.exp(-dis_av_with_bv / 8) * self.config['adv_interaction_ttc_coefficient']

                    # 如果动作为刹车操作，则奖励值加0.5
                    if action in [4]:
                        reward += 0.5
                # 如果自车在障碍车辆前方，并且与障碍车辆的距离小于8，则奖励值减0.5
                elif av_front_vehicle == ob_vehicle and dis_av_with_bv < 8:
                    reward += -0.5

                    # 根据与障碍车辆的距离计算奖励值
                    reward += -math.exp(-dis_av_with_bv / 8) * self.config['adv_interaction_ttc_coefficient']
                else:
                    # 根据最小交互时间差和最大距离差计算奖励值
                    reward += math.exp(-min_interaction_time_diff / 4) * self.config[
                        'adv_interaction_ttc_coefficient'] * (
                                          max_av_to_p_lon_diff - dis_av_with_bv) / max_av_to_p_lon_diff

            # 当即将撞上前车时，只考虑与前车产生的奖励
            # 获取与前车相邻的车辆对象
            front_vehicle, _ = self.road.neighbour_vehicles(self.vehicle, self.vehicle.lane_index)
            # 初始化与前车的时间到达（Time to Collision, TTC）为2
            front_ttc = 2
            if front_vehicle is not None:  # 如果存在前车
                # 获取车道对象
                av_lane = self.road.network.get_lane(self.vehicle.lane_index)
                # 获取车辆在车道上的局部坐标和角度
                av_lon, _ = av_lane.local_coordinates(self.vehicle.position)
                front_veh_lon, _ = av_lane.local_coordinates(front_vehicle.position)
                av_angle = av_lane.local_angle(self.vehicle.heading, av_lon)
                front_veh_angle = av_lane.local_angle(front_vehicle.heading, front_veh_lon)

                # 计算前车速度在车辆朝向方向上的分量
                bv_speed = front_vehicle.speed * np.cos(front_veh_angle)
                # 计算自身车辆速度在朝向方向上的分量
                av_speed = self.vehicle.speed * np.cos(av_angle)

                if bv_speed < av_speed:  # 如果前车比自身车辆速度慢
                    # 计算前车与自身车辆在车道上的相对距离和速度差的比值
                    front_ttc = min(front_ttc, (front_veh_lon - av_lon) / utils.not_zero(av_speed - bv_speed))

            if front_ttc < 2:  # 如果前车与自身车辆有较短的时间到达
                # 根据前车与自身车辆的时间到达，计算奖励值
                reward = -math.exp(-front_ttc / 5) * self.config['adv_front_ttc_coefficient']
                if action in [4]:  # 如果采取的动作是停车
                    reward = 0.5
                if self.vehicle.crashed:  # 如果自身车辆发生碰撞
                    reward = self.config["collision_reward"]

                return reward  # 返回奖励值
            else:  # 如果前车与自身车辆的时间到达较大
                if self.vehicle.crashed:  # 如果自身车辆发生碰撞
                    reward = self.config["collision_reward"]

                return reward  # 返回奖励值

    def _rewards(self, action: int) -> Dict[Text, float]:
        """Multi-objective rewards, for cooperative agents."""
        return 0.0

    def _agent_rewards(self, action: int, vehicle: Vehicle) -> Dict[Text, float]:
        """Per-agent per-objective reward signal."""
        scaled_speed = utils.lmap(vehicle.speed, self.config["reward_speed_range"], [0, 1])
        return {
            "collision_reward": vehicle.crashed,
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
        }

    def _is_terminated(self) -> bool:
        return any(vehicle.crashed for vehicle in self.controlled_vehicles) \
               or all(self.has_arrived(vehicle) for vehicle in self.controlled_vehicles) \
               or (self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _agent_is_terminal(self, vehicle: Vehicle) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return (vehicle.crashed or self.has_arrived(vehicle))

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles(self.config["initial_vehicle_count"])

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = super().step(action)
        self._clear_vehicles()
        self._spawn_vehicle(spawn_probability=self.config["spawn_probability"])
        return obs, reward, terminated, truncated, info

    def _make_road(self) -> None:
        """
        Make an 4-way intersection.

        The horizontal road has the right of way. More precisely, the levels of priority are:
            - 3 for horizontal straight lanes and right-turns
            - 1 for vertical straight lanes and right-turns
            - 2 for horizontal left-turns
            - 0 for vertical left-turns

        The code for nodes in the road network is:
        (o:outer | i:inner + [r:right, l:left]) + (0:south | 1:west | 2:north | 3:east)

        :return: the intersection road
        """

        lane_width = AbstractLane.DEFAULT_WIDTH
        right_turn_radius = lane_width + 20  # [m}
        left_turn_radius = right_turn_radius + lane_width * 3  # [m}
        outer_distance = right_turn_radius + lane_width / 2
        access_length = 50 + 50  # [m]

        net = RoadNetwork()
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        for corner in range(4):
            angle = np.radians(90 * corner)
            is_horizontal = corner % 2
            priority = 3 if is_horizontal else 1
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            # Incoming

            start = rotation @ np.array([lane_width / 2, access_length + outer_distance])
            end = rotation @ np.array([lane_width / 2, outer_distance])
            net.add_lane("o" + str(corner), "ir" + str(corner),
                         StraightLane(start, end, line_types=[c, s], priority=priority, speed_limit=10))

            start = rotation @ np.array([lane_width * 3 / 2, access_length + outer_distance])
            end = rotation @ np.array([lane_width * 3 / 2, outer_distance])
            net.add_lane("o" + str(corner), "ir" + str(corner),
                         StraightLane(start, end, line_types=[n, c], priority=priority, speed_limit=10))

            # Right turn
            r_center = rotation @ (np.array([outer_distance, outer_distance]))
            net.add_lane("ir" + str(corner), "il" + str((corner - 1) % 4),
                         CircularLane(r_center, right_turn_radius, angle + np.radians(180), angle + np.radians(270),
                                      line_types=[n, n], priority=priority, speed_limit=10))

            net.add_lane("ir" + str(corner), "il" + str((corner - 1) % 4),
                         CircularLane(r_center, right_turn_radius - lane_width, angle + np.radians(180), angle + np.radians(270),
                                      line_types=[n, c], priority=priority, speed_limit=10))



            # Left turn
            l_center = rotation @ (np.array([-left_turn_radius + lane_width / 2, left_turn_radius - lane_width / 2]))
            net.add_lane("ir" + str(corner), "il" + str((corner + 1) % 4),
                         CircularLane(l_center, left_turn_radius, angle + np.radians(0), angle + np.radians(-90),
                                      clockwise=False, line_types=[n, n], priority=priority - 1, speed_limit=10))

            net.add_lane("ir" + str(corner), "il" + str((corner + 1) % 4),
                         CircularLane(l_center, left_turn_radius + lane_width, angle + np.radians(0),
                                      angle + np.radians(-90),
                                      clockwise=False, line_types=[n, n], priority=priority - 1, speed_limit=10))
            # Straight
            start = rotation @ np.array([lane_width / 2, outer_distance])
            end = rotation @ np.array([lane_width / 2, -outer_distance])
            net.add_lane("ir" + str(corner), "il" + str((corner + 2) % 4),
                         StraightLane(start, end, line_types=[n, n], priority=priority, speed_limit=10))

            start = rotation @ np.array([lane_width * 3 / 2, outer_distance])
            end = rotation @ np.array([lane_width * 3 / 2, -outer_distance])
            net.add_lane("ir" + str(corner), "il" + str((corner + 2) % 4),
                         StraightLane(start, end, line_types=[n, n], priority=priority, speed_limit=10))

            # Exit
            start = rotation @ np.flip([lane_width / 2, access_length + outer_distance], axis=0)
            end = rotation @ np.flip([lane_width / 2, outer_distance], axis=0)
            net.add_lane("il" + str((corner - 1) % 4), "o" + str((corner - 1) % 4),
                         StraightLane(end, start, line_types=[c, s], priority=priority, speed_limit=10))

            start = rotation @ np.flip([lane_width * 3 / 2, access_length + outer_distance], axis=0)
            end = rotation @ np.flip([lane_width * 3 / 2, outer_distance], axis=0)
            net.add_lane("il" + str((corner - 1) % 4), "o" + str((corner - 1) % 4),
                         StraightLane(end, start, line_types=[n, c], priority=priority, speed_limit=10))

        road = RegulatedRoad(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        self.road = road

    def _make_vehicles(self, n_vehicles: int = 10) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane

        :return: the ego-vehicle
        """
        # Configure vehicles
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle_type.DISTANCE_WANTED = 10  # Low jam distance
        vehicle_type.COMFORT_ACC_MAX = 6
        vehicle_type.COMFORT_ACC_MIN = -3
        n_vehicles = random.choice(range(50, 100))
        # Random vehicles
        simulation_steps = 10
        for t in range(n_vehicles - 1):
            self._spawn_vehicle(np.linspace(0, 80, n_vehicles)[t])
        for _ in range(simulation_steps):
            [(self.road.act(), self.road.step(1 / self.config["simulation_frequency"])) for _ in range(self.config["simulation_frequency"])]

        # Challenger vehicle
        self._spawn_vehicle(60, spawn_probability=1, go_straight=True, position_deviation=0.1, speed_deviation=0)

        # Controlled vehicles
        self.controlled_vehicles = []
        for ego_id in range(0, self.config["controlled_vehicles"]):
            lane_idx = self.np_random.choice(range(4), size=2, replace=False)
            lane_id = random.choice(range(2))
            ego_lane = self.road.network.get_lane(("ir{}".format(lane_idx[0]), "il{}".format(lane_idx[1]), lane_id))
            destination = "o" + str(lane_idx[1])
            ego_vehicle = self.action_type.vehicle_class(
                             self.road,
                             ego_lane.position(0, 0),
                             speed=random.choice(range(4, 10)),
                             heading=ego_lane.heading_at(0)
                             # heading=ego_lane.heading_at(60)
            )
            try:
                ego_vehicle.plan_route_to(destination)
                ego_vehicle.speed_index = ego_vehicle.speed_to_index(ego_lane.speed_limit)
                ego_vehicle.target_speed = ego_vehicle.index_to_speed(ego_vehicle.speed_index)
            except AttributeError:
                pass

            self.road.vehicles.append(ego_vehicle)
            self.controlled_vehicles.append(ego_vehicle)
            self.vehicle = ego_vehicle
            for v in self.road.vehicles:  # Prevent early collisions
                if v is not ego_vehicle and np.linalg.norm(v.position - ego_vehicle.position) < 10:
                    self.road.vehicles.remove(v)

    def _spawn_vehicle(self,
                       longitudinal: float = 0,
                       position_deviation: float = 1.,
                       speed_deviation: float = 1.,
                       spawn_probability: float = 0.6,
                       go_straight: bool = False) -> None:
        if self.np_random.uniform() > spawn_probability:
            return

        route = self.np_random.choice(range(4), size=2, replace=False)
        route[1] = (route[0] + 2) % 4 if go_straight else route[1]
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle = vehicle_type.make_on_lane(self.road, ("o" + str(route[0]), "ir" + str(route[0]), random.choice(range(2))),
                                            longitudinal=(longitudinal + 5
                                                          + self.np_random.normal() * position_deviation),
                                            speed=random.choice(range(1, 10)))
        for v in self.road.vehicles:
            if np.linalg.norm(v.position - vehicle.position) < 10:
                return
        vehicle.plan_route_to("o" + str(route[1]))
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)
        return vehicle

    def _clear_vehicles(self) -> None:
        is_leaving = lambda vehicle: "il" in vehicle.lane_index[0] and "o" in vehicle.lane_index[1] \
                                     and vehicle.lane.local_coordinates(vehicle.position)[0] \
                                     >= vehicle.lane.length - 4 * vehicle.LENGTH
        self.road.vehicles = [vehicle for vehicle in self.road.vehicles if
                              vehicle in self.controlled_vehicles or not (is_leaving(vehicle) or vehicle.route is None)]

    def has_arrived(self, vehicle: Vehicle, exit_distance: float = 25) -> bool:
        return "il" in vehicle.lane_index[0] \
               and "o" in vehicle.lane_index[1] \
               and vehicle.lane.local_coordinates(vehicle.position)[0] >= exit_distance

    def calculate_interaction_time_difference(self, av, bv):
        max_value = 1e5
        av_lane = self.road.network.get_lane(av.lane_index)
        bv_lane = self.road.network.get_lane(bv.lane_index)
        points = []
        if isinstance(av_lane, StraightLane) and isinstance(bv_lane, StraightLane):
            points = self.calculate_two_straight_intersection(av_lane, bv_lane)

        elif isinstance(av_lane, StraightLane) and isinstance(bv_lane, CircularLane):
            points = self.calculate_circle_with_straight_intersection(bv_lane, av_lane)

        elif isinstance(av_lane, CircularLane) and isinstance(bv_lane, StraightLane):
            points = self.calculate_circle_with_straight_intersection(av_lane, bv_lane)

        elif isinstance(av_lane, CircularLane) and isinstance(bv_lane, CircularLane):
            points = self.calculate_two_circle_intersection(av_lane, bv_lane)
        av_to_p_time = bv_to_p_time = 0
        av_to_p_lon_diff  = 0
        bv_to_av_pos_time = 10
        for p in points:
            if av_lane.on_lane(p) and bv_lane.on_lane(p):
                av_p_lon, av_p_lat = av_lane.local_coordinates(p)
                av_lon, av_lat = av_lane.local_coordinates(av.position)

                if av_p_lon < av_lon:
                    continue

                bv_p_lon, bv_p_lat = bv_lane.local_coordinates(p)
                bv_lon, bv_lat = bv_lane.local_coordinates(bv.position)

                if bv_p_lon < bv_lon:
                    continue

                av_angle = av_lane.local_angle(av.heading, av_lon)
                bv_angle = bv_lane.local_angle(bv.heading, bv_lon)

                bv_speed = bv.speed * np.cos(bv_angle)
                av_speed = av.speed * np.cos(av_angle)

                av_to_p_time_ = (av_p_lon - av_lon) / av_speed
                bv_to_p_time_ = (bv_p_lon - bv_lon) / bv_speed
                if max_value > abs(av_to_p_time - bv_to_p_time):
                    max_value = abs(av_to_p_time_ - bv_to_p_time_)
                    av_to_p_time = av_to_p_time_
                    bv_to_p_time = bv_to_p_time_
                    av_to_p_lon_diff = (av_p_lon - av_lon)
                    av_lon_at_bv_lane, _ = bv_lane.local_coordinates(av.position)
                    bv_to_av_pos_time = (av_lon_at_bv_lane - bv_lon) / bv_speed

        return max_value, av_to_p_time, bv_to_p_time, bv_to_av_pos_time, av_to_p_lon_diff

    def calculate_two_circle_intersection(self, circle_lane_1, circle_lane_2):
        x1, y1 = circle_lane_1.center[0], circle_lane_1.center[1]
        x2, y2 = circle_lane_2.center[0], circle_lane_2.center[1]
        r1, r2 = circle_lane_1.radius, circle_lane_2.radius
        dx, dy = x2 - x1, y2 - y1
        dis2 = dx ** 2 + dy ** 2
        if dis2 > (r1 + r2) ** 2 or dis2 < (r1 - r2) ** 2:
            return []

        t = math.atan2(dy, dx)
        a = math.acos((r1 ** 2 - r2 ** 2 + dis2) / (2 * r1 * math.sqrt(dis2)))

        x3, y3 = x1 + r1 * math.cos(t + a), y1 + r1 * math.sin(t + a)
        x4, y4 = x1 + r1 * math.cos(t - a), y1 + r1 * math.sin(t - a)
        if abs(a) < 1e-8:
            return [np.array([x3, y3])]
        else:
            return [np.array([x4, y4]), np.array([x3, y3])]

    def calculate_circle_with_straight_intersection(self, circle_lane, straight_lane):
        # 计算直线的斜率
        x1, y1 = straight_lane.start[0], straight_lane.start[1]
        x2, y2 = straight_lane.end[0], straight_lane.end[1]
        cx, cy = circle_lane.center[0], circle_lane.center[1]
        r = circle_lane.radius
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return []  # 直线是一个点，与圆没有交点

        # 垂直于x轴的情况
        if abs(dx) <= 1e-6:
            x = x1
            y_sq = r ** 2 - (x - cx) ** 2
            # 检查是否有交点
            if y_sq < 0:
                return []
            if y_sq == 0:
                return [(x, cy)]  # 直线与圆相切，返回一个交点
            else:
                y1 = cy + math.sqrt(y_sq)
                y2 = cy - math.sqrt(y_sq)
                return [(x, y1), (x, y2)]  # 直线与圆相交，返回两个交点

        slope = dy / dx
        # 计算直线的截距
        intercept = y1 - slope * x1

        # 根据直线和圆的方程计算交点
        a = 1 + slope ** 2
        b = (-2 * cx) + (2 * slope * intercept) - (2 * cy * slope)
        c = (cx ** 2) + (intercept ** 2) - (2 * cy * intercept) + (cy ** 2) - (r ** 2)

        discriminant = b ** 2 - 4 * a * c
        if abs(discriminant) < 1e-8:
            x = -b / (2 * a)
            y = slope * x + intercept
            return [np.array([x, y])]  # 直线与圆相切，返回一个交点
        elif discriminant < 0:
            return []  # 直线与圆没有交点
        else:
            x3 = (-b + math.sqrt(discriminant)) / (2 * a)
            y3 = slope * x3 + intercept

            x4 = (-b - math.sqrt(discriminant)) / (2 * a)
            y4 = slope * x4 + intercept

            return [np.array([x3, y3]), np.array([x4, y4])]  # 直线与圆相交，返回两个交点

    def calculate_two_straight_intersection(self, straight_lane_1, straight_lane_2):
        # 解方程求交点坐标
        x_diff = (straight_lane_1.start[0] - straight_lane_1.end[0], straight_lane_2.start[0] - straight_lane_2.end[0])
        y_diff = (straight_lane_1.start[1] - straight_lane_1.end[1], straight_lane_2.start[1] - straight_lane_2.end[1])
        det = x_diff[0] * y_diff[1] - x_diff[1] * y_diff[0]

        # 如果行列式 determinant = 0，则两条直线平行，无交点
        if det == 0:
            return []

        # 计算交点坐标
        det_inv = 1 / det
        d1 = (straight_lane_1.start[0] * straight_lane_1.end[1] - straight_lane_1.start[1] * straight_lane_1.end[0])
        d2 = (straight_lane_2.start[0] * straight_lane_2.end[1] - straight_lane_2.start[1] * straight_lane_2.end[0])
        x = (d1 * x_diff[1] - d2 * x_diff[0]) * det_inv
        y = (d1 * y_diff[1] - d2 * y_diff[0]) * det_inv
        return [np.array([x, y])]


class MultiAgentIntersectionEnv(IntersectionEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "action": {
                 "type": "MultiAgentAction",
                 "action_config": {
                     "type": "DiscreteMetaAction",
                     "lateral": False,
                     "longitudinal": True
                 }
            },
            "observation": {
                "type": "MultiAgentObservation",
                "observation_config": {
                    "type": "Kinematics"
                }
            },
            "controlled_vehicles": 2
        })
        return config

class ContinuousIntersectionEnv(IntersectionEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 5,
                "features": ["presence", "x", "y", "vx", "vy", "long_off", "lat_off", "ang_off"],
            },
            "action": {
                "type": "ContinuousAction",
                "steering_range": [-np.pi / 3, np.pi / 3],
                "longitudinal": True,
                "lateral": True,
                "dynamical": True
            },
        })
        return config

TupleMultiAgentIntersectionEnv = MultiAgentWrapper(MultiAgentIntersectionEnv)
