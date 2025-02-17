from __future__ import division, print_function, absolute_import

import matplotlib.pyplot as plt
from gym.envs.registration import register
import numpy as np
import cv2
import os
from NGSIM_env.envs.common.observation import observation_factory
from NGSIM_env.envs.common.abstract import AbstractEnv
from NGSIM_env.road.road import Road, RoadNetwork
from NGSIM_env.vehicle.behavior import IDMVehicle
from NGSIM_env.vehicle.humandriving import HumanLikeVehicle, NGSIMVehicle
from NGSIM_env.road.lane import LineType, StraightLane
import pickle
from threading import Thread
from NGSIM_env import utils
import math

class NGSIMEnv(AbstractEnv):
    """
    一个使用NGSIM数据的高速公路驾驶环境。
    """
    def __init__(self, scene, path, period, vehicle_id, IDM=False, show=False):
        """
        初始化方法，接收场景、路径、时间段、车辆ID等参数。
        :param scene: 场景名称
        :param path: NGSIM数据文件路径
        :param period: 时间段
        :param vehicle_id: 车辆ID
        :param IDM: 是否使用IDM模型
        :param show: 是否显示环境
        """
        f = open(path, 'rb')  # 打开NGSIM数据文件
        self.trajectory_set = pickle.load(f)  # 加载NGSIM数据
        f.close()
        self.vehicle_id = vehicle_id
        self.scene = scene
        self.ego_length = self.trajectory_set['ego']['length'] / 3.281
        self.ego_width = self.trajectory_set['ego']['width'] / 3.281
        self.ego_trajectory = self.trajectory_set['ego']['trajectory']
        self.duration = len(self.ego_trajectory) - 3
        self.surrounding_vehicles = list(self.trajectory_set.keys())
        self.surrounding_vehicles.pop(0)
        self.run_step = 0
        self.human = False
        self.IDM = IDM
        self.show = show
        super(NGSIMEnv, self).__init__()

    def process_raw_trajectory(self, trajectory):
        """
        处理原始轨迹，将坐标、速度进行转换。
        :param trajectory: 原始轨迹数据
        :return: 转换后的轨迹数据
        """
        trajectory = np.array(trajectory)
        for i in range(trajectory.shape[0]):
            if trajectory[i][0] == trajectory[i][1] == 0:
                continue

            x = trajectory[i][0] - 6
            y = trajectory[i][1]
            speed = trajectory[i][2]
            trajectory[i][0] = y / 3.281
            trajectory[i][1] = x / 3.281
            trajectory[i][2] = speed / 3.281

        return trajectory

    def default_config(self):
        """
        默认配置，设置观察类型、观察特征、标准化等参数。
        :return: 默认配置
        """
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                'see_behind': True,
                "features": ["x", 'y', "vx", 'vy', 'heading', 'w', 'h', 'vehicle_id'],
                "normalize": False,
                "absolute": True,
                "vehicles_count": 11},
            "vehicles_count": 10,
            "show_trajectories": False,
            "screen_width": 1200,
            "screen_height": 400,
        })

        return config

    def reset(self, human=False, reset_time=1, video_save_name=''):
        '''
        重置环境，在给定时间点及是否使用人类目标的情况下重置环境。
        :param human: 是否使用人类目标
        :param reset_time: 重置时间
        :param video_save_name: 视频保存名称
        :return: 环境状态
        '''
        self.video_save_name = video_save_name
        self.human = human
        self._create_road() # 创建公路
        self._create_vehicles(reset_time) # 创建车辆
        self.steps = 0

        return super(NGSIMEnv, self).reset()

    def _create_road(self):
        """
        创建由NGSIM道路网络组成的道路。
        """
        net = RoadNetwork()
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE

        if self.scene == 'us-101':
            length = 2150 / 3.281 # m
            width = 12 / 3.281 # m
            ends = [0, 560/3.281, (698+578+150)/3.281, length]

            # first section
            line_types = [[c, n], [s, n], [s, n], [s, n], [s, c]]
            for lane in range(5):
                origin = [ends[0], lane * width]
                end = [ends[1], lane * width]
                net.add_lane('s1', 's2', StraightLane(origin, end, width=width, line_types=line_types[lane]))

            # merge_in lanes
            net.add_lane('merge_in', 's2', StraightLane([480/3.281, 5.5*width], [ends[1], 5*width], width=width, line_types=[c, c], forbidden=True))

            # second section
            line_types = [[c, n], [s, n], [s, n], [s, n], [s, n], [s, c]]
            for lane in range(6):
                origin = [ends[1], lane * width]
                end = [ends[2], lane * width]
                net.add_lane('s2', 's3', StraightLane(origin, end, width=width, line_types=line_types[lane]))

            # third section
            line_types = [[c, n], [s, n], [s, n], [s, n], [s, c]]
            for lane in range(5):
                origin = [ends[2], lane * width]
                end = [ends[3], lane * width]
                net.add_lane('s3', 's4', StraightLane(origin, end, width=width, line_types=line_types[lane]))

            # merge_out lanes
            net.add_lane('s3', 'merge_out', StraightLane([ends[2], 5*width], [1550/3.281, 7*width], width=width, line_types=[c, c], forbidden=True))

            self.road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])

        elif self.scene == 'i-80':
            length = 1700 / 3.281
            lanes = 6
            width = 12 / 3.281
            ends = [0, 600/3.281, 700/3.281, 900/3.281, length]

            # first section
            line_types = [[c, n], [s, n], [s, n], [s, n], [s, n], [s, c]]
            for lane in range(lanes):
                origin = [ends[0], lane * width]
                end = [ends[1], lane * width]
                net.add_lane('s1', 's2', StraightLane(origin, end, width=width, line_types=line_types[lane]))

            # merge_in lanes
            net.add_lane('s1', 's2', StraightLane([380/3.281, 7.1*width], [ends[1], 6*width], width=width, line_types=[c, c], forbidden=True))

            # second section
            line_types = [[c, n], [s, n], [s, n], [s, n], [s, n], [s, n]]
            for lane in range(lanes):
                origin = [ends[1], lane * width]
                end = [ends[2], lane * width]
                net.add_lane('s2', 's3', StraightLane(origin, end, width=width, line_types=line_types[lane]))

            # merge_in lanes
            net.add_lane('s2', 's3', StraightLane([ends[1], 6*width], [ends[2], 6*width], width=width, line_types=[s, c]))

            # third section
            line_types = [[c, n], [s, n], [s, n], [s, n], [s, n], [s, n]]
            for lane in range(lanes):
                origin = [ends[2], lane * width]
                end = [ends[3], lane * width]
                net.add_lane('s3', 's4', StraightLane(origin, end, width=width, line_types=line_types[lane]))

            # merge_in lane
            net.add_lane('s3', 's4', StraightLane([ends[2], 6*width], [ends[3], 5*width], width=width, line_types=[n, c]))

            # forth section
            line_types = [[c, n], [s, n], [s, n], [s, n], [s, n], [s, c]]
            for lane in range(lanes):
                origin = [ends[3], lane * width]
                end = [ends[4], lane * width]
                net.add_lane('s4', 's5', StraightLane(origin, end, width=width, line_types=line_types[lane]))

            self.road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self, reset_time):
        """
        创建 Ego 车辆和 NGSIM 车辆，并将它们添加到道路上。
        """
        # 处理原始轨迹，得到完整的轨迹
        whole_trajectory = self.process_raw_trajectory(self.ego_trajectory)

        # 获取 ego 车辆的轨迹部分
        ego_trajectory = whole_trajectory[reset_time:]

        # 计算 ego 车辆的加速度
        ego_acc = (whole_trajectory[reset_time][2] - whole_trajectory[reset_time - 1][2]) / 0.1

        # 创建 ego 车辆
        self.vehicle = NGSIMVehicle.create(self.road, self.vehicle_id, ego_trajectory[0][:2], self.ego_length,
                                           self.ego_width,
                                           ngsim_traj=ego_trajectory, velocity=ego_trajectory[0][2],
                                           )
        # 标记为 ego 车辆
        self.vehicle.is_ego = True

        # 将车辆添加到道路上
        self.road.vehicles.append(self.vehicle)

        # 遍历周围车辆的 ID
        for veh_id in self.surrounding_vehicles:
            # 处理原始轨迹，得到完整的轨迹
            other_trajectory = self.process_raw_trajectory(self.trajectory_set[veh_id]['trajectory'])[reset_time:]

            # 创建其他车辆
            self.road.vehicles.append(NGSIMVehicle.create(self.road, veh_id, other_trajectory[0][:2],
                                                          self.trajectory_set[veh_id]['length'] / 3.281,
                                                          self.trajectory_set[veh_id]['width'] / 3.281,
                                                          ngsim_traj=other_trajectory, velocity=other_trajectory[0][2]))

        # print(self.road.vehicles)
        self.v_sum = len(self.road.vehicles)
        spd_sum = 0
        for v in self.road.vehicles:
            spd_sum += v.velocity
        self.spd_mean = spd_sum / self.v_sum

    def step(self, action=None):
        """
        Perform a MDP step
        """
        if self.road is None or self.vehicle is None:
            raise NotImplementedError("The road and vehicle must be initialized in the environment implementation")

        features = self._simulate(action)
        terminal = self._is_terminal()

        info = {
            'collection_features': features,
            "action": action,
        }

        return None, 0, terminal, info

    def _simulate(self, action):
        """
        Perform several steps of simulation with the planned trajectory
        """
        self.TTC_THW = []  # 存储时距和跟车时距的列表
        self.distance = 0.0  # 距离初始位置的距离
        T = action[2] if action is not None else 20  # 如果 action 不为空，则取 action[2]；否则取 20
        trajectory_features = []  # 轨迹特征列表
        ego_last_lane_index = self.vehicle.lane_index  # 车辆的上一车道索引
        future_index = 0
        find_lc_indx = False
        obj_front_vehicle_id = None
        obj_rear_vehicle_id = None
        front_vehicle_id = None
        for i in range(int(T * self.SIMULATION_FREQUENCY) - 1):  # 根据 T 和 SIMULATION_FREQUENCY 计算循环次数
            # print(i)
            if i == 0:  # 如果是第一次循环
                if action is not None:  # 如果 action 不为空
                    self.vehicle.trajectory_planner(action[0], action[1], action[2])  # 调用车辆的轨迹规划器
                else:  # 如果 action 为空
                    # print('self.vehicle.sim_steps', self.vehicle.sim_steps)
                    self.vehicle.planned_trajectory = self.vehicle.ngsim_traj[
                                                      self.vehicle.sim_steps:(self.vehicle.sim_steps + T * 10),
                                                      :2]  # 取出车辆的规划轨迹
                self.run_step = 1  # 运行步数
                self.last_position = self.vehicle.position.copy()  # 上一位置

            self.road.act(self.run_step)  # 仿真环境进行仿真
            self.road.step(1 / self.SIMULATION_FREQUENCY)  # 仿真环境进行前进一步
            self.time += 1  # 仿真时间增加
            self.run_step += 1  # 运行步数增加
            self._automatic_rendering()  # 自动渲染

            for v in self.road.vehicles:
                if hasattr(v, 'color'):
                    delattr(v, 'color')

            if self.show:
                self.render()  # 渲染画面

            # Stop at terminal states
            if self.done or self._is_terminal():  # 如果到达终止状态或者是终止动作
                break

            self._clear_vehicles()  # 清除仿真环境中的车辆
            if self.run_step > 70 and self.vehicle.lane_index[:2] == ego_last_lane_index[:2] and self.vehicle.lane_index[2] != ego_last_lane_index[2]:
                trajectory_features = self.get_features(ego_last_lane_index)  # 获取车辆的特征
                obj_front_vehicle_id = trajectory_features[-15]
                obj_rear_vehicle_id = trajectory_features[-10]
                front_vehicle_id = trajectory_features[-5]
                find_lc_indx = True

            ego_last_lane_index = self.vehicle.lane_index  # 更新车辆的上一车道索引

            if find_lc_indx:
                future_index += 1

                if future_index in [3, 5]:
                    trajectory_features.extend([self.vehicle.heading, np.inf, np.inf, np.inf])

                    for v in self.road.vehicles:
                        for j, veh_id in enumerate([obj_front_vehicle_id,
                                                    obj_rear_vehicle_id,
                                                    front_vehicle_id][::-1]):
                            if v.vehicle_ID == veh_id:
                                trajectory_features[-(j+1)] = v.heading

            if future_index >= 5:
                break

        self.enable_auto_render = False  # 关闭自动渲染

        return trajectory_features  # 返回轨迹特征列表

    def get_features(self, last_lane_index):

        last_lane = self.road.network.get_lane(last_lane_index)  # 获取上一车道
        current_lane = self.road.network.get_lane(self.vehicle.lane_index)  # 获取当前车道

        obj_front_vehicle, obj_rear_vehicle = self.road.neighbour_vehicles(self.vehicle)  # 获取车辆的前车和后车
        front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self.vehicle, lane_index=last_lane_index)  # 获取车辆的前车和后车

        if obj_front_vehicle is not None:
            if np.linalg.norm(obj_front_vehicle.position - self.vehicle.position) > 50:
                obj_front_vehicle = None

        if obj_rear_vehicle is not None:
            if np.linalg.norm(obj_rear_vehicle.position - self.vehicle.position) > 50:
                obj_rear_vehicle = None

        if front_vehicle is not None:
            if np.linalg.norm(front_vehicle.position - self.vehicle.position) > 50:
                front_vehicle = None

        lon, lat = last_lane.local_coordinates(self.vehicle.position)  # 获取车辆在当前车道的局部坐标
        # lat = last_lane.local_angle(self.vehicle.heading, lon)  # 获取车辆在上一车道的局部角度
        feature = [lat, self.vehicle.heading, self.vehicle.heading_history[-3], self.vehicle.heading_history[-5]]

        if obj_front_vehicle is not None:
            ttc = self.calculate_thw_and_ttc(obj_front_vehicle, self.vehicle)  # 计算 TTC
            ind_1 = min(3, len(obj_front_vehicle.heading_history))
            ind_2 = min(5, len(obj_front_vehicle.heading_history))
            feature.extend([obj_front_vehicle.vehicle_ID, ttc, obj_front_vehicle.heading, obj_front_vehicle.heading_history[-ind_1], obj_front_vehicle.heading_history[-ind_2]])
        else:
            feature.extend(['', 100.0, np.inf, np.inf, np.inf])

        if obj_rear_vehicle is not None:
            ttc = self.calculate_thw_and_ttc(self.vehicle, obj_rear_vehicle)  # 计算 TTC
            ind_1 = min(3, len(obj_rear_vehicle.heading_history))
            ind_2 = min(5, len(obj_rear_vehicle.heading_history))
            feature.extend([obj_rear_vehicle.vehicle_ID, ttc, obj_rear_vehicle.heading, obj_rear_vehicle.heading_history[-ind_1], obj_rear_vehicle.heading_history[-ind_2]])
        else:
            feature.extend(['', 100.0, np.inf, np.inf, np.inf])

        if front_vehicle is not None:
            ttc = self.calculate_thw_and_ttc(front_vehicle, self.vehicle)  # 计算 TTC
            ind_1 = min(3, len(front_vehicle.heading_history))
            ind_2 = min(5, len(front_vehicle.heading_history))
            feature.extend([front_vehicle.vehicle_ID, ttc, front_vehicle.heading, front_vehicle.heading_history[-ind_1], front_vehicle.heading_history[-ind_2]])
        else:
            feature.extend(['', 100.0, np.inf, np.inf, np.inf])

        return feature

    def calculate_thw_and_ttc(self, front_vehicle, rear_vehicle):

        TTC = 100
        if front_vehicle.position[0] > rear_vehicle.position[0] and abs(
                front_vehicle.position[1] - rear_vehicle.position[1]) < rear_vehicle.WIDTH and rear_vehicle.velocity >= 1:
            v_speed = front_vehicle.velocity * np.cos(front_vehicle.heading)
            vehicle_speed = rear_vehicle.velocity * np.cos(rear_vehicle.heading)

            # 计算TTC
            if v_speed > vehicle_speed:
                TTC = 100
            else:
                TTC = (front_vehicle.position[0] - rear_vehicle.position[0]) / utils.not_zero(vehicle_speed - v_speed)


        return TTC

    def _is_terminal(self):
        """
        The episode is over if the ego vehicle crashed or go off road or the time is out.
        """
        if self.vehicle.linear is not None and self.vehicle.IDM:
            # 计算车辆在车道上的位置和偏移
            s_v, lat_v = self.vehicle.linear.local_coordinates(self.vehicle.position)
            # 如果车辆坠毁或者车辆超过车道长度，则返回终止状态
            return self.vehicle.crashed or s_v >= self.vehicle.linear.length

        return self.vehicle.crashed or self.time >= self.duration-1 or self.vehicle.position[0] >= 2150/3.281 or not self.vehicle.on_road

    def _clear_vehicles(self) -> None:

        is_leaving = lambda vehicle: (not vehicle.IDM and (
                self.run_step >= (vehicle.planned_trajectory.shape[0] - 1) or vehicle.next_position is None or ~
        np.array(vehicle.next_position).reshape(1, -1).any(axis=1)[0])) or \
                                     (vehicle.IDM and vehicle.linear.local_coordinates(vehicle.position)[
                                         0] >= vehicle.linear.length - vehicle.LENGTH)

        vehicles = []
        for vehicle in self.road.vehicles:

            # lane_index = self.road.network.get_closest_lane_index(vehicle.position, vehicle.heading)
            # if lane_index in [('s3', 'merge_out', -1)]:
            #     lane = self.road.network.get_lane(lane_index)
            #     longitudinal, lateral = lane.local_coordinates(vehicle.position)
            #     if longitudinal >= lane.length - vehicle.LENGTH:
            #         pass

            if vehicle.linear is not None and vehicle.IDM:
                s_v, lat_v = vehicle.linear.local_coordinates(vehicle.position)
                if vehicle is self.vehicle or not s_v >= vehicle.linear.length - vehicle.LENGTH / 2:
                    vehicles.append(vehicle)
                    pass
                else:
                    vehicle.linear = None
                    vehicles.append(vehicle)
            else:
                if vehicle.position[0] >= 2150 / 3.281 or not self.vehicle.on_road:
                    pass
                else:
                    vehicles.append(vehicle)

        self.road.vehicles = vehicles
