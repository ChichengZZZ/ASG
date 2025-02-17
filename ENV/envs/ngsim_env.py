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
        self.duration = len(self.ego_trajectory) // 2
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
            "screen_width": 800,
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
                                           self.ego_width, ngsim_traj=ego_trajectory, velocity=ego_trajectory[0][2],)
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

        # whole_trajectory = self.process_raw_trajectory(self.ego_trajectory)
        # ego_trajectory = whole_trajectory[reset_time:]
        # # target_speed = np.max(whole_trajectory[:, 3])
        # ego_acc = (whole_trajectory[reset_time][2] - whole_trajectory[reset_time-1][2]) / 0.1
        # self.vehicle = HumanLikeVehicle.create(self.road, self.vehicle_id, ego_trajectory[0][:2], self.ego_length, self.ego_width,
        #                                        ego_trajectory, acc=ego_acc, velocity=ego_trajectory[0][2], human=self.human, IDM=True)
        # self.vehicle.make_linear()
        # self.vehicle.is_ego = True
        # self.road.vehicles.append(self.vehicle)
        #
        # for veh_id in self.surrounding_vehicles:
        #     other_trajectory = self.process_raw_trajectory(self.trajectory_set[veh_id]['trajectory'])[reset_time:]
        #     other_acc = (other_trajectory[reset_time][2] - other_trajectory[reset_time - 1][2]) / 0.1
        #     # target_speed = np.max(other_trajectory[:, 3])
        #     # self.road.vehicles.append(NGSIMVehicle.create(self.road, veh_id, other_trajectory[0][:2], self.trajectory_set[veh_id]['length']/3.281,
        #     #                                               self.trajectory_set[veh_id]['width']/3.281, other_trajectory, velocity=other_trajectory[0][2]))
        #     other_vehicle = HumanLikeVehicle.create(self.road, veh_id, other_trajectory[0][:2],
        #                             self.trajectory_set[veh_id]['length'] / 3.281,
        #                             self.trajectory_set[veh_id]['width'] / 3.281, other_trajectory, acc=other_acc,
        #                             velocity=other_trajectory[0][2],
        #                             human=self.human, IDM=True)
        #     other_vehicle.make_linear()
        #     other_vehicle.color = (100, 200, 255)
        #     self.road.vehicles.append(other_vehicle)

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
        obs = self.observation.observe()
        terminal = self._is_terminal()

        info = {
            "features": features,
            "distance": self.distance,
            "TTC_THW": self.TTC_THW,
            "velocity": self.vehicle.velocity,
            "crashed": self.vehicle.crashed,
            'offroad': not self.vehicle.on_road,
            "action": action,
            "time": self.time
        }

        return obs, 0, terminal, info

    def save_video(self, imgs: list):

        if self.video_save_name != '':
            # if not os.path.exists('data/videos/highway'):
            #     os.makedirs('data/videos/highway')
            # if not os.path.exists(os.path):
            #     return
            path = f"{self.video_save_name}.mp4"
            print('save video in ', path)
            # t = Thread(target=utils.img_2_video, args=(path, imgs, True))
            # t.setDaemon(True)
            # t.start()
            # t.join()
            utils.img_2_video(path, imgs, True)

    def _simulate(self, action):
        """
        Perform several steps of simulation with the planned trajectory
        """
        self.TTC_THW = []  # 存储时距和跟车时距的列表
        self.distance = 0.0  # 距离初始位置的距离
        trajectory_features = []  # 存储轨迹特征的列表
        T = action[2] if action is not None else 20  # 如果 action 不为空，则取 action[2]；否则取 20
        # self.enable_auto_render = True
        imgs = []  # 存储视频帧的列表
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
                    # print(self.vehicle.planned_trajectory.shape)
                    # print(self.vehicle.planned_trajectory[0])
                    # self.vehicle.trajectory_planner(self.vehicle.ngsim_traj[self.vehicle.sim_steps+T*10][1],
                    #                                (self.vehicle.ngsim_traj[self.vehicle.sim_steps+T*10][0]-self.vehicle.ngsim_traj[self.vehicle.sim_steps+T*10-1][0])/0.1, T)
                    # print(self.vehicle.planned_trajectory.shape)
                    # print(self.vehicle.planned_trajectory[:5])
                self.run_step = 1  # 运行步数
                self.last_position = self.vehicle.position.copy()  # 上一位置

            self.road.act(self.run_step)  # 仿真环境进行仿真
            self.road.step(1 / self.SIMULATION_FREQUENCY)  # 仿真环境进行前进一步
            self.time += 1  # 仿真时间增加
            self.run_step += 1  # 运行步数增加
            # features = self._features()
            # trajectory_features.append(features)

            # print(self.vehicle.to_dict())
            # obs = self.observation.observe()
            # print(obs)
            self.nearest = self.road.close_vehicles_to(self.vehicle, 50, 10, sort=True)  # 获取车辆周围最近的车辆
            # print(self.vehicle)
            # print(self.nearest[0], self.nearest[0].heading)
            # if self.nearest:
            #     for v in self.nearest:
            #         # print(v.position)
            #         v.color = (255, 0, 0)

            self._automatic_rendering()  # 自动渲染

            for v in self.road.vehicles:
                if hasattr(v, 'color'):
                    delattr(v, 'color')
            if self.show:
                self.render()  # 渲染画面

            # print(self.vehicle.crashed)
            if self.video_save_name != '':
                imgs.append(self.render('rgb_array'))  # 保存视频帧

            # Stop at terminal states
            if self.done or self._is_terminal():  # 如果到达终止状态或者是终止动作
                break
            # features = self.gail_features_v3()
            features = self.gail_features()  # 提取 GAIL 特征
            action = [self.vehicle.action['steering'], self.vehicle.action['acceleration']]  # 获取车辆的操作
            # print(action)
            # print(features, action)
            features += action

            data = self.calculate_thw_and_ttc(self.vehicle)  # 计算车辆的时距和跟车时距
            self.TTC_THW.append(data)  # 存储时距和跟车时距
            if i > 0:
                self.distance += np.linalg.norm(self.vehicle.position - self.last_position)  # 计算车辆位置之间的距离
                self.last_position = self.vehicle.position.copy()  # 更新上一位置

            self._clear_vehicles()  # 清除仿真环境中的车辆
            # print(features)
            if self.time >= 1:
                trajectory_features.append(features)  # 存储轨迹特征

        self.enable_auto_render = False  # 关闭自动渲染
        self.save_video(imgs)  # 保存视频

        return trajectory_features  # 返回轨迹特征列表

    def gail_features(self):
        # 观察环境，得到当前的观测值
        obs = self.observation.observe()
        # 获取离车辆最近的车道索引
        lane_index = self.road.network.get_closest_lane_index(self.vehicle.position, self.vehicle.heading)
        # 通过车道索引获取车道
        lane = self.road.network.get_lane(lane_index)
        # 获取车辆在车道上的纵向和横向坐标
        longitudinal, lateral = lane.local_coordinates(self.vehicle.position)
        # 获取车道的宽度
        lane_w = lane.width_at(longitudinal)
        # 车道偏移量为横向坐标
        lane_offset = lateral
        # 车道朝向为车道上纵向坐标对应点的朝向
        lane_heading = lane.heading_at(longitudinal)

        features = [lane_offset, lane_heading, lane_w]

        # 从观测值中提取一部分特征
        features += obs[0][2:5].tolist()
        for vb in obs[1:]:
            # 计算观测值中的核心特征
            core = obs[0] - vb
            features += core[:5].tolist()
        # print(len(features), features)
        return features

    def calculate_thw_and_ttc(self, vehicle):
        # 初始化THW和TTC为100
        THWs = [100]
        TTCs = [100]
        for v in self.road.vehicles:
            if v is vehicle:
                continue

            if v.position[0] > vehicle.position[0] and abs(
                    v.position[1] - vehicle.position[1]) < vehicle.WIDTH and vehicle.velocity >= 1:
                v_speed = v.velocity * np.cos(v.heading)
                vehicle_speed = vehicle.velocity * np.cos(vehicle.heading)
                # 计算THW
                if vehicle_speed > 0:
                    THW = (v.position[0] - vehicle.position[0]) / utils.not_zero(vehicle_speed)
                else:
                    THW = 100

                # 计算TTC
                if v_speed > vehicle_speed:
                    TTC = 100
                else:
                    TTC = (v.position[0] - vehicle.position[0]) / utils.not_zero(vehicle_speed - v_speed)
                THWs.append(THW)
                TTCs.append(TTC)

        return [min(TTCs), min(THWs)]

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
