from __future__ import division, print_function, absolute_import

import os.path
from abc import ABC

from gym.envs.registration import register
import numpy as np
import cv2
from NGSIM_env import utils
from NGSIM_env.envs.common.observation import observation_factory
from NGSIM_env import utils
from NGSIM_env.envs.common.abstract import AbstractEnv
from NGSIM_env.road.road import Road, RoadNetwork
from NGSIM_env.vehicle.behavior import IDMVehicle
from NGSIM_env.vehicle.humandriving import HumanLikeVehicle, InterActionVehicle, Pedestrian, IntersectionHumanLikeVehicle
from NGSIM_env.road.lane import LineType, StraightLane, PolyLane, PolyLaneFixedWidth
from NGSIM_env.utils import *
import pickle
import lanelet2
from threading import Thread
from shapely.geometry import LineString, Point


def will_intersect_no_speed(x1, y1, theta1, x2, y2, theta2):
    # 将角度转换为弧度
    theta1 = math.radians(theta1)
    theta2 = math.radians(theta2)

    # 计算方向向量
    v1x = math.cos(theta1)
    v1y = math.sin(theta1)
    v2x = math.cos(theta2)
    v2y = math.sin(theta2)

    # 设置方程
    # x1 + t * v1x = x2 + s * v2x
    # y1 + t * v1y = y2 + s * v2y

    # 表示为
    # t * v1x - s * v2x = x2 - x1
    # t * v1y - s * v2y = y2 - y1

    # 系数矩阵
    A = [[v1x, -v2x],
         [v1y, -v2y]]
    B = [x2 - x1,
         y2 - y1]

    # 计算行列式
    det = A[0][0] * A[1][1] - A[0][1] * A[1][0]

    if det == 0:
        return False  # 平行或重合，不会相交

    # 计算t和s
    t = (B[0] * A[1][1] - B[1] * A[0][1]) / det
    s = (A[0][0] * B[1] - A[1][0] * B[0]) / det

    # 检查t和s是否为正
    if t >= 0 and s >= 0:
        return True

    return False


class InterActionEnv(AbstractEnv):
    """
    一个带有交互数据的十字路口驾驶环境，用于收集gail训练数据。
    """

    def __init__(self, path, vehicle_id, IDM=False, render=True, save_video=False):
        # 打开包含路径的文件
        f = open(path, 'rb')
        # 读取pickle文件中的轨迹数据
        self.trajectory_set = pickle.load(f)
        f.close()
        self.vehicle_id = vehicle_id
        # 获取自车的长度和宽度
        self.ego_length = self.trajectory_set['ego']['length']
        self.ego_width = self.trajectory_set['ego']['width']
        # 获取自车的轨迹数据和持续时间
        self.ego_trajectory = self.trajectory_set['ego']['trajectory']
        self.duration = len(self.ego_trajectory) - 1
        # 获取周围车辆的ID
        self.surrounding_vehicles = list(self.trajectory_set.keys())
        self.surrounding_vehicles.pop(0)
        self.run_step = 0
        self.human = False
        self.IDM = IDM
        self.reset_time = 0
        self.show = render
        self.save_video = save_video
        self.laneletmap = None  # 地图
        # 如果vehicle_id是字符串且包含'P'，则获取行人ID
        if isinstance(vehicle_id, str) and 'P' in vehicle_id:
            self.ped_ids = self.trajectory_set['ego']['ped_ids']

        super(InterActionEnv, self).__init__()

    def default_config(self):
        # 设置默认的配置
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                'see_behind': True,
                "features": ["x", 'y', "vx", 'vy', 'heading', 'w', 'h', 'vehicle_id'],
                "normalize": False,
                "absolute": True,
                "vehicles_count": 11},
            # "osm_path": "/home/sx/wjc/wjc/datasets/Interaction/INTERACTION-Dataset-TC-v1_0/maps/TC_BGR_Intersection_VA.osm",
            "osm_path": "/home/sx/wjc/wjc/datasets/Interaction/maps/DR_USA_Intersection_MA.osm",
            "vehicles_count": 10,
            "show_trajectories": False,
            "screen_width": 600,
            "screen_height": 600,
            'collection_feature_type': 'scenario'
        })

        return config

    def reset(self, human=False, reset_time=0, video_save_name=''):
        '''
        重置环境，指定是否使用人类目标和重置时间
        '''

        self.video_save_name = video_save_name
        self.human = human
        self.load_map()
        self._create_road()
        self._create_vehicles(reset_time)
        self.steps = 0
        self.reset_time = reset_time
        return super(InterActionEnv, self).reset()

    def load_map(self):
        # 加载地图
        if not hasattr(self, 'roads_dict'):
            projector = lanelet2.projection.UtmProjector(lanelet2.io.Origin(0.0, 0.0))
            laneletmap = lanelet2.io.load(self.config['osm_path'], projector)
            self.roads_dict, self.graph, self.laneletmap, self.indegree, self.outdegree = utils.load_lanelet_map(laneletmap)

    def _create_road(self):
        # 创建道路
        net = RoadNetwork()
        none = LineType.NONE
        for i, (k, road) in enumerate(self.roads_dict.items()):
            center = road['center']
            left = road['left']
            right = road['right']
            start_point = (int(road['center'][0][0]), int(road['center'][0][1]))
            end_point = (int(road['center'][-1][0]), int(road['center'][-1][1]))
            net.add_lane(f"{start_point}", f"{end_point}",
                         PolyLane(center, left, right, line_types=(none, none)))

        # 加载人行道
        pedestrian_marking_id = 0
        for k, v in self.laneletmap.items():
            ls = v['type']
            if ls["type"] == "pedestrian_marking":
                pedestrian_marking_id += 1
                ls_points = v['points']
                net.add_lane(f"P_{pedestrian_marking_id}_start", f"P_{pedestrian_marking_id}_end",
                             PolyLaneFixedWidth(ls_points, line_types=(none, none), width=5))

        self.road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"],
                         lanelet=self.laneletmap)

    def _create_vehicles(self, reset_time):
        """
        创建自车和NGSIM车辆，并将它们添加到道路上。
        """
        T = 100  # 设置T的值
        self.controlled_vehicles = []  # 初始化受控车辆列表
        whole_trajectory = self.ego_trajectory  # 获取整个轨迹
        ego_trajectory = np.array(whole_trajectory[reset_time:])  # 获取自车轨迹
        self.vehicle = IntersectionHumanLikeVehicle.create(self.road, self.vehicle_id, ego_trajectory[0][:2], self.ego_length,
                                               self.ego_width, ego_trajectory, acc=(ego_trajectory[1][2]-ego_trajectory[0][2]) * 10, velocity=ego_trajectory[0][2],
                                               heading=ego_trajectory[0][3], target_velocity=ego_trajectory[1][2], human=self.human, IDM=self.IDM)  # 创建自车实例
        target = self.road.network.get_closest_lane_index(position=ego_trajectory[-1][:2])  # 获取目标车道
        self.vehicle.plan_route_to(target[1])  # 规划车辆行驶路线
        self.vehicle.is_ego = True  # 标记车辆为自车
        self.road.vehicles.append(self.vehicle)  # 将车辆添加到道路上
        self.controlled_vehicles.append(self.vehicle)  # 将车辆添加到受控车辆列表

    def _create_bv_vehicles(self, reset_time, T, current_time):
        vehicles = []  # 初始化其他车辆列表
        for veh_id in self.surrounding_vehicles:  # 遍历周围车辆
            try:
                other_trajectory = np.array(self.trajectory_set[veh_id]['trajectory'][reset_time:])  # 获取其他车辆轨迹
                flag = ~(np.array(other_trajectory[current_time])).reshape(1, -1).any(axis=1)[0]  # 判断是否存在轨迹点
                if current_time == 0:  # 如果当前时间为0
                    pass
                else:
                    trajectory = np.array(self.trajectory_set[veh_id]['trajectory'][reset_time:])  # 获取当前时间轨迹点
                    if not flag and ~(np.array(trajectory[current_time-1])).reshape(1, -1).any(axis=1)[0]:  # 如果当前时间和上一时间存在轨迹点
                        flag = False
                    else:
                        flag = True

                if not flag:  # 如果不存在轨迹点
                    other_vehicle = IntersectionHumanLikeVehicle.create(self.road, veh_id, other_trajectory[current_time][:2],
                                                              self.trajectory_set[veh_id]['length'],
                                                              self.trajectory_set[veh_id]['width'],
                                                              other_trajectory, acc=0.0,
                                                              velocity=other_trajectory[current_time][2],
                                                              heading=other_trajectory[current_time][3],
                                                              human=self.human,
                                                              IDM=self.IDM,start_step=self.steps)  # 创建其他车辆实例
                    mask = np.where(np.sum(other_vehicle.ngsim_traj[:(T * 10), :2], axis=1) == 0, False, True)  # 过滤无效轨迹点
                    other_vehicle.planned_trajectory = other_vehicle.ngsim_traj[:(T * 10), :2][mask]  # 设置计划轨迹
                    other_vehicle.planned_speed = other_vehicle.ngsim_traj[:(T * 10), 2:3][mask]  # 设置计划速度
                    other_vehicle.planned_heading = other_vehicle.ngsim_traj[:(T * 10), 3:][mask]  # 设置计划方向

                    if other_vehicle.planned_trajectory.shape[0] <= 5:  # 如果计划轨迹点不足5个，则跳过
                        continue

                    other_target = self.road.network.get_closest_lane_index(
                        position=other_vehicle.planned_trajectory[-1])  # 获取其他车辆的目标车道
                    other_vehicle.plan_route_to(other_target[1])  # 规划其他车辆行驶路线
                    vehicles.append(other_vehicle)  # 将其他车辆添加到列表中

            except Exception as e:  # 捕获异常
                print("_create_bv_vehicles", e)
        else:
            if len(vehicles) > 0:
                for vh in self.road.vehicles:
                    vehicles.append(vh)  # 将其他车辆添加到列表中
                self.road.vehicles = vehicles  # 将道路上的车辆替换为新的车辆列表

    # def _create_vehicles(self, reset_time):
    #     """
    #     Create ego vehicle and NGSIM vehicles and add them on the road.
    #     """
    #     self.controlled_vehicles = []
    #     whole_trajectory = self.ego_trajectory
    #     ego_trajectory = np.array(whole_trajectory[reset_time:])
    #     # ego_acc = (whole_trajectory[reset_time][2] - whole_trajectory[reset_time - 1][2]) / 0.1
    #     if not self.vehicle_id.startswith('P'):
    #         self.vehicle = IntersectionHumanLikeVehicle.create(self.road, self.vehicle_id, ego_trajectory[0][:2], self.ego_length,
    #                                                self.ego_width, ego_trajectory, acc=(ego_trajectory[1][2]-ego_trajectory[0][2]) * 10, velocity=ego_trajectory[0][2],
    #                                                heading=ego_trajectory[0][3], target_velocity=ego_trajectory[1][2], human=self.human, IDM=self.IDM)
    #     else:
    #         self.vehicle = Pedestrian.create(self.road, self.vehicle_id, ego_trajectory[0][:2], self.ego_length * 2,
    #                                                self.ego_width * 2, ego_trajectory, acc=(ego_trajectory[1][2] - ego_trajectory[0][2]) * 10, velocity=ego_trajectory[0][2],
    #                                                heading=ego_trajectory[0][3], target_velocity=ego_trajectory[1][2], human=self.human, IDM=self.IDM)
    #     target = self.road.network.get_closest_lane_index(position=ego_trajectory[-1][:2]) #, heading=ego_trajectory[-1][3]
    #     self.vehicle.plan_route_to(target[1])
    #     self.vehicle.is_ego = True
    #     self.road.vehicles.append(self.vehicle)
    #     self.controlled_vehicles.append(self.vehicle)
    #
    # def _create_bv_vehicles(self, reset_time, T, current_time):
    #     vehicles = []
    #     for veh_id in self.surrounding_vehicles:
    #
    #         other_trajectory = np.array(self.trajectory_set[veh_id]['trajectory'][reset_time:])
    #         flag = ~(np.array(other_trajectory[current_time])).reshape(1, -1).any(axis=1)[0]
    #         if current_time == 0:
    #             pass
    #         else:
    #             trajectory = np.array(self.trajectory_set[veh_id]['trajectory'][reset_time:])
    #             if not flag and ~(np.array(trajectory[current_time-1])).reshape(1, -1).any(axis=1)[0]:
    #                 flag = False
    #             else:
    #                 flag = True
    #
    #         if not flag:
    #             # print("add vehicle of No.{} step.".format(current_time))
    #             other_vehicle = None
    #             if not isinstance(veh_id, str):
    #                 other_vehicle = IntersectionHumanLikeVehicle.create(self.road, veh_id, other_trajectory[current_time][:2],
    #                                                           self.trajectory_set[veh_id]['length'],
    #                                                           self.trajectory_set[veh_id]['width'],
    #                                                           other_trajectory, acc=(other_trajectory[1][2]- other_trajectory[0][2]) * 10,
    #                                                           velocity=other_trajectory[current_time][2],
    #                                                           heading=other_trajectory[current_time][3],
    #                                                           human=self.human,
    #                                                           IDM=False)
    #                 other_vehicle.planned_trajectory = other_vehicle.ngsim_traj[
    #                                                    :(T * 10), :2]
    #                 other_vehicle.planned_speed = other_vehicle.ngsim_traj[
    #                                               :(T * 10), 2:3]
    #                 other_vehicle.planned_heading = other_vehicle.ngsim_traj[
    #                                                 :(T * 10), 3:]
    #
    #                 zeros = np.where(~other_trajectory.any(axis=1))[0]
    #                 if len(zeros) == 0:
    #                     zeros = [0]
    #                 other_target = self.road.network.get_closest_lane_index(
    #                     position=other_trajectory[int(zeros[0] - 1)][:2])  # heading=other_trajectory[-1][3]
    #                 other_vehicle.plan_route_to(other_target[1])
    #                 vehicles.append(other_vehicle)
    #             elif veh_id.startswith('P'):
    #                 pass
    #                 # other_vehicle = Pedestrian.create(self.road, veh_id, other_trajectory[current_time][:2],
    #                 #                                           self.trajectory_set[veh_id]['length'] * 2,
    #                 #                                           self.trajectory_set[veh_id]['width'] * 2,
    #                 #                                           other_trajectory, acc=0.0,
    #                 #                                           velocity=other_trajectory[current_time][2],
    #                 #                                           heading=other_trajectory[current_time][3],
    #                 #                                           human=self.human,
    #                 #                                           IDM=self.IDM)
    #
    #     else:
    #         # print("road length:", len(self.road.vehicles))
    #         if len(vehicles) > 0:
    #             # print("before road length:", len(self.road.vehicles))
    #             for vh in self.road.vehicles:
    #                 vehicles.append(vh)
    #             self.road.vehicles = vehicles
    #             # print("road length:", len(self.road.vehicles))

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
            "TTC_THW": self.TTC_THW,
            "velocity": self.vehicle.velocity,
            "crashed": self.vehicle.crashed,
            'offroad': not self.vehicle.on_road,
            "action": action,
            "time": self.time
        }

        return obs, 0, terminal, info

    def _reward(self, action):
        return 0

    def save_videos(self, imgs: list):

        if self.save_video and len(imgs) > 0:
            if not os.path.exists('./videos'):
                os.mkdir('./videos')
            path = f"./videos/{self.video_save_name}.mp4"
            print(f"save video in {path}")
            t = Thread(target=img_2_video, args=(path, imgs, True))
            t.setDaemon(True)
            t.start()
            t.join()

    def statistic_data_step_before(self):
            pass

    def statistic_data_step_after(self):
        data = self.calculate_ttc_thw()
        self.TTC_THW.append(data)

    def _simulate(self, action):
        """
        执行计划轨迹的多个仿真步骤
        """
        trajectory_features = []  # 存储轨迹特征
        T = action[2] if action is not None else self.duration // 10  # 如果action不为空，则T等于action[2]，否则T等于100
        imgs = []  # 存储生成的图像
        self.TTC_THW = []  # 存储TTC和THW

        for i in range(int(T * self.SIMULATION_FREQUENCY) - 1):  # 执行 T * SIMULATION_FREQUENCY - 1 步仿真
            self._create_bv_vehicles(self.reset_time, T, i)  # 创建虚拟车辆
            if i == 0:  # 第一步
                if action is not None:  # 如果action不为空，表示采样的目标
                    self.vehicle.trajectory_planner(action[0], action[1], action[2])
                else:  # 如果action为空，表示人类的目标
                    self.vehicle.planned_trajectory = self.vehicle.ngsim_traj[
                                                      self.vehicle.sim_steps:(self.vehicle.sim_steps + T * 10), :2]
                    self.vehicle.planned_speed = self.vehicle.ngsim_traj[
                                                 self.vehicle.sim_steps:(self.vehicle.sim_steps + T * 10), 2:3]
                    self.vehicle.planned_heading = self.vehicle.ngsim_traj[
                                                   self.vehicle.sim_steps:(self.vehicle.sim_steps + T * 10), 3:]

                self.run_step = 1  # 运行的步数
            self.road.act(self.run_step)  # 执行道路行为
            self.road.step(1 / self.SIMULATION_FREQUENCY)  # 道路前进一个时间步长
            self.time += 1  # 时间步加1
            self.run_step += 1  # 运行步加1
            self.steps += 1  # 步数加1
            if self.config['collection_feature_type'] == 'gail':
                features = self.gail_features()  # 提取GAIL特征
                action = [self.vehicle.action['steering'], self.vehicle.action['acceleration']]  # 车辆动作
                features += action  # 将动作添加到特征中
            elif self.config['collection_feature_type'] == 'scenario':
                features = self.scenario_features()  # 提取场景特征
            else:
                raise ValueError("collection_feature_type must be 'gail' or 'scenario'")  # 抛出异常

            trajectory_features.append(features)  # 添加特征到轨迹特征中
            self.statistic_data_step_after()  # 统计数据
            self._automatic_rendering()  # 自动渲染

            if self.show:  # 如果显示图像
                imgs.append(self.render('rgb_array'))  # 将渲染的图片添加到imgs中

            self.calculate_ttc_thw()  # 计算TTC和THW

            if self.done or self._is_terminal():  # 判断是否终止
                break

            self._clear_vehicles()  # 清除车辆

        self.enable_auto_render = False  # 关闭自动渲染
        self.save_videos(imgs)  # 保存生成的图像
        return trajectory_features  # 返回轨迹特征数据

    def _is_terminal(self):
        """
        判断是否终止
        """
        return self.vehicle.crashed or self.time >= self.duration or self.time >= (
                    self.vehicle.planned_heading.shape[0] - 3) or not self.vehicle.on_road

    def _clear_vehicles(self) -> None:
        """
        清除车辆
        """
        # 判断车辆是否要离开仿真环境
        is_leaving = lambda vehicle: (self.run_step >= (
                    vehicle.planned_trajectory.shape[0] + vehicle.start_step - 1) and not vehicle.IDM)

        vehicles = []
        for vh in self.road.vehicles:
            try:
                if vh in self.controlled_vehicles or not is_leaving(vh):  # 如果车辆是受控车辆或不需要离开，则保留在环境中
                    vehicles.append(vh)
            except Exception as e:
                print(e)

        self.road.vehicles = vehicles  # 清除需要离开的车辆

        # self.road.vehicles = [vehicle for vehicle in self.road.vehicles if
        #                       vehicle in self.controlled_vehicles or not (is_leaving(vehicle) or vehicle.route is None)]

    # def _is_terminal(self):
    #     """
    #     The episode is over if the ego vehicle crashed or go off road or the time is out.
    #     """
    #     lane_index = self.road.network.get_closest_lane_index(self.vehicle.position, self.vehicle.heading)
    #     lane = self.road.network.get_lane(lane_index)
    #     longitudinal, lateral = lane.local_coordinates(self.vehicle.position)
    #     # flag = self.time >= self.duration or self.vehicle.crashed or self.vehicle.linear.local_coordinates(self.vehicle.position)[0] >= self.vehicle.linear.length
    #     flag = self.time >= self.duration or self.vehicle.crashed or (lane_index[1] not in self.outdegree and longitudinal >= lane.length - self.vehicle.LENGTH)
    #
    #     return flag
    #
    # def _clear_vehicles(self) -> None:
    #
    #     # is_leaving = lambda vehicle: (not vehicle.IDM and (
    #     #             self.run_step >= (vehicle.planned_trajectory.shape[0] - 1) or vehicle.next_position is None or ~
    #     #     np.array(vehicle.next_position).reshape(1, -1).any(axis=1)[0])) or \
    #     #                              (vehicle.IDM and vehicle.linear.local_coordinates(vehicle.position)[
    #     #                                  0] >= vehicle.linear.length - vehicle.LENGTH)
    #
    #     vehicles = []
    #     for vehicle in self.road.vehicles:
    #         try:
    #             lane_index = self.road.network.get_closest_lane_index(vehicle.position, vehicle.heading)
    #             lane = self.road.network.get_lane(lane_index)
    #             longitudinal, lateral = lane.local_coordinates(self.vehicle.position)
    #             flag = not self.vehicle.on_road or (not vehicle.IDM and (self.run_step >= (vehicle.planned_trajectory.shape[0] - 1) or vehicle.next_position is None or
    #                                          ~np.array(vehicle.next_position).reshape(1, -1).any(axis=1)[0])) or \
    #                                  (vehicle.IDM and (lane_index[1] not in self.outdegree and longitudinal >= lane.length - self.vehicle.LENGTH) or
    #                                   vehicle.linear is not None and vehicle.linear.local_coordinates(vehicle.position)[0] >= vehicle.linear.length)
    #
    #             if vehicle in self.controlled_vehicles or not flag:
    #                 vehicles.append(vehicle)
    #         except Exception as e:
    #             print(e)
    #         # else:
    #         #     print(vh.lane_index)
    #         #     print(vh.lane.local_coordinates(vh.position)[0])
    #         #     print(vh.lane.length)
    #         #     print(vh.LENGTH)
    #         #     print(vh.next_position)
    #         #     print(vh)
    #
    #     self.road.vehicles = vehicles
    #
    #     # self.road.vehicles = [vehicle for vehicle in self.road.vehicles if
    #     #                       vehicle in self.controlled_vehicles or not (is_leaving(vehicle) or vehicle.route is None)]

    def scenario_features(self):

        accelerations = []
        speeds = [0]
        ego_speed = self.vehicle.velocity
        ego_longitudinal, lateral_distance = self.vehicle.lane.local_coordinates(self.vehicle.position)
        ego_vehicle_lane_start_heading = self.vehicle.lane.heading_at(0)
        ego_vehicle_lane_end_heading = self.vehicle.lane.heading_at(self.vehicle.lane.length)
        for vehicle in self.road.vehicles:
            accelerations.append(vehicle.action_history[-1]['acceleration'])
            if vehicle is self.vehicle:
                continue

            speeds.append(vehicle.velocity - ego_speed)

        front_vehicle, _ = self.road.neighbour_vehicles(self.vehicle)  # 获取前车
        TTC, _ = self.calculate_ttc_thw()
        TTC_distance = (100 - TTC) / 100.

        k = self.vehicle.lane.curve.calc_curvature(ego_longitudinal)
        is_change_lane = False
        have_pedestrain = False

        for vehicle in self.road.vehicles:
            if isinstance(vehicle.vehicle_ID, str) and 'P' in vehicle.vehicle_ID:
                have_pedestrain = will_intersect_no_speed(self.vehicle.position[0], self.vehicle.position[1], self.vehicle.heading,
                                                          vehicle.position[0], vehicle.position[1], vehicle.heading)
                if have_pedestrain:
                    break

        front_vehicle, _ = self.road.neighbour_vehicles(self.vehicle)
        following_distance = 1.
        lateral_distance = (lateral_distance * 2) / self.vehicle.lane.width_at(ego_longitudinal)
        if front_vehicle is not None:
            front_longitudinal, _ = self.vehicle.lane.local_coordinates(front_vehicle.position)
            following_distance = 1. - (front_longitudinal - ego_longitudinal) / 100.

        is_inside_the_intersection = True
        is_go_straight = False
        is_turn_left = False
        is_turn_right = False
        is_turn_round = False
        if abs(ego_vehicle_lane_end_heading - ego_vehicle_lane_start_heading) < np.pi / 6:
            is_go_straight = True
        elif ego_vehicle_lane_start_heading - ego_vehicle_lane_end_heading < -np.pi / 4:
            is_turn_left = True
        elif ego_vehicle_lane_start_heading - ego_vehicle_lane_end_heading > np.pi / 4:
            is_turn_right = True

        ego_current_heading = self.vehicle.heading
        future_step = min(self.steps + 100, len(self.vehicle.planned_heading) - 1)
        future_heading = self.vehicle.planned_heading[future_step]
        if abs(ego_current_heading - future_heading) > (np.pi * 5) / 6:
            is_turn_round = True

        if len(self.vehicle.lane.lane_points) > 5:
            is_inside_the_intersection = True

        lane_change_future_step = min(self.steps + 10, len(self.vehicle.planned_heading) - 1)
        future_traj_points = self.vehicle.planned_trajectory[lane_change_future_step]
        lane_change_longitudinal, lane_change_lateral = self.vehicle.lane.local_coordinates(future_traj_points)

        if lane_change_lateral > self.vehicle.lane.width_at(lane_change_longitudinal) / 2:
            is_change_lane = True

        features = [len(self.road.vehicles), np.mean(accelerations), np.std(accelerations), np.mean(speeds), np.std(speeds),
                    k, self.vehicle.velocity, TTC_distance, following_distance, lateral_distance,
                    int(is_change_lane), int(have_pedestrain), int(is_inside_the_intersection), int(is_go_straight), int(is_turn_left),
                    int(is_turn_right), int(is_turn_round)]

        return features

    def gail_features(self):
        obs = self.observation.observe()  # 获取观测值
        lane_index = self.road.network.get_closest_lane_index(self.vehicle.position, self.vehicle.heading)  # 获取最近车道的索引
        lane = self.road.network.get_lane(lane_index)  # 获取最近车道
        longitudinal, lateral = lane.local_coordinates(self.vehicle.position)  # 获取车辆在车道中的局部坐标
        lane_w = lane.width_at(longitudinal)  # 获取车道在局部坐标处的宽度
        lane_offset = lateral  # 车辆在车道中的横向偏移
        lane_heading = lane.heading_at(longitudinal)  # 车道在局部坐标处的方向

        features = [lane_offset, lane_heading, lane_w]  # 特征列表，包括车道偏移、车道方向和车道宽度

        features += obs[0][2:5].tolist()  # 添加观测值中的一些特征
        for vb in obs[1:]:
            core = obs[0] - vb
            features += core[:5].tolist()
        # print(len(features), features)
        return features  # 返回特征列表

    def cal_angle(self, v1, v2):
        dx1, dy1 = v1
        dx2, dy2 = v2
        angle1 = np.arctan2(dy1, dx1)  # 计算向量v1与x轴之间的夹角
        angle1 = int(angle1 * 180 / np.pi)  # 弧度转为角度
        angle2 = np.arctan2(dy2, dx2)  # 计算向量v2与x轴之间的夹角
        angle2 = int(angle2 * 180 / np.pi)  # 弧度转为角度
        if angle1 * angle2 >= 0:
            included_angle = abs(angle1 - angle2)  # 计算两个夹角的差值
        else:
            included_angle = abs(angle1) + abs(angle2)  # 计算两个夹角的绝对值之和
            if included_angle > 180:
                included_angle = 360 - included_angle  # 如果夹角大于180度，则取其补角
        return included_angle  # 返回夹角值

    def calculate_ttc_thw(self):
        MAX_TTC = 100  # 最大TTC值
        MAX_THW = 100  # 最大THW值
        THWs = [100]  # THW列表
        TTCs = [100]  # TTC列表
        agent_dict = self.vehicle.to_dict()  # 将车辆转换为字典类型
        p_0 = (agent_dict['x'], agent_dict['y'])  # 获取车辆的位置坐标
        v_0 = (agent_dict['vx'], agent_dict['vy'])  # 获取车辆的速度向量
        th_0 = agent_dict['heading']  # 获取车辆的航向角

        for v in self.road.vehicles:  # 遍历所有车辆
            if v is self.vehicle:
                continue
            # count the angle of vo and the x-axis
            v_dict = v.to_dict()
            p_i = (v_dict['x'], v_dict['y'])
            v_i = (v_dict['vx'], v_dict['vy'])
            th_i = v_dict['heading']

            # Calculate determinant of the matrix formed by the direction vectors
            det = v_0[0] * v_i[1] - v_i[0] * v_0[1]  # 计算两个向量的叉乘
            if det == 0:
                # Vectors are parallel, there is no intersection
                # 采用平行情况的算法
                vector_p = np.array(
                    [p_0[0] - p_i[0], p_0[1] - p_i[1]])  # 计算两个点的坐标差
                vector_v = np.array([v_0[0], v_0[1]])  # 将速度向量转换为numpy数组
                angle = self.cal_angle(vector_p, vector_v)  # 计算夹角
                v = np.sqrt(vector_v[0] ** 2 + vector_v[1] ** 2)  # 计算速度向量的模长
                v_projection = v * np.cos(angle / 180 * np.pi)  # 计算速度在向量p和向量v之间的投影
                if v_projection < 0:
                    thw = np.sqrt(
                        vector_p[0] ** 2 + vector_p[1] ** 2) / v_projection  # 计算THW
                    TTCs.append(abs(thw))
                    THWs.append(abs(thw))
            else:
                # Calculate the parameter values for each vector
                t1 = (v_i[0] * (p_0[1] - p_i[1]) -
                      v_i[1] * (p_0[0] - p_i[0])) / det
                t2 = (v_0[0] * (p_0[1] - p_i[1]) -
                      v_0[1] * (p_0[0] - p_i[0])) / det

                # Calculate the intersection point
                x_cross = p_0[0] + v_0[0] * t1
                y_cross = p_0[1] + v_0[1] * t1

                p_c = (x_cross, y_cross)

                dis_to_x_0 = np.sqrt(
                    (p_c[0] - p_0[0]) ** 2 + (p_c[1] - p_0[1]) ** 2)
                v_project_0 = np.sqrt(
                    v_0[0] ** 2 + v_0[1] ** 2) * np.sign((p_c[0] - p_0[0]) * v_0[0] + (p_c[1] - p_0[1]) * v_0[1])

                dis_to_x_i = np.sqrt(
                    (p_c[0] - p_i[0]) ** 2 + (p_c[1] - p_i[1]) ** 2)
                v_project_i = np.sqrt(
                    v_i[0] ** 2 + v_i[1] ** 2) * np.sign((p_c[0] - p_i[0]) * v_i[0] + (p_c[1] - p_i[1]) * v_i[1])

                TTX_0 = dis_to_x_0 / v_project_0
                TTX_i = dis_to_x_i / v_project_i

                # 如果距离足够近，则进行thw的运算
                if max(TTX_0, TTX_i) < MAX_THW + 5 and min(TTX_0, TTX_i) > 0:
                    thw = (dis_to_x_0 - dis_to_x_i) / v_project_0

                    if thw > 0:
                        THWs.append(thw)

                TTX_0 = np.sqrt((p_c[0] - p_0[0]) ** 2 + (p_c[1] - p_0[1]) ** 2) / np.sqrt(
                    v_0[0] ** 2 + v_0[1] ** 2) * np.sign((p_c[0] - p_0[0]) * v_0[0] + (p_c[1] - p_0[1]) * v_0[1])
                TTX_i = np.sqrt((p_c[0] - p_i[0]) ** 2 + (p_c[1] - p_i[1]) ** 2) / np.sqrt(
                    v_i[0] ** 2 + v_i[1] ** 2) * np.sign((p_c[0] - p_i[0]) * v_i[0] + (p_c[1] - p_i[1]) * v_i[1])

                # 阈值取车身长度/最大速度
                delta_threshold = 5 / \
                                  max(np.sqrt(v_0[0] ** 2 + v_0[1] ** 2),
                                      np.sqrt(v_i[0] ** 2 + v_i[1] ** 2))

                if TTX_0 > 0 and TTX_i > 0:
                    if abs(TTX_0 - TTX_i) < delta_threshold:
                        TTCs.append(TTX_0)

        return min(TTCs), min(THWs)  # 返回最小的TTC和THW值