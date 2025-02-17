from __future__ import division, print_function, absolute_import

from abc import ABC

from gym.envs.registration import register
import numpy as np
import cv2
from Utils.math import calculate_angle
from NGSIM_env import utils
from NGSIM_env.envs.common.observation import observation_factory
from NGSIM_env import utils
from NGSIM_env.envs.common.abstract import AbstractEnv
from NGSIM_env.road.road import Road, RoadNetwork
from NGSIM_env.vehicle.behavior import IDMVehicle
from NGSIM_env.vehicle.humandriving import HumanLikeVehicle, InterActionVehicle, IntersectionHumanLikeVehicle
from NGSIM_env.road.lane import LineType, StraightLane, PolyLane
from NGSIM_env.utils import *
import pickle
import lanelet2
from threading import Thread


class IntersectionAdEnv(AbstractEnv):
    """
    A Intersection driving environment with Interaction data for collecting gail training data.
    """

    def __init__(self, path, vehicle_id, IDM=False, render=True, adversarial_config=None):
        # f = open('NGSIM_env/data/trajectory_set.pickle', 'rb')
        f = open(path, 'rb')
        self.trajectory_set = pickle.load(f)
        f.close()
        self.vehicle_id = vehicle_id
        # self.trajectory_set = build_trajecotry(scene, period, vehicle_id)
        self.ego_length = self.trajectory_set['ego']['length']
        self.ego_width = self.trajectory_set['ego']['width']
        self.ego_trajectory = self.trajectory_set['ego']['trajectory']
        self.duration = len(self.ego_trajectory) - 3
        self.surrounding_vehicles = list(self.trajectory_set.keys())
        self.surrounding_vehicles.pop(0)
        self.run_step = 0
        self.human = False
        self.IDM = IDM
        self.reset_time = 0
        self.show = render
        self.gail = False
        self.adversarial_sconfig = adversarial_config if adversarial_config is not None else {'distance_limit_mi': 20,
                                                                                              'time_limit_steps': 10}
        super(IntersectionAdEnv, self).__init__()

    def default_config(self):
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                'see_behind': True,
                "features": ["x", 'y', "vx", 'vy', 'heading', 'vehicle_id'],
                "normalize": False,
                "absolute": True,
                "vehicles_count": 11},
            # "observation": {
            #     "type": "GrayscaleObservation",
            #     "observation_shape": (500, 500),
            #     "stack_size": 3,
            #     "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
            #     "scaling": 1.75,
            # },
            "vehicles_count": 10,
            "show_trajectories": False,
            "screen_width": 300,
            "screen_height": 200,
        })

        return config

    def reset(self, human=False, reset_time=0, video_save_name=''):
        '''
        Reset the environment at a given time (scene) and specify whether use human target
        '''

        self.video_save_name = video_save_name
        self.human = human
        self.load_map()
        self._create_road()
        self._create_vehicles(reset_time)
        self.steps = 0
        self.reset_time = reset_time
        return super(IntersectionAdEnv, self).reset()

    def load_map(self):
        """
        加载环境的地图
        """
        if not hasattr(self, 'roads_dict'):  # 如果没有加载过道路字典
            projector = lanelet2.projection.UtmProjector(lanelet2.io.Origin(0.0, 0.0))  # 设置坐标转换器
            laneletmap = lanelet2.io.load(self.config['osm_path'], projector)  # 加载地图
            self.roads_dict, self.graph, self.laneletmap, self.indegree, self.outdegree = utils.load_lanelet_map(
                laneletmap)  # 加载车道地图

    def _create_road(self):
        """
        创建环境的道路结构
        """
        net = RoadNetwork()  # 创建路网
        none = LineType.NONE  # 线类型为空
        for i, (k, road) in enumerate(self.roads_dict.items()):  # 遍历道路字典中的道路
            for j in range(len(road['center']) - 1):  # 遍历道路的每个中心点
                start_point = tuple(road['center'][j])  # 起始点
                end_point = tuple(road['center'][j + 1])  # 终点
                net.add_lane(f"{start_point}", f"{end_point}",  # 添加车道
                             StraightLane(road['center'][j], road['center'][j + 1], line_types=(none, none)))

        # 加载人行道
        pedestrian_marking_id = 0
        for k, v in self.laneletmap.items():
            ls = v['type']
            if ls["type"] == "pedestrian_marking":
                pedestrian_marking_id += 1
                ls_points = v['points']
                net.add_lane(f"P_{pedestrian_marking_id}_start", f"P_{pedestrian_marking_id}_end",
                             StraightLane(ls_points[0], ls_points[-1], line_types=(none, none), width=5))

        self.road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"],
                         lanelet=self.laneletmap)  # 创建道路对象

    def _create_vehicles(self, reset_time):
        """
        创建自车和NGSIM车辆，并将它们放入道路上。
        """
        self.controlled_vehicles = [] # 初始化一个空列表来存放受控车辆
        whole_trajectory = self.ego_trajectory
        ego_trajectory = np.array(whole_trajectory[reset_time:]) # 获取自车轨迹数据
        # ego_acc = (whole_trajectory[reset_time][2] - whole_trajectory[reset_time - 1][2]) / 0.1
        self.ego = IntersectionHumanLikeVehicle.create(self.road, self.vehicle_id, ego_trajectory[0][:2], self.ego_length,
                                               self.ego_width, ego_trajectory, acc=(ego_trajectory[1][2]-ego_trajectory[0][2]) * 10, velocity=ego_trajectory[0][2],
                                               heading=ego_trajectory[0][3], target_velocity=ego_trajectory[1][2], human=self.human, IDM=self.IDM)
        # 创建自车对象并设置相关属性
        target = self.road.network.get_closest_lane_index(position=ego_trajectory[-1][:2]) # 获取自车最终目标车道
        self.ego.plan_route_to(target[1]) # 为自车规划路径到目标车道
        self.road.vehicles.append(self.ego) # 将自车添加到道路上的车辆列表
        self.controlled_vehicles.append(self.ego) # 将自车添加到受控车辆列表
        self._create_bv_vehicles(self.reset_time, 10, self.steps) # 创建周围车辆，设置参数为重置时间、时间间隔、当前时间

    def _create_bv_vehicles(self, reset_time, T, current_time):
        vehicles = [] # 初始化一个空列表来存放周围车辆
        for veh_id in self.surrounding_vehicles: # 遍历周围车辆列表
            try:
                other_trajectory = np.array(self.trajectory_set[veh_id]['trajectory'][reset_time:]) # 获取周围车辆的轨迹数据
                flag = ~(np.array(other_trajectory[current_time])).reshape(1, -1).any(axis=1)[0] # 判断当前时刻是否有轨迹数据
                if current_time == 0:
                    pass
                else:
                    trajectory = np.array(self.trajectory_set[veh_id]['trajectory'][reset_time:])
                    if not flag and ~(np.array(trajectory[current_time-1])).reshape(1, -1).any(axis=1)[0]:
                        flag = False
                    else:
                        flag = True

                if not flag:
                    # print("add vehicle of No.{} step.".format(current_time))
                    other_vehicle = InterActionVehicle.create(self.road, veh_id, other_trajectory[current_time][:2],
                                                              self.trajectory_set[veh_id]['length'],
                                                              self.trajectory_set[veh_id]['width'],
                                                              other_trajectory, acc=0.0,
                                                              velocity=other_trajectory[current_time][2],
                                                              heading=other_trajectory[current_time][3],
                                                              human=self.human,
                                                              IDM=False) # 创建周围车辆对象并设置相关属性
                    other_vehicle.planned_trajectory = other_vehicle.ngsim_traj[:(T * 10), :2]
                    other_vehicle.planned_speed = other_vehicle.ngsim_traj[:(T * 10), 2:3]
                    other_vehicle.planned_heading = other_vehicle.ngsim_traj[:(T * 10), 3:]

                    zeros = np.where(~other_trajectory.any(axis=1))[0]
                    if len(zeros) == 0:
                        zeros = [0]
                    other_target = self.road.network.get_closest_lane_index(
                        position=other_trajectory[int(zeros[0] - 1)][:2])  # heading=other_trajectory[-1][3]
                    other_vehicle.plan_route_to(other_target[1]) # 为周围车辆规划路径到目标车道
                    vehicles.append(other_vehicle) # 将周围车辆添加到车辆列表

            except Exception as e:
                print("_create_bv_vehicles", e)
        else:
            if len(vehicles) > 0:
                for vh in self.road.vehicles:
                    vehicles.append(vh)
                self.road.vehicles = vehicles

        if not self.gail and current_time == 0:
            # 设置对抗函数的开关以及计时
            self.open_adversarial_function = False
            self.adversarial_time = 0
            self.no_adversarial_time = 0
            dis = 1e6
            self.vehicle = 0
            # 查找与自车距离在50米以内，速度差不超过5米/秒的邻车
            self.neighbours = self.road.close_vehicles_to(self.ego, 50, 5, sort=True)
            for v in self.neighbours:
                # 计算邻车和自车的速度矢量，并计算它们的夹角
                bv_vx = np.cos(v.heading) * v.velocity
                bv_vy = np.sin(v.heading) * v.velocity
                av_vx = np.cos(self.vehicle.heading) * self.vehicle.velocity
                av_vy = np.sin(self.vehicle.heading) * self.vehicle.velocity
                vector_a = [bv_vx, bv_vy]
                vector_b = [av_vx, av_vy]
                angle = calculate_angle(vector_a, vector_b)

                # 求最短距离和夹角符合条件的邻车作为对抗车辆
                tmp_dis = np.linalg.norm(v.position - self.vehicle.position)
                if tmp_dis < dis and angle < 30:
                    self.open_adversarial_function = True
                    dis = tmp_dis
                    self.vehicle = v
                    self.vehicle.color = (0, 0, 255)
            # 如果没有找到对抗车辆，则选择最近的邻车
            if not self.vehicle:
                try:
                    self.vehicle = self.neighbours[0]
                    self.vehicle.color = (0, 0, 255)
                except:
                    # 如果邻车范围内没有车辆，则扩大搜索范围
                    self.neighbours = self.road.close_vehicles_to(self.ego, 1000, 1, sort=True)
                    if len(self.neighbours) >= 1:
                        self.vehicle = self.neighbours[0]
                        self.vehicle.color = (0, 0, 255)
                    else:
                        # 如果还是没有车辆，则禁用对抗功能，使用当前车辆作为训练数据
                        self.gail = True
                        self.spd_mean = self.ego.velocity
                        if self.reset_time == 0:
                            self.vehicle = self.ego
                            self.vehicle.color = (200, 200, 0)
                        return

            # 如果没有对抗车辆或者对抗剩余步数小于等于0，则关闭对抗功能，将自车设置为训练数据
            if not self.vehicle or self.adversarial_sconfig['time_limit_steps'] <= 0:
                self.open_adversarial_function = False
                self.ego.IDM = True
                self.vehicle = self.neighbours[0]
                self.vehicle.color = (0, 0, 255)

            # 将选择的车辆设置为蓝色（对抗车辆）
            self.vehicle.color = (0, 0, 255)
            self.vehicle.is_ego = True
            # 计算自车与对抗车辆的初始距离
            self.l2_init = np.linalg.norm(self.ego.position - self.vehicle.position)

        # 将选定的车辆添加到controlled_vehicles列表中
        self.controlled_vehicles.append(self.vehicle)
        self.v_sum = len(self.road.vehicles)
        spd_sum = 0
        # 计算路上所有车辆的速度总和
        for v in self.road.vehicles:
            spd_sum += v.velocity
        # 计算平均速度
        self.spd_mean = spd_sum / self.v_sum

    def _reward_normal(self, action=None) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        # 将 ego 车速从 [5, 10] 映射到 [0, 1] 之间的值
        scaled_speed = utils.lmap(self.ego.velocity, [5, 10], [0, 1])
        # 根据车辆是否碰撞和缩放后的车速计算奖励
        reward = -10 * self.ego.crashed + 1 * np.clip(scaled_speed, 0, 1)
        # 根据时间和持续时间计算奖励
        reward += 1 * self.time / self.duration
        # 如果 ego 车不在道路上，则奖励为 -10
        reward = -10 if not self.ego.on_road else reward
        return reward

    def _reward_adv(self, action=None) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        reward = 0
        # 计算 ego 车和其他车辆之间的欧氏距离
        l2 = np.linalg.norm(self.vehicle.position - self.ego.position)
        # 根据初始距离和当前距离之差计算奖励
        rate = (self.l2_init - l2) / (self.l2_init)
        reward += np.clip(rate, -1, 1) - 0.5
        # 如果 ego车和其他车辆在垂直方向上的差距超过1个车道，则奖励为-1
        # if abs(self.ego.lane_index[2] - self.vehicle.lane_index[2]) > 1:
        #     reward += -1
        # 如果 ego 车和其他车辆都碰撞，则奖励为1；如果只有 ego 车碰撞，则奖励为-1
        if self.ego.crashed:
            if self.vehicle.crashed:
                reward += 1
            else:
                reward += -1
        # 如果 ego 车不在道路上，则奖励为 -1
        reward = -1 if not self.ego.on_road else reward
        return reward

    def step(self, action=None):
        """
        Perform a MDP step
        """
        if self.road is None or self.vehicle is None:
            raise NotImplementedError("The road and vehicle must be initialized in the environment implementation")

        (gail_features, adv_feature) = self._simulate(action)
        obs = self.observation.observe()
        terminal = self._is_terminal()
        reward = self._reward_adv(action)
        info = {
            "TTC_THW": self.TTC_THW,
            "distance": self.distance,
            "ego_distance": self.ego_distance,
            "velocity": self.ego.velocity,
            "crashed": self.ego.crashed,
            'offroad': not self.ego.on_road,
            "action": action,
            "time": self.time
        }

        return (gail_features, adv_feature), reward, terminal, info

    def _reward(self, action):
        return 0

    def save_video(self, imgs: list):
        """
        保存视频方法，接收一个图片列表作为参数
        """
        if self.video_save_name != '' and len(imgs) > 0:
            path = f"data/videos/intersection/{self.video_save_name}.mp4"
            print(f"save video in {path}")
            t = Thread(target=img_2_video, args=(path, imgs))
            t.setDaemon(True)
            t.start()
            t.join()

    def statistic_data_step_before(self):
        """
        统计数据方法，在每一步执行前调用
        """
        if self.steps == 0:
            try:
                self.vehicle_position = self.ego.position.copy()
            except:
                pass
            if not self.gail:
                self.ego_position = self.vehicle.position.copy()

    def statistic_data_step_after(self):
        """
        统计数据方法，在每一步执行后调用
        """
        if self.open_adversarial_function or self.adversarial_sconfig["time_limit_steps"] == 0:
            data = self.calculate_ttc_thw(self.ego)
            data.extend(self.calculate_ttc_thw(self.vehicle))
            self.TTC_THW.append(data)
            if self.steps >= 1:
                self.distance = np.linalg.norm(self.vehicle_position - self.ego.position)
                self.ego_distance = np.linalg.norm(self.ego_position - self.vehicle.position)
        else:
            self.ego_distance = 0
            self.distance = 0
            self.TTC_THW.append([100, 100, 100, 100])

        if self.ego is not None:
            self.vehicle_position = self.ego.position.copy()

        self.ego_position = self.vehicle.position.copy()

    def _simulate(self, action):
        """
        执行一系列仿真步骤，接收一个动作作为参数

        参数:
            action: 动作

        返回值:
            轨迹特征
        """
        self.distance = 0.0  # 初始距离为0
        self.ego_distance = 0.0  # 初始EGO车辆的距离为0
        trajectory_features = []  # 轨迹特征列表
        self.TTC_THW = []  # 时间到碰撞和时间头车车距离列表
        for i in range(1):  # 循环一次
            if self.run_step > 0:  # 如果当前运行步数大于0
                self._create_bv_vehicles(self.reset_time, 10, self.run_step)  # 创建背景车辆

            self.statistic_data_step_before()  # 记录统计数据之前的步骤
            self.road.act(self.run_step)  # 执行道路行为
            if self.open_adversarial_function and self.adversarial_sconfig[
                'time_limit_steps'] != 0:  # 如果开启对抗模式，并且设置的时间步限制不为0
                self.ego.action['steering'] = action[0]  # 设置EGO车辆的转向动作
                self.ego.action['acceleration'] = action[1]  # 设置EGO车辆的加速度动作
                self.ego.color = (0, 255, 0)  # 设置EGO车辆的颜色为绿色

            self.road.step(1 / self.SIMULATION_FREQUENCY)  # 道路进行一步仿真
            self.time += 1  # 时间增加1
            self.run_step += 1  # 运行步数增加1
            self.steps += 1  # 步数增加1
            gail_features = self.gail_features()  # 获取GAIL特征
            if not self.gail:  # 如果没有GAIL功能
                adv_feature = self.adv_features()  # 获取对抗特征

            self.statistic_data_step_after()  # 记录统计数据之后的步骤
            action = [self.vehicle.action['steering'], self.vehicle.action['acceleration']]  # 获取车辆的动作
            # print(features, action)
            # features += action
            # trajectory_features.append(features)
            # features = self._features()
            # trajectory_features.append(features)

            # print(self.vehicle.to_dict())
            # obs = self.observation.observe()
            # print(obs)
            # self.nearest = self.road.close_vehicles_to(self.vehicle, 50, 10, sort=True)
            # print(self.vehicle)
            # print(self.nearest[0], self.nearest[0].heading)
            # if self.nearest:
            #     for v in self.nearest:
            #         # print(v.position)
            #         v.color = (255, 0, 0)

            self._automatic_rendering()  # 自动渲染

            # for v in self.road.vehicles:
            #     if hasattr(v, 'color'):
            #         delattr(v, 'color')
            # print(self.vehicle.crashed)
            # Stop at terminal states
            self._clear_vehicles()  # 清除离开道路的车辆
            # self._create_bv_vehicles(self.reset_time, T, i+1)

            self.change_adversarial_vehicle()  # 改变对抗车辆

            if self.done or self._is_terminal():  # 如果到达终止状态
                break  # 跳出循环

        self.enable_auto_render = False  # 关闭自动渲染

        # human_likeness = features[-1]
        # interaction = np.max([feature[-2] for feature in trajectory_features])
        # trajectory_features = np.sum(trajectory_features, axis=0)
        # trajectory_features[-1] = human_likeness
        return (gail_features, adv_feature)  # 返回GAIL特征和对抗特征

    def _is_terminal(self):
        """
        判断仿真是否终止

        返回值:
            True表示仿真终止，False表示仿真未终止
        """
        # return self.vehicle.crashed or self.time >= self.duration or not self.vehicle.on_road
        # print(type(self.vehicle))
        return self.time >= self.duration or self.vehicle.velocity == 0 or len(
            self.road.vehicles) < 2 or self.run_step >= (self.vehicle.ngsim_traj.shape[
                                                             0] - self.reset_time - 1) or self.vehicle.crashed or not self.vehicle.on_road

    def _clear_vehicles(self) -> None:
        """
        清除离开道路或即将离开道路的车辆
        """
        is_leaving = lambda vehicle: self.run_step >= (vehicle.planned_trajectory.shape[0] - 1) or \
                                     vehicle.lane.local_coordinates(vehicle.position)[0] \
                                     >= vehicle.lane.length - vehicle.LENGTH or (
                                                 not vehicle.IDM and (vehicle.next_position is None or ~
                                             np.array(vehicle.next_position).reshape(1, -1).any(axis=1)[0]))

        vehicles = []  # 存储要保留的车辆
        for vh in self.road.vehicles:
            try:
                if vh in self.controlled_vehicles or not is_leaving(vh):  # 如果车辆是受控车辆或不是离开车辆
                    vehicles.append(vh)  # 将车辆添加到列表中
            except Exception as e:
                print(e)
            # else:
            #     print(vh.lane_index)
            #     print(vh.lane.local_coordinates(vh.position)[0])
            #     print(vh.lane.length)
            #     print(vh.LENGTH)
            #     print(vh.next_position)
            #     print(vh)

        self.road.vehicles = vehicles  # 更新道路的车辆列表


        # self.road.vehicles = [vehicle for vehicle in self.road.vehicles if
        #                       vehicle in self.controlled_vehicles or not (is_leaving(vehicle) or vehicle.route is None)]

    def change_adversarial_vehicle(self):
        """
            0.基于强化学习的控制模型每次控制对抗NPCn秒，n=2。
            1.如果被测AV与对抗NPC距离超过D米，那么需要将Agent替换为AV距离最近的背景车辆。D=10
            2.如果选择失败，NPC则由IDM控制
        """
        if self.open_adversarial_function:
            self.adversarial_time += 1
        else:
            self.no_adversarial_time += 1

        dis = float(np.linalg.norm(self.vehicle.position - self.ego.position))
        vehicle = None
        # if self.ego.crashed or dis > 20 or self.adversarial_time >= 10:
        if self.ego.crashed or (not self.ego.on_road) or (
                self.adversarial_time >= self.adversarial_sconfig['time_limit_steps']) or \
                (dis >= self.adversarial_sconfig['distance_limit_mi']):
            self.open_adversarial_function = False
            self.adversarial_time = 0
            dis = 1e6
            neighbours = self.road.close_vehicles_to(self.vehicle, self.adversarial_sconfig['distance_limit_mi'], 5,
                                                     sort=True)
            lane = self.vehicle.lane_index
            # 当满足条件对周围车辆进行搜索

            if self.ego.crashed or not self.ego.on_road:
                self.ego.color = (255, 100, 100)
                self.ego.velocity = 0
                self.ego.crashed = True

            for v in neighbours:
                if v.crashed or not v.on_road:
                    continue
                bv_vx = np.cos(v.heading) * v.velocity
                bv_vy = np.sin(v.heading) * v.velocity
                av_vx = np.cos(self.vehicle.heading) * self.vehicle.velocity
                av_vy = np.sin(self.vehicle.heading) * self.vehicle.velocity
                vector_a = [bv_vx, bv_vy]
                vector_b = [av_vx, av_vy]
                angle = calculate_angle(vector_a, vector_b)
                tmp_dis = np.linalg.norm(v.position - self.vehicle.position)

                if tmp_dis < dis and angle < 30:
                    dis = tmp_dis
                    vehicle = v

            if vehicle is self.ego:
                if self.no_adversarial_time >= self.adversarial_sconfig['time_limit_steps'] and \
                        (float(np.linalg.norm(self.vehicle.position - self.ego.position)) < self.adversarial_sconfig[
                            'distance_limit_mi']):
                    self.ego.color = (0, 255, 0)
                    self.adversarial_time = 0
                    self.no_adversarial_time = 0
                    self.open_adversarial_function = True
                else:
                    self.ego.color = (200, 200, 0)
                    self.adversarial_time = self.adversarial_sconfig['time_limit_steps']
                return

            # 搜索到可用于对抗的背景车
            if vehicle is not None:
                # 如果原始agent没有碰撞，则换成背景车
                if not self.ego.crashed:
                    # bv = NGSIMVehicle.create(self.road, self.ego.vehicle_ID, self.ego.position, self.ego.LENGTH,
                    #                               self.ego.WIDTH, self.ego.ngsim_traj, self.ego.heading, self.ego.velocity)
                    # bv.sim_steps = self.ego.sim_steps
                    # bv.target_lane_index = self.ego.target_lane_index
                    # # self.road.vehicles = [v for v in self.road.vehicles if v is not self.ego]
                    # self.road.vehicles.remove(self.ego)
                    self.ego.IDM = True
                    self.ego.color = (200, 200, 0)
                    self.ego = None
                    # if hasattr(bv, 'color'):
                    #     delattr(bv, 'color')
                    # self.road.vehicles.append(bv)

                # adversarial_agent = HumanLikeVehicle.create(self.road, vehicle.vehicle_ID, vehicle.position, vehicle.LENGTH,
                #                               vehicle.WIDTH, vehicle.ngsim_traj, vehicle.heading, vehicle.velocity)
                # adversarial_agent.sim_steps = vehicle.sim_steps
                # adversarial_agent.target_lane_index = vehicle.target_lane_index

                # self.road.vehicles = [v for v in self.road.vehicles if v is not vehicle]
                # try:
                #     self.road.vehicles.remove(vehicle)
                # except:
                #     pass
                # self.ego = adversarial_agent
                # self.road.vehicles.append(self.ego)
                # self.open_adversarial_function = True
                vehicle.control_by_agent = True
                self.ego = vehicle
                self.open_adversarial_function = True
                self.ego.color = (0, 255, 0)

            # 如果没有搜索到，且 agent 未发生碰撞，将agent替换成非对抗状态
            elif not self.ego.crashed and self.ego.on_road:
                # vehicle = NGSIMVehicle.create(self.road, self.ego.vehicle_ID, self.ego.position, self.ego.LENGTH,
                #                               self.ego.WIDTH, self.ego.ngsim_traj, self.ego.heading, self.ego.velocity)
                # vehicle.sim_steps = self.ego.sim_steps
                # vehicle.target_lane_index = self.ego.target_lane_index
                # # self.road.vehicles = [v for v in self.road.vehicles if v is not self.ego]
                # try:
                #     self.road.vehicles.remove(self.ego)
                # except:
                #     pass

                # self.ego = vehicle
                self.ego.color = (200, 200, 0)
                # self.road.vehicles.append(self.ego)
                self.adversarial_time = self.adversarial_sconfig['time_limit_steps']
            else:  # 发生碰撞
                neighbours = self.road.close_vehicles_to(self.vehicle, 100, 1, sort=True)
                # adversarial_agent = HumanLikeVehicle.create(self.road, neighbours[0].vehicle_ID, neighbours[0].position,
                #                                             neighbours[0].LENGTH,
                #                                             neighbours[0].WIDTH, neighbours[0].ngsim_traj, neighbours[0].heading,
                #                                             neighbours[0].velocity)
                # adversarial_agent.sim_steps = neighbours[0].sim_steps
                # adversarial_agent.target_lane_index = neighbours[0].target_lane_index
                # # self.road.vehicles = [v for v in self.road.vehicles if v is not neighbours[0]]
                # try:
                #     self.road.vehicles.remove(neighbours[0])
                # except:
                #     pass
                # self.ego = adversarial_agent
                # self.ego.color = (144, 238, 144)
                # self.road.vehicles.append(self.ego)
                # self.adversarial_time = 10
                if len(neighbours) > 0:
                    neighbours[0].control_by_agent = True # IDM 控制
                    self.ego = neighbours[0]
                    self.ego.color = (200, 200, 0)
                    self.adversarial_time = self.adversarial_sconfig['time_limit_steps']
                else:
                    self.done = True
                    return

        if self.adversarial_sconfig['time_limit_steps'] <= 0:
            self.ego.color = (200, 200, 0)
            self.open_adversarial_function = False

    def sampling_space(self):
        """
        The target sampling space (longitudinal speed and lateral offset)
        """
        lane_center = self.ego.lane.start[1]
        current_y = self.ego.position[1]
        current_speed = self.ego.velocity
        lateral_offsets = np.array([lane_center - 12 / 3.281, current_y, lane_center + 12 / 3.281])
        min_speed = current_speed - 5 if current_speed > 5 else 0
        max_speed = current_speed + 5
        target_speeds = np.linspace(min_speed, max_speed, 10)

        return lateral_offsets, target_speeds

    def gail_features(self):
        obs = self.observation.observe()
        lane_index = self.road.network.get_closest_lane_index(self.ego.position, self.ego.heading)
        lane = self.road.network.get_lane(lane_index)
        longitudinal, lateral = lane.local_coordinates(self.ego.position)
        lane_w = lane.width_at(longitudinal)
        lane_offset = lateral
        lane_heading = lane.heading_at(longitudinal)

        features = [lane_offset, lane_heading, lane_w]
        features += obs[0][2:5].tolist()
        for vb in obs[1:]:
            core = obs[0]-vb
            features += core[:5].tolist()
        # print(len(features), features)
        return features

    def adv_features(self):
        lane_index = self.road.network.get_closest_lane_index(self.ego.position, self.ego.heading)
        lane = self.road.network.get_lane(lane_index)
        longitudinal, lateral = lane.local_coordinates(self.ego.position)
        lane_w = lane.width_at(longitudinal)
        lane_offset = lateral
        lane_heading = lane.heading_at(longitudinal)
        # vehicle_fea = obs[0][:5].tolist()
        features = [lane_heading, lane_w, lane_offset, self.ego.heading]
        ego_fea = [self.ego.to_dict()['x'] - self.vehicle.to_dict()['x'], self.ego.to_dict()['y'] - self.vehicle.to_dict()['y'], self.ego.to_dict()['vx'] - self.vehicle.to_dict()['vx'],
                   self.ego.to_dict()['vy'] -self.vehicle.to_dict()['vy'], self.vehicle.to_dict()['heading']]
        features.extend(ego_fea)

        # exp 2
        # for i in range(5):
        #     fea = obs[i][:5].tolist()
        #     features.extend(fea)

        adv_features = np.array(features)
        # print(adv_features.shape)
        return adv_features

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