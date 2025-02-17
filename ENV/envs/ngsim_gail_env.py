from __future__ import division, print_function, absolute_import

import math

from gym.envs.registration import register
import numpy as np

from NGSIM_env import utils
from NGSIM_env.envs.common.observation import observation_factory
from NGSIM_env import utils
from NGSIM_env.envs.common.abstract import AbstractEnv
from NGSIM_env.road.road import Road, RoadNetwork
from NGSIM_env.vehicle.behavior import IDMVehicle
from NGSIM_env.vehicle.humandriving import HumanLikeVehicle, NGSIMVehicle
from NGSIM_env.vehicle.control import MDPVehicle
from NGSIM_env.road.lane import LineType, StraightLane
from NGSIM_env.utils import *
import pickle
import os


class NGSIMGAILEnv(AbstractEnv):
    """
    A highway driving environment with NGSIM data.
    """

    def __init__(self, scene, path, period, vehicle_id, IDM=False, gail=False, action_type='continuous'):
        # f = open('NGSIM_env/data/trajectory_set.pickle', 'rb')
        # 初始化函数，接受参数scene, path, period, vehicle_id, IDM, gail, action_type
        self.path = path
        f = open(self.path, 'rb')  # 打开路径path对应的文件，并以二进制读取模式打开
        self.trajectory_set = pickle.load(f)  # 从文件中加载数据，并赋值给self.trajectory_set
        f.close()  # 关闭文件
        self.vehicle_id = vehicle_id  # 车辆id
        self.scene = scene  # 场景
        self.ego_length = self.trajectory_set['ego']['length'] / 3.281  # 获取自车长度
        self.ego_width = self.trajectory_set['ego']['width'] / 3.281  # 获取自车宽度
        self.ego_trajectory = self.trajectory_set['ego']['trajectory']  # 获取自车轨迹
        self.duration = 200  # 获取轨迹时长
        self.surrounding_vehicles = list(self.trajectory_set.keys())  # 获取周围车辆信息
        self.surrounding_vehicles.pop(0)  # 弹出列表的第一个元素
        self.run_step = 0  # 运行步数
        self.human = False  # 人类标识
        self.IDM = IDM  # IDM标识
        self.gail = gail  # GAIL标识
        self.kind_action = action_type  # 动作类型
        self.traffic_flow_action = None  # 交通流动作
        super(NGSIMGAILEnv, self).__init__()  # 调用父类的初始化函数

    def t_init(self, path):
        self.path = path  # 路径赋值给path
        f = open(self.path, 'rb')  # 打开路径path对应的文件，并以二进制读取模式打开
        self.trajectory_set = pickle.load(f)  # 从文件中加载数据，并赋值给self.trajectory_set
        f.close()  # 关闭文件
        self.ego_length = self.trajectory_set['ego']['length'] / 3.281  # 获取自车长度
        self.ego_width = self.trajectory_set['ego']['width'] / 3.281  # 获取自车宽度
        self.ego_trajectory = self.trajectory_set['ego']['trajectory']  # 获取自车轨迹
        self.duration = len(self.ego_trajectory) - 3  # 获取轨迹时长
        self.surrounding_vehicles = list(self.trajectory_set.keys())  # 获取周围车辆信息
        self.surrounding_vehicles.pop(0)  # 弹出列表的第一个元素

    def process_raw_trajectory(self, trajectory):
        trajectory = np.array(trajectory)  # 将轨迹转化为numpy数组
        for i in range(trajectory.shape[0]):  # 遍历轨迹数组
            x = trajectory[i][0] - 6  # x坐标减去6
            y = trajectory[i][1]  # y坐标
            speed = trajectory[i][2]  # 速度
            trajectory[i][0] = y / 3.281  # 更新x坐标
            trajectory[i][1] = x / 3.281  # 更新y坐标
            trajectory[i][2] = speed / 3.281  # 更新速度

        return trajectory  # 返回处理后的轨迹数组

    def default_config(self):
        config = super().default_config()  # 调用父类的default_config方法并将返回值赋值给config
        config.update({  # 更新配置
            "observation": {
                "type": "Kinematics",
                'see_behind': True,
                "features": ["x", 'y', "vx", 'vy', 'heading', 'w', 'h', 'vehicle_id'],
                "normalize": False,
                "absolute": True,
                "vehicles_count": 11
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "policy_frequency": 1,
            "vehicles_count": 10,
            "show_trajectories": True,
            "screen_width": 300,
            "screen_height": 200,
            "simulation_frequency": 10,
            "bv_collision_reward": -10,  # The reward received when colliding with a bv vehicle.
            "ego_collision_reward": 10,  # The reward received when colliding with ego vehicle.
        })

        return config  # 返回修改后的配置

    def reset(self, human=False, reset_time=1):
        '''
        Reset the environment at a given time (scene) and specify whether use human target
        '''
        self.human = human  # 标识是否为人类目标
        self._create_road()  # 创建道路
        self._create_vehicles(reset_time)  # 创建车辆
        self.steps = 0  # 步数计数器
        self.sum_ttc_reward = 0.0  # ttc奖励之和

        return super(NGSIMGAILEnv, self).reset()  # 调用父类的reset方法

    def _create_road(self):
        """
        Create a road composed of NGSIM road network
        """
        # 创建道路网络对象
        net = RoadNetwork()
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE

        if self.scene == 'us-101':
            length = 2150 / 3.281  # m
            width = 12 / 3.281  # m
            ends = [0, 560 / 3.281, (698 + 578 + 150) / 3.281, length]

            # first section
            line_types = [[c, n], [s, n], [s, n], [s, n], [s, c]]
            for lane in range(5):
                origin = [ends[0], lane * width]
                end = [ends[1], lane * width]
                # 向道路网络中添加车道
                net.add_lane('s1', 's2', StraightLane(origin, end, width=width, line_types=line_types[lane]))

            # merge_in lanes
            net.add_lane('merge_in', 's2',
                         StraightLane([480 / 3.281, 5.5 * width], [ends[1], 5 * width], width=width, line_types=[c, c],
                                      forbidden=True))

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
            net.add_lane('s3', 'merge_out',
                         StraightLane([ends[2], 5 * width], [1550 / 3.281, 7 * width], width=width, line_types=[c, c],
                                      forbidden=True))

            self.road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])

        elif self.scene == 'i-80':
            length = 1700 / 3.281
            lanes = 6
            width = 12 / 3.281
            ends = [0, 600 / 3.281, 700 / 3.281, 900 / 3.281, length]

            # first section
            line_types = [[c, n], [s, n], [s, n], [s, n], [s, n], [s, c]]
            for lane in range(lanes):
                origin = [ends[0], lane * width]
                end = [ends[1], lane * width]
                net.add_lane('s1', 's2', StraightLane(origin, end, width=width, line_types=line_types[lane]))

            # merge_in lanes
            net.add_lane('s1', 's2',
                         StraightLane([380 / 3.281, 7.1 * width], [ends[1], 6 * width], width=width, line_types=[c, c],
                                      forbidden=True))

            # second section
            line_types = [[c, n], [s, n], [s, n], [s, n], [s, n], [s, n]]
            for lane in range(lanes):
                origin = [ends[1], lane * width]
                end = [ends[2], lane * width]
                net.add_lane('s2', 's3', StraightLane(origin, end, width=width, line_types=line_types[lane]))

            # merge_in lanes
            net.add_lane('s2', 's3',
                         StraightLane([ends[1], 6 * width], [ends[2], 6 * width], width=width, line_types=[s, c]))

            # third section
            line_types = [[c, n], [s, n], [s, n], [s, n], [s, n], [s, n]]
            for lane in range(lanes):
                origin = [ends[2], lane * width]
                end = [ends[3], lane * width]
                net.add_lane('s3', 's4', StraightLane(origin, end, width=width, line_types=line_types[lane]))

            # merge_in lane
            net.add_lane('s3', 's4',
                         StraightLane([ends[2], 6 * width], [ends[3], 5 * width], width=width, line_types=[n, c]))

            # forth section
            line_types = [[c, n], [s, n], [s, n], [s, n], [s, n], [s, c]]
            for lane in range(lanes):
                origin = [ends[3], lane * width]
                end = [ends[4], lane * width]
                net.add_lane('s4', 's5', StraightLane(origin, end, width=width, line_types=line_types[lane]))

            self.road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])

    def change_adversarial_vehicle(self):
        """
            0. 基于强化学习的控制模型每次控制对抗NPC最多n秒，n=2。
            1. 如果被测AV与对抗NPC距离超过D米，那么需要将Agent替换为AV距离最近的背景车辆。D=10
            2. 如果选择失败，NPC则由IDM控制
        """
        if self.open_adversarial_function:  # 检查是否开启对抗NPC功能
            self.adversarial_time += 1
        dis = float(np.linalg.norm(self.vehicle.position - self.ego.position))  # 计算被测AV与对抗NPC之间的距离

        if not self.gail and (self.vehicle.crashed or dis > 20 or self.adversarial_time >= 10):  # 检查是否需要更换对抗NPC
            self.open_adversarial_function = False
            self.adversarial_time = 0
            dis = 1e6  # 设置一个较大的初始距离值
            neighbours = self.road.close_vehicles_to(self.ego, 20, 5, sort=True)  # 获取邻近的车辆
            lane = self.ego.lane_index
            vehicle = None
            for v in neighbours:
                if abs(v.lane_index[2] - lane[2]) < 2:  # 车辆在相同或相邻车道上
                    tmp_dis = np.linalg.norm(v.position - self.ego.position)  # 计算当前车辆与被测AV之间的距离
                    if tmp_dis < dis and (not v.crashed):
                        self.open_adversarial_function = True  # 标记开启对抗NPC功能
                        dis = tmp_dis  # 更新最小距离
                        vehicle = v  # 获取与被测AV距离最近的背景车辆
            if vehicle is not None:
                # 先根据agent创建bv车辆
                if not self.vehicle.crashed:  # 如果被测AV未碰撞
                    bv = NGSIMVehicle.create(self.road, self.vehicle.vehicle_ID, self.vehicle.position,
                                             self.vehicle.LENGTH,
                                             self.vehicle.WIDTH, self.vehicle.ngsim_traj, self.vehicle.heading,
                                             self.vehicle.velocity)
                    bv.sim_steps = self.vehicle.sim_steps
                    bv.target_lane_index = self.vehicle.target_lane_index
                    self.road.vehicles = [v for v in self.road.vehicles if v is not self.vehicle]
                    self.vehicle = None
                    self.road.vehicles.append(bv)  # 将被测AV替换为背景车辆

                adversarial_agent = HumanLikeVehicle.create(self.road, vehicle.vehicle_ID, vehicle.position,
                                                            vehicle.LENGTH,
                                                            vehicle.WIDTH, vehicle.ngsim_traj, vehicle.heading,
                                                            vehicle.velocity)  # 根据距离最近的背景车辆创建对抗NPC
                adversarial_agent.sim_steps = vehicle.sim_steps
                adversarial_agent.target_lane_index = vehicle.target_lane_index

                self.road.vehicles = [v for v in self.road.vehicles if v is not vehicle]
                self.vehicle = adversarial_agent
                self.road.vehicles.append(self.vehicle)  # 将对抗NPC添加到道路车辆列表

            elif not self.vehicle.crashed:  # 如果没有找到距离最近的背景车辆
                vehicle = NGSIMVehicle.create(self.road, self.vehicle.vehicle_ID, self.vehicle.position,
                                              self.vehicle.LENGTH,
                                              self.vehicle.WIDTH, self.vehicle.ngsim_traj, self.vehicle.heading,
                                              self.vehicle.velocity)
                vehicle.sim_steps = self.vehicle.sim_steps
                vehicle.target_lane_index = self.vehicle.target_lane_index
                self.road.vehicles = [v for v in self.road.vehicles if v is not self.vehicle]
                self.vehicle = vehicle
                self.vehicle.color = (144, 238, 144)  # 将车辆颜色设置为绿色
                self.road.vehicles.append(self.vehicle)  # 将背景车辆添加到道路车辆列表
            else:  # 如果没有找到背景车辆，在附近搜索背景车辆并创建对抗NPC
                neighbours = self.road.close_vehicles_to(self.ego, 100, 1, sort=True)
                adversarial_agent = HumanLikeVehicle.create(self.road, neighbours[0].vehicle_ID, neighbours[0].position,
                                                            neighbours[0].LENGTH,
                                                            neighbours[0].WIDTH, neighbours[0].ngsim_traj,
                                                            neighbours[0].heading,
                                                            neighbours[0].velocity)
                adversarial_agent.sim_steps = neighbours[0].sim_steps
                adversarial_agent.target_lane_index = neighbours[0].target_lane_index
                self.road.vehicles = [v for v in self.road.vehicles if v is not neighbours[0]]
                self.vehicle = adversarial_agent
                self.road.vehicles.append(self.vehicle)  # 将对抗NPC添加到道路车辆列表

    def _create_vehicles(self, reset_time):
        """
        Create ego vehicle and NGSIM vehicles and add them on the road.
        """
        # 处理ego车辆轨迹
        whole_trajectory = self.process_raw_trajectory(self.ego_trajectory)
        ego_trajectory = whole_trajectory[reset_time:]
        ego_acc = (whole_trajectory[reset_time][2] - whole_trajectory[reset_time - 1][2]) / 0.1
        if self.kind_action == 'continuous':
            # 创建HumanLikeVehicle作为ego车辆
            self.vehicle = HumanLikeVehicle.create(self.road, self.vehicle_id, ego_trajectory[0][:2], self.ego_length,
                                                   self.ego_width,
                                                   ego_trajectory, acc=ego_acc, velocity=ego_trajectory[0][2],
                                                   human=self.human, IDM=self.IDM)
            # self.vehicle = NGSIMVehicle.create(self.road, self.vehicle_id, ego_trajectory[0][:2], self.ego_length,
            #                                        self.ego_width,
            #                                        ngsim_traj=ego_trajectory, velocity=ego_trajectory[0][2],
            #                                        )
            self.vehicle.is_ego = True
            self.road.vehicles.append(self.vehicle)
        elif self.kind_action == 'discrete':
            # 创建MDPVehicle作为ego车辆
            self.vehicle = MDPVehicle.create(self.road, self.vehicle_id, ego_trajectory[0][:2], self.ego_length, self.ego_width,
                                                   ego_trajectory, velocity=ego_trajectory[0][2])
            self.vehicle.is_ego = True
            self.road.vehicles.append(self.vehicle)
        else:
            raise 'Action Type Not Found {}'.format(self.kind_action)

        for veh_id in self.surrounding_vehicles:
            # 处理其他NGSIM车辆的轨迹
            other_trajectory = self.process_raw_trajectory(self.trajectory_set[veh_id]['trajectory'])[reset_time:]
            # 创建NGSIMVehicle作为其他车辆
            self.road.vehicles.append(NGSIMVehicle.create(self.road, veh_id, other_trajectory[0][:2],
                                                          self.trajectory_set[veh_id]['length'] / 3.281,
                                                          self.trajectory_set[veh_id]['width'] / 3.281,
                                                          ngsim_traj=other_trajectory, velocity=other_trajectory[0][2]))
        self.ego = None
        # 新方法，运算速度较慢
        # whole_trajectory = self.process_raw_trajectory(self.ego_trajectory)
        # ego_trajectory = whole_trajectory[reset_time:]
        # # target_speed = np.max(whole_trajectory[:, 3])
        # ego_acc = (whole_trajectory[reset_time][2] - whole_trajectory[reset_time - 1][2]) / 0.1
        # self.vehicle = HumanLikeVehicle.create(self.road, self.vehicle_id, ego_trajectory[0][:2], self.ego_length,
        #                                        self.ego_width,
        #                                        ego_trajectory, acc=ego_acc, velocity=ego_trajectory[0][2],
        #                                        human=self.human, IDM=True)
        # self.vehicle.make_linear()
        # self.vehicle.is_ego = True
        # self.road.vehicles.append(self.vehicle)
        #
        # for veh_id in self.surrounding_vehicles:
        #     other_trajectory = self.process_raw_trajectory(self.trajectory_set[veh_id]['trajectory'])[reset_time:]
        #     other_acc = (other_trajectory[reset_time][2] - other_trajectory[reset_time - 1][2]) / 0.1
        #     other_vehicle = HumanLikeVehicle.create(self.road, veh_id, other_trajectory[0][:2],
        #                                             self.trajectory_set[veh_id]['length'] / 3.281,
        #                                             self.trajectory_set[veh_id]['width'] / 3.281, other_trajectory,
        #                                             acc=other_acc,
        #                                             velocity=other_trajectory[0][2],
        #                                             human=self.human, IDM=True)
        #     other_vehicle.make_linear()
        #     other_vehicle.color = (100, 200, 255)
        #     self.road.vehicles.append(other_vehicle)

        # self.vehicle.color = (255, 0, 0)
        # 更换为对抗整个交通流
        if not self.gail:
            self.open_adversarial_function = False
            self.adversarial_time = 0
            lane = self.vehicle.lane_index
            dis = 1e6
            self.ego = 0
            # 获取离ego车辆最近的其他车辆列表
            self.neighbours = self.road.close_vehicles_to(self.vehicle, 50, 5, sort=True)
            # print(self.neighbours)
            # 遍历邻近车辆，找到距离ego车辆最近且在同一车道上的车辆
            for v in self.neighbours:
                # print(v.lane_index[2], lane[2])
                if abs(v.lane_index[2] - lane[2]) < 2:
                    tmp_dis = np.linalg.norm(v.position - self.vehicle.position)
                    # print(tmp_dis)
                    if tmp_dis < dis and self.vehicle.position[0] > v.position[0]:
                        self.open_adversarial_function = True
                        dis = tmp_dis
                        self.ego = v
                        self.ego.color = (0, 0, 255)

            if not self.ego:

                self.ego = self.neighbours[0]
                self.ego.color = (0, 0, 255)

            self.ego.controlled_by_model = True
            # 被测车由交通流模型控制，step不执行
            # self.ego.color = (0, 0, 255)
            # try:
            #     vehicle = MDPVehicle.create(self.road, self.ego.vehicle_ID, self.ego.position, self.ego.LENGTH, self.ego.WIDTH,
            #                                        self.ego.ngsim_traj.copy(), velocity=self.ego.velocity)
            #     vehicles = []
            #     for v in self.road.vehicles:
            #         if v is self.ego:
            #             vehicle.color = (0, 0, 255)
            #             vehicle.controlled_by_model = True
            #             vehicles.append(vehicle)
            #             self.ego = vehicle
            #         else:
            #             vehicles.append(v)
            #
            #     self.road.vehicles = vehicles
            #
            #     self.ego.controlled_by_model = True
            # except Exception as e:
            #     print("ngsim_gail_env: 366: ", e)

            self.l2_init = np.linalg.norm(self.ego.position - self.vehicle.position)

        # self.v_sum = len(self.neighbours)
        # spd_sum = 0
        # for v in self.neighbours:
        #     spd_sum += v.velocity
        # self.spd_mean = spd_sum / self.v_sum

    def lane_offset(self):
        obs = self.observation.observe()  # 观察环境
        lane_w = 12 / 3.281  # 每条车道的宽度（单位为米）
        lane_offset = obs[0][1] - lane_w * self.vehicle.lane_index[2]  # 车辆相对于当前车道的偏移量
        return abs(lane_offset)

    def _reward_normal(self, action=None) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        lane_index = self.road.network.get_closest_lane_index(self.vehicle.position, self.vehicle.heading)  # 车辆最近的车道索引
        lane = self.road.network.get_lane(lane_index)  # 获取车道对象
        longitudinal, lateral = lane.local_coordinates(self.vehicle.position)  # 车辆在车道上的本地坐标
        lane_heading = lane.heading_at(longitudinal)  # 车道上对应本地坐标的航向角
        resul_heading = abs(self.vehicle.heading - lane_heading) * (180 / math.pi)  # 车辆航向角和车道航向角的差值
        angle = 20  # 航向角阈值
        heading_reward = 0 if resul_heading < angle else (
                    - np.clip((resul_heading - angle) / (180 - angle), 0, 1) - 0.5)  # 航向角奖励

        scaled_speed = utils.lmap(self.vehicle.velocity, [0, 20], [0, 1])  # 将车辆速度映射到0-1之间
        reward = heading_reward + scaled_speed + 0.5 * self.time / self.duration  # 组合奖励
        if self.vehicle.crashed or not self.vehicle.on_road:  # 如果车辆碰撞或不在道路上，奖励为-2
            reward = -2
        return reward

    def _reward(self, obs) -> float:
        rewards = []
        for vb in obs[1:2]:
            reward = 0
            distance = self.vehicle.lane_distance_to(self.ego)  # 计算车辆和Ego车辆的车道距离
            # print(distance)
            other_projected_speed = self.vehicle.velocity * np.dot(self.vehicle.direction,
                                                                   self.ego.direction)  # 计算车辆和Ego车辆在相对方向上的速度差
            # print(self.ego.speed - other_projected_speed)
            rr = self.ego.velocity - other_projected_speed  # Ego车辆和车辆的速度差
            ttc = distance / utils.not_zero(rr)  # 车辆和Ego车辆的时空最短距离
            # print('ttc', ttc)
            if np.sign(distance * rr) < 0:  # 车辆和Ego车辆的相对速度与距离的乘积小于0，即车辆和Ego车辆相对靠近
                if distance < 0:  # 车辆在Ego车辆的后方
                    reward = 1 / abs(ttc) * 6  # 根据碰撞时间给予奖励
                else:  # 车辆在Ego车辆的前方
                    reward = ttc / 5  # 根据时空最短距离给予奖励
            else:  # 车辆和Ego车辆相对远离
                reward = -10  # 给予固定的负奖励
            # print(reward)
            # if distance < -2:
            #     reward += 12
            if self.vehicle.crashed:  # 车辆被撞毁
                if self.ego.crashed and distance < -1:  # Ego车辆和车辆相撞
                    reward += self.config["ego_collision_reward"]  # 给予碰撞奖励
                else:
                    reward += self.config["bv_collision_reward"]  # 给予车辆相撞奖励
            # print(self.ego.lane_index[2], self.vehicle.lane_index[2])
            if abs(self.ego.lane_index[2] - self.vehicle.lane_index[2]) > 1:  # Ego车辆和车辆的车道索引之差大于1
                reward += -5  # 给予车道差距惩罚
            # else:
            #     lane_reward = (2- abs(self.ego.lane_index[2]-self.vehicle.lane_index[2])) * self.config["lane_reward"]
            #     reward += lane_reward
            rewards.append(reward)  # 将每一个观测到的车辆的奖励添加到列表中
        # print(rewards)
        reward = -10 if not self.vehicle.on_road else max(rewards)  # 如果车辆不在道路上则给予固定的负奖励，否则给予最大奖励
        # print(self.vehicle.crashed, reward)

        return reward

    def _reward_adv_v1(self, action=None) -> float:
        scaled_speed = utils.lmap(self.vehicle.velocity, [5, 20], [0, 1])  # 将车辆的速度在[5, 20]范围内线性映射到[0, 1]上
        dx = np.clip(abs(self.ego.position[0] - self.vehicle.position[0]), 0, 5)  # 计算车辆和Ego车辆在x轴上的距离
        dy = np.clip(abs(self.ego.position[1] - self.vehicle.position[1]), 0, 2)  # 计算车辆和Ego车辆在y轴上的距离
        reward_dx = 5 - dx - 2.5  # 根据x轴上的距离给予奖励
        reward_dy = 2 - dy - 1  # 根据y轴上的距离给予奖励
        reward = reward_dx + reward_dy + scaled_speed  # 根据距离和速度给予奖励
        if self.vehicle.crashed:  # 车辆被撞毁
            if self.ego.crashed:  # Ego车辆和车辆相撞
                reward += 10  # 给予碰撞奖励
            else:
                reward = -10  # 给予固定的负奖励
        reward = -10 if not self.vehicle.on_road else reward  # 如果车辆不在道路上则给予固定的负奖励，否则给予计算得到的奖励
        return reward

    def _reward_adv_new(self, action=None):
        # 计算对抗车与道路中心偏移距离
        lane_index = self.road.network.get_closest_lane_index(self.vehicle.position, self.vehicle.heading)
        lane = self.road.network.get_lane(lane_index)
        longitudinal, lateral = lane.local_coordinates(self.vehicle.position)

        # 计算被测车与道路中心偏移距离
        ego_lane_index = self.road.network.get_closest_lane_index(self.ego.position, self.ego.heading)
        ego_lane = self.road.network.get_lane(ego_lane_index)
        ego_longitudinal, ego_lateral = ego_lane.local_coordinates(self.ego.position)
        vehicle_longitudinal, vehicle_lateral = ego_lane.local_coordinates(self.vehicle.position)

        # exp 1 计算主车航向与道路朝向夹角，如果大于20度则对其惩罚
        angle = 20
        lane_heading = lane.heading_at(longitudinal)
        resul_heading = abs(self.vehicle.heading - lane_heading) * (180 / math.pi)
        heading_reward = 0 if resul_heading < angle else (- np.clip((resul_heading - angle) / (180 - angle), 0, 1) - 10)

        # 相对速度奖励
        scaled_speed_reward = (utils.lmap(self.vehicle.velocity, [0, 20], [0, 1])) * 5

        # 距离差奖励
        dx = np.clip(abs(self.ego.position[0] - self.vehicle.position[0]), 0, 10)
        dy = np.clip(abs(self.ego.position[1] - self.vehicle.position[1]), 0, 10)
        reward_dx = (10 - dx) * 0.4
        reward_dy = (10 - dy) * 0.4

        # 前车切入奖励
        ego_speed_reward = 0
        acc = self.ego.action['acceleration']
        # 如果自动驾驶车辆在前车后方且与前车的横向距离小于1.5，则奖励固定值5
        if (ego_longitudinal < vehicle_longitudinal) and abs(ego_lateral - vehicle_lateral) < 1.5:
            ego_speed_reward += 5

        # 防止自动驾驶车辆与NPC车辆静止不动，如果自动驾驶车辆速度大于等于1，则奖励为0；否则，奖励为-5
        if self.ego.velocity >= 1:
            ego_speed_reward += 0
        else:
            ego_speed_reward += -5

        acc_reward = 0.0
        if acc < 0:
            acc_reward = abs(acc) * 5

        # 综合奖励
        reward = reward_dx + reward_dy + scaled_speed_reward + heading_reward + ego_speed_reward + (
                    self.time / self.duration) * 1 + acc_reward

        # 发生碰撞时的奖励
        if self.vehicle.crashed:
            if self.ego.crashed:
                reward += -10

        # 离开道路时的奖励
        if not self.vehicle.on_road:
            reward += -10

        ########### SVO Reward: r_s
        w_1 = 6
        w_2 = 4
        r_1 = pow(2, 0.5)/2
        r_2 = -pow(2, 0.5)/2

        # ratio = 0.1 #原adversarial_reward的比重
        ratio = 0.05 # final

        distance = pow(dx**2 + dy**2, 0.5)
        U_ego = w_1 * self.ego.velocity
        U_sv = w_2 * (self.vehicle.velocity / distance)

        r_s = U_ego * r_1 + U_sv * r_2
        rewards = reward * ratio + r_s
        ######################## 不引入SVO ###############
#        rewards = reward
        
        return rewards
    def _reward_adv(self, action=None) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        if self.gail:
            return self._reward_normal(action)
        else:
            return self._reward_adv_new(action)

    def step(self, action=None):
        """
        Perform a MDP step
        """
        if self.road is None or self.vehicle is None:
            raise NotImplementedError("The road and vehicle must be initialized in the environment implementation")

        gail_features, adv_fea = self._simulate(action)
        obs = self.observation.observe()
        reward = self._reward_adv()
        terminal = self._is_terminal()
        # print(reward)

        info = {
            "TTC_THW": self.TTC_THW,
            "distance": self.distance,
            "ego_distance": self.ego_distance,
            "velocity": self.vehicle.velocity,
            "crashed": self.vehicle.crashed,
            'offroad': not self.vehicle.on_road,
            "action": action,
            "time": self.time,
            'collision_data': self.before_collision_data,
        }

        return (gail_features, adv_fea), reward, terminal, info

    def _simulate(self, action):
        """
        执行规划轨迹的几个模拟步骤
        """
        trajectory_features = []  # 存储轨迹特征
        self.distance = 0.0
        self.ego_distance = 0.0
        # T = action[2] if action is not None else 5
        self.TTC_THW = []  # 存储TTC和THW
        self.before_collision_data = {}  # 存储碰撞前的数据
        frames = 1  # 每一帧的数目
        if self.kind_action == 'discrete':  # 离散控制
            frames = 10 if self.ACTIONS[int(action)] in ['LANE_LEFT', 'LANE_RIGHT'] else 1

        frames = frames if self.kind_action == 'discrete' else 1
        for i in range(frames):
            # if not self.gail and self.steps > 0:
            #     self.change_adversarial_vehicle()
            if i == 0:
                if action is not None:  # 采样目标
                    pass
                    # self.vehicle.trajectory_planner(action[0], action[1], action[2])
                else:  # 人类目标
                    pass
                    # print('self.vehicle.sim_steps', self.vehicle.sim_steps)
                    # self.vehicle.planned_trajectory = self.vehicle.ngsim_traj[
                    #                                   self.vehicle.sim_steps:(self.vehicle.sim_steps + T * 10), :2]
                    # print(self.vehicle.planned_trajectory.shape)
                    # print(self.vehicle.planned_trajectory[0])
                    # self.vehicle.trajectory_planner(self.vehicle.ngsim_traj[self.vehicle.sim_steps+T*10][1],
                    #                                (self.vehicle.ngsim_traj[self.vehicle.sim_steps+T*10][0]-self.vehicle.ngsim_traj[self.vehicle.sim_steps+T*10-1][0])/0.1, T)
                    # print(self.vehicle.planned_trajectory.shape)
                    # print(self.vehicle.planned_trajectory[:5])
                self.run_step = 1
            self.before_collision_data = {
                'av_speed': self.vehicle.velocity,
                'av_heading': self.vehicle.heading,
            }
            if not self.gail:
                self.before_collision_data.update(
                    {
                        'bv_speed': self.ego.velocity,
                        'bv_heading': self.ego.heading,
                        'dx': self.vehicle.position[0] - self.ego.position[0],
                        'dy': self.vehicle.position[1] - self.ego.position[1]
                    }
                )

            self.vehicle_pre_lane = self.vehicle.lane_index[2]
            # 用于计算行使距离
            if self.steps == 0:

                self.vehicle_position = self.vehicle.position.copy()
                if not self.gail:
                    self.ego_position = self.ego.position.copy()

            self.road.act(self.run_step)
            if action is not None and (self.gail or self.open_adversarial_function):
                if self.kind_action == 'continuous':  # 连续控制
                    self.vehicle.action['steering'] = action[0]
                    self.vehicle.action['acceleration'] = action[1]
                else:  # 离散控制
                    if i == 0:
                        self.vehicle.act(self.ACTIONS[int(action)])
                    else:
                        self.vehicle.act()
            # 被测主车使用交通流模型控制
            if self.ego is not None and self.traffic_flow_action is not None and self.open_adversarial_function:
                self.ego.act(self.ACTIONS[int(self.traffic_flow_action)])

            # self.vehicle.color = (0, 0, 255)
            # print(self.vehicle.action)
            self.road.step(1 / self.SIMULATION_FREQUENCY)
            self.time += 1
            self.run_step += 1
            self.steps += 1
            # 获取特征
            features = self.gail_features_v2()
            features_adv = None if self.gail else self.adv_features()

            # action = [self.vehicle.action['steering'], self.vehicle.action['acceleration']]
            # print(features, action)
            trajectory_features.append(features)
            data = self.calculate_thw_and_ttc(self.vehicle)
            if not self.gail:
                data.extend(self.calculate_thw_and_ttc(self.ego))
            self.TTC_THW.append(data)
            # 用于计算行使距离
            if self.steps >= 1:
                self.distance = np.linalg.norm(self.vehicle_position - self.vehicle.position)
                self.vehicle_position = self.vehicle.position.copy()
                if not self.gail:
                    self.ego_distance = np.linalg.norm(self.ego_position - self.ego.position)
                    self.ego_position = self.ego.position.copy()

            self.nearest = self.road.close_vehicles_to(self.vehicle, 50, 10, sort=True)
            # print(self.vehicle)
            # print(self.nearest)
            # if self.nearest:
            #     for v in self.nearest:
            #         # print(v.position)
            #         v.color = (255, 0, 0)
            try:
                self._automatic_rendering()
            except:
                self.done = True
                break

            # Stop at terminal states
            if self.done or self._is_terminal():
                break

            self._clear_vehicles()
            # 当完成变道时可跳出循环
            if self.kind_action == 'discrete' and self.ACTIONS[int(action)] in ['LANE_LEFT', 'LANE_RIGHT']:
                target_lane_index = self.vehicle.target_lane_index
                target_lane = self.road.network.get_lane(target_lane_index)
                longitudinal, lateral = target_lane.local_coordinates(self.vehicle.position)
                if abs(lateral) < 0.5:
                    break
        self.enable_auto_render = False
        # for v in self.road.vehicles:
        #     if hasattr(v, 'color'):
        #         delattr(v, 'color')

        return features, features_adv

    def adv_features(self):
        """
        获取高级特征，用于生成adversarial样本
        """
        # 获取当前观测值
        obs = self.observation.observe()

        # 获取离车辆最近的车道索引
        lane_index = self.road.network.get_closest_lane_index(self.vehicle.position, self.vehicle.heading)

        # 根据车道索引获取车道对象
        lane = self.road.network.get_lane(lane_index)

        # 根据车道对象和车辆位置获取车辆在车道上的纵向和横向坐标
        longitudinal, lateral = lane.local_coordinates(self.vehicle.position)

        # 根据车道对象和纵向坐标获取该位置的车道宽度
        lane_w = lane.width_at(longitudinal)

        # 计算车辆在车道上的横向偏移
        lane_offset = lateral

        # 获取车道在纵向坐标上的方向角
        lane_heading = lane.heading_at(longitudinal)

        # 从观测值中获取车辆特征
        vehicle_fea = obs[0][:5].tolist()

        # 获取车辆当前加速度
        acc = self.ego.action['acceleration']

        # 构造特征列表
        features = [lane_heading, lane_w, lane_offset, (lane_heading - vehicle_fea[4]) / math.pi * 180, vehicle_fea[4],
                    acc, self.ego.velocity, self.vehicle.velocity]

        # 构造当前车辆特征列表
        ego_fea = [vehicle_fea[0] - self.ego.to_dict()['x'], vehicle_fea[1] - self.ego.to_dict()['y'],
                   vehicle_fea[2] - self.ego.to_dict()['vx'], vehicle_fea[3] - self.ego.to_dict()['vy'],
                   self.ego.to_dict()['heading']]

        # 将当前车辆特征列表添加到特征列表中
        features.extend(ego_fea)

        # 将特征列表转为numpy数组
        adv_features = np.array(features)

        # 返回高级特征
        return adv_features

    def gail_features_v2(self):
        """
        获取GAIL特征，用于生成adversarial样本
        """
        # 获取当前观测值
        obs = self.observation.observe()

        # 获取离车辆最近的车道索引
        lane_index = self.road.network.get_closest_lane_index(self.vehicle.position, self.vehicle.heading)

        # 根据车道索引获取车道对象
        lane = self.road.network.get_lane(lane_index)

        # 根据车道对象和车辆位置获取车辆在车道上的纵向和横向坐标
        longitudinal, lateral = lane.local_coordinates(self.vehicle.position)

        # 根据车道对象和纵向坐标获取该位置的车道宽度
        lane_w = lane.width_at(longitudinal)

        # 计算车辆在车道上的横向偏移
        lane_offset = lateral

        # 获取车道在纵向坐标上的方向角
        lane_heading = lane.heading_at(longitudinal)

        # 构造特征列表
        features = [lane_offset, lane_heading, lane_w]

        # 将观测值的一部分特征添加到特征列表中
        features += obs[0][2:5].tolist()

        # 遍历观测值的剩余部分
        for vb in obs[1:]:
            # 计算观测值的差值
            core = obs[0] - vb

            # 将差值前5个特征添加到特征列表中
            features += core[:5].tolist()

        # 返回GAIL特征
        return features

    def set_traffic_flow_action(self, traffic_flow_action):
        self.traffic_flow_action = traffic_flow_action

    def ego_gail_features(self):
        obs = self.observation.observe(self.ego)
        lane_index = self.road.network.get_closest_lane_index(self.ego.position, self.ego.heading)
        lane = self.road.network.get_lane(lane_index)
        longitudinal, lateral = lane.local_coordinates(self.ego.position)
        lane_w = lane.width_at(longitudinal)
        lane_offset = lateral
        lane_heading = lane.heading_at(longitudinal)

        features = [lane_offset, lane_heading, lane_w]
        features += obs[0][2:5].tolist()
        for vb in obs[1:]:
            core = obs[0] - vb
            features += core[:5].tolist()
        # print(len(features), features)
        return features

    def calculate_ttc(self, front_veh, rear_veh):
        # 计算车辆之间的时间到碰撞（TTC）
        long_ttc = 10  # 默认值为10

        if front_veh is rear_veh:
            return long_ttc

        front_speed_x = front_veh.velocity * np.cos(front_veh.heading)
        rear_speed_x = rear_veh.velocity * np.cos(rear_veh.heading)
        # front_speed_y = front_veh.velocity * np.sin(front_veh.heading)
        # rear_speed_y = rear_veh.velocity * np.sin(rear_veh.heading)

        if front_veh.position[0] > rear_veh.position[0]:
            if rear_speed_x > front_speed_x:
                pass
            else:
                # 根据速度计算TTC
                long_ttc = (front_veh.position[0] - rear_veh.position[0]) / utils.not_zero(rear_speed_x - front_speed_x)

        return min(10, long_ttc)

    def calculate_thw_and_ttc(self, vehicle):
        THWs = [100]
        TTCs = [100]

        for v in self.road.vehicles:
            if v is vehicle:
                continue

            # 仅考虑前方相遇的车辆
            if v.position[0] > vehicle.position[0] and abs(
                    v.position[1] - vehicle.position[1]) < vehicle.WIDTH and vehicle.velocity >= 1:
                v_speed = v.velocity * np.cos(v.heading)
                vehicle_speed = vehicle.velocity * np.cos(vehicle.heading)

                if vehicle_speed > 0:
                    # 根据速度计算THW
                    THW = (v.position[0] - vehicle.position[0]) / utils.not_zero(vehicle_speed)
                else:
                    THW = 100

                if v_speed > vehicle_speed:
                    TTC = 100
                else:
                    # 根据速度计算TTC
                    TTC = (v.position[0] - vehicle.position[0]) / utils.not_zero(vehicle_speed - v_speed)

                THWs.append(THW)
                TTCs.append(TTC)

        return [min(TTCs), min(THWs)]

    def _is_terminal(self):
        """
        The episode is over if the ego vehicle crashed or go off road or the time is out.
        该函数用于判断当前episode是否结束，条件包括：自车碰撞、偏离道路或超时。
        """
        if self.gail:
            if self.vehicle.linear is not None and self.vehicle.IDM:
                s_v, lat_v = self.vehicle.linear.local_coordinates(self.vehicle.position)
                return self.vehicle.crashed or s_v >= self.vehicle.linear.length

            return self.vehicle.crashed or self.time >= self.duration - 1 or self.vehicle.position[
                0] >= 2150 / 3.281 or not self.vehicle.on_road
        else:
            return self.vehicle.crashed or self.ego is None or np.linalg.norm(
                self.ego.position - self.vehicle.position) >= 50 or self.time >= self.duration - 1 or \
                self.vehicle.position[
                    0] >= 2150 / 3.281 or not self.vehicle.on_road or self.ego.crashed or not self.ego.on_road

    def _clear_vehicles(self) -> None:
        """
        The function is used to clear vehicles on the road.
        该函数用于清除道路上的车辆。
        """
        is_leaving = lambda vehicle: (not vehicle.IDM and (
                self.run_step >= (vehicle.planned_trajectory.shape[0] - 1) or vehicle.next_position is None or ~
        np.array(vehicle.next_position).reshape(1, -1).any(axis=1)[0])) or \
                                     (vehicle.IDM and vehicle.linear.local_coordinates(vehicle.position)[
                                         0] >= vehicle.linear.length - vehicle.LENGTH)

        vehicles = []
        for vehicle in self.road.vehicles:
            """
            lane_index = self.road.network.get_closest_lane_index(vehicle.position, vehicle.heading)
            if lane_index in [('s3', 'merge_out', -1)]:
                lane = self.road.network.get_lane(lane_index)
                longitudinal, lateral = lane.local_coordinates(vehicle.position)
                if longitudinal >= lane.length - vehicle.LENGTH:
                    pass
            """
            if vehicle.linear is not None and vehicle.IDM:
                s_v, lat_v = vehicle.linear.local_coordinates(vehicle.position)
                if vehicle is self.vehicle or not s_v >= vehicle.linear.length - vehicle.LENGTH / 2:
                    vehicles.append(vehicle)
                else:
                    vehicle.linear = None
                    vehicles.append(vehicle)
            else:
                if vehicle.position[0] >= 2150 / 3.281 or not self.vehicle.on_road:
                    continue
                else:
                    vehicles.append(vehicle)

        self.road.vehicles = vehicles
