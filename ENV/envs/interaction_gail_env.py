from __future__ import division, print_function, absolute_import

import random
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
from NGSIM_env.vehicle.humandriving import HumanLikeVehicle, InterActionVehicle, IntersectionHumanLikeVehicle, Pedestrian
from NGSIM_env.road.lane import LineType, StraightLane, PolyLane
from NGSIM_env.utils import *
import pickle
import lanelet2
from threading import Thread


class InterActionGAILEnv(AbstractEnv):
    """
    A Intersection driving environment with Interaction data for collecting gail training data.
    一个交叉驾驶环境，用于收集GAIL训练数据的交互数据。
    """

    def __init__(self, path, vehicle_id, IDM=False, render=True, gail=True,  action_type='continuous'):
        # 初始化环境并设置必要的参数
        # path: 轨迹数据文件的路径
        # vehicle_id: 自车ID
        # IDM: 是否使用IDM控制器来控制自车
        # render: 是否显示环境的可视化界面
        # gail: 是否使用GAIL进行训练
        # action_type: 行为空间的类型，可以是'continuous'或'discrete'
        f = open(path, 'rb')
        self.trajectory_set = pickle.load(f)  # 加载轨迹数据集
        f.close()
        self.vehicle_id = vehicle_id
        self.ego_length = self.trajectory_set['ego']['length']  # 自车长度
        self.ego_width = self.trajectory_set['ego']['width']  # 自车宽度
        self.ego_trajectory = self.trajectory_set['ego']['trajectory']  # 自车轨迹
        self.duration = len(self.ego_trajectory) - 3  # 自车轨迹长度减去3
        self.surrounding_vehicles = list(self.trajectory_set.keys())  # 周围车辆ID列表
        self.surrounding_vehicles.pop(0)  # 去掉自车ID
        self.run_step = 0  # 运行步数
        self.human = False  # 是否使用人作为目标
        self.IDM = IDM  # 是否使用IDM
        self.reset_time = 0  # 重置环境的时间
        self.show = render  # 是否显示可视化界面
        self.gail = gail  # 是否使用GAIL
        self.laneletmap = None  # 地图
        self.kind_action = action_type  # 行为空间的类型
        if isinstance(vehicle_id, str) and 'P' in vehicle_id:
            self.ped_ids = self.trajectory_set['ego']['ped_ids']  # 行人ID列表
        super(InterActionGAILEnv, self).__init__()  # 调用父类的构造函数

    def default_config(self):
        # 设置环境的默认配置
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                'see_behind': True,
                "features": ["x", 'y', "vx", 'vy', 'heading', 'vehicle_id'],
                "normalize": False,
                "absolute": True,
                "vehicles_count": 11},
            "osm_path": "/home/sx/wjc/wjc/datasets/Interaction/INTERACTION-Dataset-TC-v1_0/maps/TC_BGR_Intersection_VA.osm",
            "vehicles_count": 10,
            "show_trajectories": False,
            "screen_width": 400,
            "screen_height": 400,
        })

        return config

    def reset(self, human=False, reset_time=0, video_save_name=''):
        '''
        重置环境到指定的时间（场景）并指定是否使用人作为目标
        '''
        self.video_save_name = video_save_name  # 视频保存的名称
        self.human = human  # 是否使用人作为目标
        self.load_map()  # 加载地图
        self._create_road()  # 创建道路结构
        self.reset_time = reset_time  # 设置重置时间
        self._create_vehicles(reset_time)  # 创建车辆
        self.steps = 0  # 步数归零

        return super(InterActionGAILEnv, self).reset()  # 调用父类的reset方法

    def load_map(self):
        """
        加载环境的地图
        """
        if not hasattr(self, 'roads_dict'):  # 如果没有加载过道路字典
            projector = lanelet2.projection.UtmProjector(lanelet2.io.Origin(0.0, 0.0))  # 设置坐标转换器
            laneletmap = lanelet2.io.load(self.config['osm_path'], projector)  # 加载地图
            self.roads_dict, self.graph, self.laneletmap, self.indegree, self.outdegree = utils.load_lanelet_map(laneletmap)  # 加载车道地图

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
        创建自车和NGSIM车辆，并将它们添加到道路上。
        """
        T = 100  # 计划轨迹的长度
        self.controlled_vehicles = []  # 可控制车辆列表
        whole_trajectory = self.ego_trajectory  # 自车轨迹数据
        ego_trajectory = np.array(whole_trajectory[reset_time:])  # 根据重置时间获取自车轨迹的子集

        v0 = np.linalg.norm(ego_trajectory[0, 2])  # 自车初始速度
        v1 = np.linalg.norm(ego_trajectory[1, 2])  # 自车目标速度

        if not self.vehicle_id.startswith('P'):  # 如果车辆id不以'P'开头，即非行人车辆
            self.vehicle = IntersectionHumanLikeVehicle.create(self.road, self.vehicle_id, ego_trajectory[0][:2],
                                                               self.ego_length,
                                                               self.ego_width, ego_trajectory, acc=(v1 - v0) * 10,
                                                               velocity=v0,
                                                               heading=ego_trajectory[0][3], target_velocity=v1,
                                                               human=self.human, IDM=True)
        else:
            self.vehicle = Pedestrian.create(self.road, self.vehicle_id, ego_trajectory[0][:2], self.ego_length * 2,
                                             self.ego_width * 2, ego_trajectory, acc=(v1 - v0) * 10, velocity=v0,
                                             heading=ego_trajectory[0][3], target_velocity=v1, human=self.human,
                                             IDM=True)

        self.vehicle.planned_trajectory = self.vehicle.ngsim_traj[:(T * 10), :2]  # 计划轨迹的位置信息
        self.vehicle.planned_speed = self.vehicle.ngsim_traj[:(T * 10), 2:3]  # 计划轨迹的速度信息
        self.vehicle.planned_heading = self.vehicle.ngsim_traj[:(T * 10), 3:]  # 计划轨迹的朝向信息

        self.vehicle.is_ego = True  # 将车辆标记为自车
        self.vehicle.make_linear()  # 将车辆设置为线性移动

        self.road.vehicles.append(self.vehicle)  # 将车辆添加到道路上的车辆列表中
        self.controlled_vehicles.append(self.vehicle)  # 将车辆添加到可控制车辆列表中
        self.ego = None  # 将ego属性设置为None
        self._create_bv_vehicles(reset_time, 10, self.steps)  # 创建背景车辆

    def _create_bv_vehicles(self, reset_time, T, current_time):
        # 如果重置时间加上当前时间大于等于自车轨迹长度，则返回
        if (reset_time + current_time) >= len(self.ego_trajectory):
            return

        # vehicles = []
        # for veh_id in self.surrounding_vehicles:
        #     # try:
        #     other_trajectory = np.array(self.trajectory_set[veh_id]['trajectory'][reset_time:])
        #     flag = ~(np.array(other_trajectory[current_time])).reshape(1, -1).any(axis=1)[0]
        #     if current_time == 0:
        #         pass
        #     else:
        #         trajectory = np.array(self.trajectory_set[veh_id]['trajectory'][reset_time:])
        #
        #         if not flag and ~(np.array(trajectory[current_time-1])).reshape(1, -1).any(axis=1)[0]:
        #             flag = False
        #         else:
        #             flag = True
        #
        #
        #     if not flag:
        #         # print("add vehicle of No.{} step.".format(current_time))
        #         v0 = np.linalg.norm(other_trajectory[current_time, 2])
        #         v1 = np.linalg.norm(other_trajectory[current_time, 2])
        #         target_v = np.max(other_trajectory[:, 2])
        #         if not isinstance(veh_id, str):
        #             other_vehicle = IntersectionHumanLikeVehicle.create(self.road, veh_id, other_trajectory[current_time][:2],
        #                                                       self.trajectory_set[veh_id]['length'],
        #                                                       self.trajectory_set[veh_id]['width'],
        #                                                       other_trajectory, acc=(v1 - v0) * 10,
        #                                                       velocity=v0,
        #                                                       heading=other_trajectory[current_time][3],
        #                                                       target_velocity=target_v,
        #                                                       human=self.human,
        #                                                       IDM=False)
        #         elif veh_id.startswith('P'):
        #
        #             other_vehicle = Pedestrian.create(self.road, veh_id, other_trajectory[current_time][:2],
        #                                                       self.trajectory_set[veh_id]['length'] * 2,
        #                                                       self.trajectory_set[veh_id]['width'] * 2,
        #                                                       other_trajectory, acc=(v1 - v0) * 10,
        #                                                       velocity=v0, target_velocity=target_v,
        #                                                       heading=other_trajectory[current_time][3],
        #                                                       human=self.human,
        #                                                       IDM=False)
        #
        #         other_vehicle.planned_trajectory = other_vehicle.ngsim_traj[
        #                                            :(T * 10), :2]
        #         other_vehicle.planned_speed = other_vehicle.ngsim_traj[
        #                                       :(T * 10), 2:3]
        #         other_vehicle.planned_heading = other_vehicle.ngsim_traj[
        #                                         :(T * 10), 3:]
        #
        #         # zeros = np.where(~other_trajectory.any(axis=1))[0]
        #         # if len(zeros) == 0:
        #         #     zeros = [0]
        #         # other_target = self.road.network.get_closest_lane_index(
        #         #     position=other_trajectory[int(zeros[0] - 1)][:2])  # heading=other_trajectory[-1][3]
        #         # other_vehicle.plan_route_to(other_target[1])
        #         vehicles.append(other_vehicle)
        #
        #     # except Exception as e:
        #     #     print("_create_bv_vehicles", e)
        # else:
        #     # print("road length:", len(self.road.vehicles))
        #     if len(vehicles) > 0:
        #         # print("before road length:", len(self.road.vehicles))
        #         for vh in self.road.vehicles:
        #             vehicles.append(vh)
        #         self.road.vehicles = vehicles
        #         # print("road length:", len(self.road.vehicles))

        vehicles = []  # 存储周围车辆的列表
        for veh_id in self.surrounding_vehicles:  # 遍历周围车辆的ID
            try:
                # 获取其他车辆的轨迹数据
                other_trajectory = np.array(self.trajectory_set[veh_id]['trajectory'][reset_time:])
                # 判断当前时间步是否允许添加车辆
                flag = ~(np.array(other_trajectory[current_time])).reshape(1, -1).any(axis=1)[0]

                # 如果当前时间步不为0，则进行额外的判断
                if current_time == 0:
                    pass
                else:
                    trajectory = np.array(self.trajectory_set[veh_id]['trajectory'][reset_time:])
                    # 如果flag为false且前一时间步车辆的行进方向也为false，则将flag设为False，否则设为True
                    if not flag and ~(np.array(trajectory[current_time - 1])).reshape(1, -1).any(axis=1)[0]:
                        flag = False
                    else:
                        flag = True

                # 如果flag为False，创建新车辆对象
                if not flag:
                    other_vehicle = IntersectionHumanLikeVehicle.create(self.road, veh_id,
                                                                        other_trajectory[current_time][:2],
                                                                        self.trajectory_set[veh_id]['length'],
                                                                        self.trajectory_set[veh_id]['width'],
                                                                        other_trajectory, acc=0.0,
                                                                        velocity=other_trajectory[current_time][2],
                                                                        heading=other_trajectory[current_time][3],
                                                                        human=self.human,
                                                                        IDM=self.IDM, start_step=self.steps)
                    mask = np.where(np.sum(other_vehicle.ngsim_traj[:(T * 10), :2], axis=1) == 0, False, True)
                    other_vehicle.planned_trajectory = other_vehicle.ngsim_traj[:(T * 10), :2][mask]
                    other_vehicle.planned_speed = other_vehicle.ngsim_traj[:(T * 10), 2:3][mask]
                    other_vehicle.planned_heading = other_vehicle.ngsim_traj[:(T * 10), 3:][mask]

                    # 如果计划轨迹的长度小于等于5，则跳过本次迭代
                    if other_vehicle.planned_trajectory.shape[0] <= 5:
                        continue

                    # 获取目标车道索引
                    other_target = self.road.network.get_closest_lane_index(
                        position=other_vehicle.planned_trajectory[-1])
                    # 规划车辆到目标位置的路径
                    other_vehicle.plan_route_to(other_target[1])
                    vehicles.append(other_vehicle)

            except Exception as e:  # 捕获异常并打印错误信息
                print("_create_bv_vehicles", e)
        else:
            # 如果vehicles列表中有车辆，将其添加到道路中
            if len(vehicles) > 0:
                for vh in self.road.vehicles:
                    vehicles.append(vh)
                self.road.vehicles = vehicles

        # 如果gail为False且当前时间为0
        if not self.gail and current_time == 0:
            self.ego = None  # 将自车对象置空
            if not self.vehicle_id.startswith('P'):  # 如果vehicle_id不以'P'开头
                self.open_adversarial = False  # 关闭对抗模式
                dis = 1e6  # 初始化距离为一个极大值
                # 获取到周围车辆列表
                self.neighbours = self.road.close_vehicles_to(self.vehicle, 50, 5, sort=True)

                # 遍历周围车辆
                for v in self.neighbours:
                    bv_vx = np.cos(v.heading) * v.velocity
                    bv_vy = np.sin(v.heading) * v.velocity
                    av_vx = np.cos(self.vehicle.heading) * self.vehicle.velocity
                    av_vy = np.sin(self.vehicle.heading) * self.vehicle.velocity
                    vector_a = [bv_vx, bv_vy]
                    vector_b = [av_vx, av_vy]
                    # 计算两个向量的夹角
                    angle = calculate_angle(vector_a, vector_b)
                    tmp_dis = np.linalg.norm(v.position - self.vehicle.position)
                    # 如果距离小于dis且角度小于30度，则设置自车为对抗模式
                    if tmp_dis < dis and angle < 30:
                        self.open_adversarial = True
                        dis = tmp_dis
                        self.ego = v
                        self.ego.color = (0, 0, 255)
                # 如果没有找到自车，则使用最近的车辆
                if not self.ego:
                    try:
                        self.ego = self.neighbours[0]
                        self.ego.color = (200, 200, 0)
                    except:
                        # 如果周围车辆数仍小于1，则将gail设置为True并返回
                        self.neighbours = self.road.close_vehicles_to(self.vehicle, 1000, 1, sort=True)
                        if len(self.neighbours) >= 1:
                            self.ego = self.neighbours[0]
                            self.ego.color = (200, 200, 0)
                        else:
                            self.gail = True
                            self.spd_mean = self.vehicle.velocity
                            return
            else:  # 如果vehicle_id以'P'开头，则随机选择一个行人作为自车
                seq = self.ped_ids.get(reset_time)
                print(self.vehicle.vehicle_ID)
                if seq is not None:
                    veh_id = random.choice(seq)
                    for v in self.road.vehicles:
                        if v.vehicle_ID == veh_id:
                            self.ego = v
                            self.ego.color = (0, 0, 255)
                            self.ego.IDM = True
                            break

            if self.ego is not None:  # 如果自车不为空
                self.open_adversarial = True
                self.ego.make_linear()
                self.ego.controlled_by_model = True
                if self.ego.linear is not None:
                    self.ego.IDM = True
            else:
                print('no test vehicle')

        self.v_sum = len(self.road.vehicles)  # 道路上的车辆总数
        spd_sum = 0
        for v in self.road.vehicles:
            spd_sum += v.velocity
        self.spd_mean = spd_sum / self.v_sum  # 计算速度均值

    def _reward_normal(self, action=None) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        # 如果是连续动作模式
        if self.kind_action == 'continuous':
            # 获取最近车道的索引和车道对象
            lane_index = self.road.network.get_closest_lane_index(self.vehicle.position, self.vehicle.heading)
            lane = self.road.network.get_lane(lane_index)
            # 获取车辆相对于车道的纵向和横向位置
            longitudinal, lateral = lane.local_coordinates(self.vehicle.position)
            # 获取车道的朝向角度和车辆的朝向角度之间的差异，单位为度
            lane_heading = lane.heading_at(longitudinal)
            result_heading = abs(self.vehicle.heading - lane_heading) * (180 / math.pi)
            # 设置阈值角度为20度
            angle = 20
            # 根据差异值计算奖励值
            heading_reward = 0 if result_heading < angle else (
                    - np.clip((result_heading - angle) / (180 - angle), 0, 1) - 0.5)

            # 根据车速计算奖励值
            scaled_speed = utils.lmap(self.vehicle.velocity, [0, 20], [0, 1])
            reward = heading_reward + scaled_speed + 0.5 * self.time / self.duration
            # 如果车辆撞车或者不在路上，奖励值为-2
            if self.vehicle.crashed or not self.vehicle.on_road:
                reward = -2
            return reward
        else:
            # 计算车辆在正向速度上的分量
            forward_speed = self.vehicle.velocity * np.cos(self.vehicle.heading)
            # 根据速度值计算奖励值
            scaled_speed = utils.lmap(forward_speed, [0, 20], [0, 1])
            reward = 0.1 * self.time / self.duration + scaled_speed
            # 如果车辆撞车或者不在路上，奖励值为-2
            if self.vehicle.crashed or not self.vehicle.on_road:
                reward = -2
            return reward

    def _reward_adv(self, action=None) -> float:
        if self.gail:
            return self._reward_normal(action)
        else:
            return self._reward_adv_v3(action)

    def _reward_adv_v1(self, action=None) -> float:
        # 计算速度变化程度，并将其映射到[0, 1]之间
        scaled_speed = utils.lmap(self.vehicle.velocity, [5, 20], [0, 1])

        # 计算与前车的横向和纵向距离，并将其限制在[0, 5]和[0, 2]之间
        dx = np.clip(abs(self.ego.position[0] - self.vehicle.position[0]), 0, 5)
        dy = np.clip(abs(self.ego.position[1] - self.vehicle.position[1]), 0, 2)

        # 计算纵向距离对应的奖励
        reward_dx = 5 - dx - 2.5
        # 计算横向距离对应的奖励
        reward_dy = 2 - dy - 1

        # 计算总的奖励，包括横向距离、纵向距离和速度的贡献
        reward = reward_dx + reward_dy + scaled_speed

        # 如果前车发生碰撞
        if self.vehicle.crashed:
            # 如果AV（自动驾驶汽车）也发生碰撞，奖励加10
            if self.ego.crashed:
                reward += 10
            # 如果只有前车发生碰撞，奖励为负10
            else:
                reward = -10

        # 如果前车不在道路上，奖励为负10
        reward = -10 if not self.vehicle.on_road else reward

        return reward

    def _reward_adv_v2(self, action=None):
        # 获取当前车道的索引
        lane_index = self.road.network.get_closest_lane_index(self.vehicle.position, self.vehicle.heading)
        # 获取当前车道
        lane = self.road.network.get_lane(lane_index)
        # 计算当前车的纵向和横向坐标
        longitudinal, lateral = lane.local_coordinates(self.vehicle.position)
        # 计算车道的朝向角度
        lane_heading = lane.heading_at(longitudinal)
        # 计算车辆的朝向与车道朝向的差值（以角度表示）
        resul_heading = abs(self.vehicle.heading - lane_heading) * (180 / math.pi)

        # 计算与车道朝向的夹角奖励
        angle = 20
        heading_reward = 0 if resul_heading < angle else (- np.clip((resul_heading - angle) / (180 - angle), 0, 1) - 10)

        # 计算速度变化程度对应的奖励
        scaled_speed_reward = (utils.lmap(self.vehicle.velocity, [0, 20], [0, 1]) - 0.25) * 8

        # 计算与前车的横向和纵向距离，并将其限制在[0, 5]和[0, 2]之间
        dx = np.clip(abs(self.ego.position[0] - self.vehicle.position[0]), 0, 5)
        dy = np.clip(abs(self.ego.position[1] - self.vehicle.position[1]), 0, 2)
        # 计算与前车横向距离对应的奖励
        reward_dx = (5 - dx) * 0.2
        # 计算与前车纵向距离对应的奖励
        reward_dy = (2 - dy) * 0.5

        # 前车切入的奖励
        acc = self.ego.action['acceleration']
        # 如果前车没有发生碰撞，并且AV在前车速度大于5的情况下减速，计算相应的奖励
        if not self.vehicle.crashed and acc < 0 and self.ego.velocity > 5:
            # 获取AV当前所在的车道
            ego_lane_index = self.road.network.get_closest_lane_index(self.ego.position, self.ego.heading)
            ego_lane = self.road.network.get_lane(ego_lane_index)
            # 计算AV和前车在该车道上的纵向和横向坐标
            ego_longitudinal, ego_lateral = ego_lane.local_coordinates(self.ego.position)
            vehicle_longitudinal, vehicle_lateral = ego_lane.local_coordinates(self.vehicle.position)
            # 计算AV速度变化程度对应的奖励
            ego_speed_reward = np.clip(abs(acc), 0, 5) + (
                        np.clip(abs(ego_longitudinal - vehicle_longitudinal), 0, 5) - 2)
        else:
            ego_speed_reward = 0 if self.ego.velocity >= 5 else -2

        # 计算总的奖励，包括横向距离、纵向距离、速度和夹角的贡献
        reward = reward_dx + reward_dy + scaled_speed_reward + heading_reward + ego_speed_reward + (
                    self.time / self.duration) * 0.5

        # 如果前车发生碰撞
        if self.vehicle.crashed:
            # 如果AV（自动驾驶汽车）也发生碰撞，奖励加10
            if self.ego.crashed:
                reward += 10

        # 如果前车不在道路上，奖励为负10
        if not self.vehicle.on_road:
            reward += -10

        return reward

    def _reward_adv_v3(self, action=None):
        # 获取当前车道的索引
        lane_index = self.road.network.get_closest_lane_index(self.vehicle.position, self.vehicle.heading)
        # 获取当前车道
        lane = self.road.network.get_lane(lane_index)
        # 计算当前车的纵向和横向坐标
        longitudinal, lateral = lane.local_coordinates(self.vehicle.position)
        # 计算车道的朝向角度
        lane_heading = lane.heading_at(longitudinal)
        # 计算车辆的朝向与车道朝向的差值（以角度表示）
        resul_heading = abs(self.vehicle.heading - lane_heading) * (180 / math.pi)

        # 获取AV当前所在的车道
        ego_lane_index = self.road.network.get_closest_lane_index(self.ego.position, self.ego.heading)
        ego_lane = self.road.network.get_lane(ego_lane_index)
        # 计算AV和前车在该车道上的纵向和横向坐标
        ego_longitudinal, ego_lateral = ego_lane.local_coordinates(self.ego.position)
        vehicle_longitudinal, vehicle_lateral = ego_lane.local_coordinates(self.vehicle.position)

        # 计算与车道朝向的夹角奖励
        angle = 20
        heading_reward = 0 if resul_heading < angle else (- np.clip((resul_heading - angle) / (180 - angle), 0, 1) - 10)

        # 计算速度变化程度对应的奖励
        scaled_speed_reward = (utils.lmap(self.vehicle.velocity, [0, 20], [0, 1]) - 0.25) * 4

        # 计算与前车的横向和纵向距离，并将其限制在[0, 10]和[0, 10]之间
        dx = np.clip(abs(self.ego.position[0] - self.vehicle.position[0]), 0, 10)
        dy = np.clip(abs(self.ego.position[1] - self.vehicle.position[1]), 0, 10)
        # 计算与前车横向距离对应的奖励
        reward_dx = (10 - dx) * 0.4
        # 计算与前车纵向距离对应的奖励
        reward_dy = (10 - dy) * 0.4

        # 前车切入的奖励
        ego_speed_reward = 0
        acc = self.ego.action['acceleration']
        # 如果AV在前车速度大于5的情况下减速，并且与前车的横向距离小于1.5，计算相应的奖励
        if self.ego.velocity > 5 and abs(ego_lateral - vehicle_lateral) < 1.5:
            # 计算AV速度贡献的奖励
            ego_speed_reward += np.clip(abs(acc), 0, 10)
        # 如果AV的纵向坐标大于前车的纵向坐标，并且与前车的横向距离小于1.5，计算相应的奖励
        if (ego_longitudinal > vehicle_longitudinal) and abs(ego_lateral - vehicle_lateral) < 1.5:
            # 计算AV与前车纵向距离对应的奖励
            ego_speed_reward += 5

        # 防止AV与NPC静止不动
        # 如果AV的速度大于等于2，奖励为0；否则，奖励为-5
        if self.ego.velocity >= 2:
            ego_speed_reward += 0
        else:
            ego_speed_reward += -5

        # 计算总的奖励，包括横向距离、纵向距离、速度、夹角和前车切入的贡献
        reward = reward_dx + reward_dy + scaled_speed_reward + heading_reward + ego_speed_reward + (
                    self.time / self.duration) * 0.5

        # 如果前车发生碰撞
        if self.vehicle.crashed:
            # 如果AV（自动驾驶汽车）也发生碰撞，奖励加10
            if self.ego.crashed:
                reward += 10

        # 如果前车不在道路上，奖励为负10
        if not self.vehicle.on_road:
            reward += -10

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
            "velocity": self.vehicle.velocity,
            "crashed": self.vehicle.crashed,
            'offroad': not self.vehicle.on_road,
            "action": action,
            "time": self.time,
            "collision_data": self.before_collision_data
        }

        return (gail_features, adv_feature), reward, terminal, info

    def _reward(self, action):
        return 0

    def save_video(self, imgs: list):

        if self.video_save_name != '' and len(imgs) > 0:
            path = f"data/videos/intersection/{self.video_save_name}.mp4"
            print(f"save video in {path}")
            t = Thread(target=img_2_video, args=(path, imgs))
            t.setDaemon(True)
            t.start()
            t.join()

    # 统计数据步骤之前的函数
    def statistic_data_step_before(self):
        if self.steps == 0:  # 如果步数为0
            self.vehicle_position = self.vehicle.position.copy()
            # 设置车辆位置为当前车辆位置的副本
            if not self.gail:  # 如果不是GAIL模式
                try:
                    self.ego_position = self.ego.position.copy()
                    # 设置自车位置为当前自车位置的副本
                except Exception:
                    print(Exception)
                    # 输出异常信息

    # 统计数据步骤之后的函数
    def statistic_data_step_after(self):
        data2 = self.calculate_ttc_thw(self.vehicle)
        # 计算目标车辆的TTC和THW数据
        data1 = self.calculate_ttc_thw(self.ego)
        # 计算自车的TTC和THW数据

        data1.extend(data2)
        # 将自车和目标车辆的TTC和THW数据合并
        self.TTC_THW.append(data1)
        # 将合并后的数据添加到TTC_THW列表中
        if self.steps >= 1:  # 如果步数大于等于1
            self.distance = np.linalg.norm(self.vehicle_position - self.vehicle.position)
            # 计算目标车辆的位置变化距离
            self.vehicle_position = self.vehicle.position.copy()
            # 更新车辆位置为当前车辆位置的副本
            if not self.gail:  # 如果不是GAIL模式
                self.ego_distance = np.linalg.norm(self.ego_position - self.ego.position)
                # 计算自车的位置变化距离
                self.ego_position = self.ego.position.copy()
                # 更新自车位置为当前自车位置的副本

    def _simulate(self, action):
        """
        执行几步模拟，使用计划的轨迹
        """
        self.distance = 0.0  # 总行驶距离
        self.ego_distance = 0.0  # 自车行驶距离
        trajectory_features = []  # 轨迹特征
        self.TTC_THW = []  # TTC_THW（Time-to-Collision和Time Headway）特征
        self.before_collision_data = {}  # 碰撞前数据
        for i in range(1):
            if self.run_step > 0:
                self._create_bv_vehicles(self.reset_time, 10, self.run_step)  # 创建后车车辆

            self.statistic_data_step_before()  # 统计步骤之前的数据
            self.road.act(self.run_step)  # 执行道路行为

            if (self.gail and action is not None) or (action is not None and self.open_adversarial):
                self.vehicle.action['steering'] = action[0]  # 设置车辆的转向动作
                self.vehicle.action['acceleration'] = action[1]  # 设置车辆的加速度动作

            self.before_collision_data = {
                'av_speed': self.vehicle.velocity,  # 主要车速度
                'av_heading': self.vehicle.heading,  # 主要车航向
            }
            if not self.gail:
                self.before_collision_data.update(
                    {
                        'bv_speed': self.ego.velocity,  # 后车速度
                        'bv_heading': self.ego.heading,  # 后车航向
                        'dx': self.vehicle.position[0] - self.ego.position[0],  # 后车和自车的x轴距离
                        'dy': self.vehicle.position[1] - self.ego.position[1]  # 后车和自车的y轴距离
                    }
                )

            self.road.step(1 / self.SIMULATION_FREQUENCY)  # 道路模拟一步
            self.time += 1  # 时间+1
            self.run_step += 1  # 运行步骤+1
            self.steps += 1  # 步骤+1
            gail_features = self.gail_features()  # 获取GAIL特征
            if not self.gail:
                adv_feature = self.adv_features()  # 获取对抗特征
            else:
                adv_feature = None

            self.statistic_data_step_after()  # 统计步骤之后的数据
            action = [self.vehicle.action['steering'], self.vehicle.action['acceleration']]  # 车辆的转向和加速度动作列表
            self._automatic_rendering()  # 自动渲染显示

            # 停止于终止状态
            if self.done or self._is_terminal():
                break

            self._clear_vehicles()  # 清除车辆

            if self._is_terminal():
                break

        self.enable_auto_render = False

        return (gail_features, adv_feature)

    def _is_terminal(self):
        """
        如果自车碰撞、离开道路或时间已到，则该场景结束
        """
        try:
            flag = self.time >= self.duration or len(self.road.vehicles) < 2 or self.vehicle is None or (
                        self.ego not in self.road.vehicles and not self.gail) or self.vehicle.crashed or \
                   (self.vehicle.linear is not None and
                    self.vehicle.linear.local_coordinates(self.vehicle.position)[
                        0] >= self.vehicle.linear.length) or \
                   (not self.gail and (self.ego is None or self.ego.linear is not None and
                                       self.ego.linear.local_coordinates(self.ego.position)[
                                           0] >= self.ego.linear.length))
        except Exception as e:
            pass

        return flag

    def _clear_vehicles(self) -> None:
        is_leaving = lambda vehicle: (self.run_step >= (
                    vehicle.planned_trajectory.shape[0] + vehicle.start_step - 1) and not vehicle.IDM) or \
                                     (vehicle.IDM and vehicle.linear.local_coordinates(vehicle.position)[
                                         0] >= vehicle.linear.length)
        vehicles = []
        for vh in self.road.vehicles:
            try:
                if vh in self.controlled_vehicles or not is_leaving(vh):
                    vehicles.append(vh)
            except Exception as e:
                print(e)

        self.road.vehicles = vehicles
    # self.road.vehicles = [vehicle for vehicle in self.road.vehicles if
    #                       vehicle in self.controlled_vehicles or not (is_leaving(vehicle) or vehicle.route is None)]

    def gail_features(self):
        obs = self.observation.observe()  # 获取观测数据
        lane_index = self.road.network.get_closest_lane_index(self.vehicle.position, self.vehicle.heading)  # 获取最近车道的索引
        lane = self.road.network.get_lane(lane_index)  # 获取车道对象
        longitudinal, lateral = lane.local_coordinates(self.vehicle.position)  # 获取车辆在车道上的纵向和横向坐标
        lane_w = lane.width_at(longitudinal)  # 获取车道在该纵向坐标处的宽度
        lane_offset = lateral  # 车辆在车道上的横向偏移量
        lane_heading = lane.heading_at(longitudinal)  # 车辆在车道上的航向角

        features = [lane_offset, lane_heading, lane_w]  # 特征列表，包括车辆在车道上的偏移量、航向角和车道宽度
        features += obs[0][2:5].tolist()  # 将观测数据中的一些特征值添加到特征列表中
        for vb in obs[1:]:
            core = obs[0] - vb  # 求出车辆自身特征与其他车辆特征的差值（核心特征）
            features += core[:5].tolist()  # 将核心特征添加到特征列表中
        # print(len(features), features)
        return features  # 返回特征列表

    def adv_features(self):
        obs = self.observation.observe()  # 获取观测数据
        lane_index = self.road.network.get_closest_lane_index(self.vehicle.position, self.vehicle.heading)  # 获取最近车道的索引
        lane = self.road.network.get_lane(lane_index)  # 获取车道对象
        longitudinal, lateral = lane.local_coordinates(self.vehicle.position)  # 获取车辆在车道上的纵向和横向坐标
        lane_w = lane.width_at(longitudinal)  # 获取车道在该纵向坐标处的宽度
        lane_offset = lateral  # 车辆在车道上的横向偏移量
        lane_heading = lane.heading_at(longitudinal)  # 车辆在车道上的航向角
        vehicle_fea = obs[0][:5].tolist()  # 获取观测数据中第一个车辆的特征值

        acc = self.ego.action['acceleration']  # 获取自车的加速度
        features = [lane_heading, lane_w, lane_offset, (lane_heading - vehicle_fea[4]) / math.pi * 180, vehicle_fea[4],
                    acc, self.ego.velocity,
                    self.vehicle.velocity]  # 特征列表，包括车辆在车道上的航向角、车道宽度、横向偏移量、航向角之差、车辆自身的航向角、加速度以及自车和其他车辆的速度
        ego_fea = [vehicle_fea[0] - self.ego.to_dict()['x'], vehicle_fea[1] - self.ego.to_dict()['y'],
                   vehicle_fea[2] - self.ego.to_dict()['vx'],
                   vehicle_fea[3] - self.ego.to_dict()['vy'],
                   self.ego.to_dict()['heading']]  # 自车特征，包括自车与第一个车辆在x、y、vx、vy方向上的位置和自车的航向角
        features.extend(ego_fea)  # 将自车特征添加到特征列表中

        adv_features = np.array(features)  # 将特征列表转换为numpy数组

        # obs = self.observation.observe()
        # lane_index = self.road.network.get_closest_lane_index(self.vehicle.position, self.vehicle.heading)
        # lane = self.road.network.get_lane(lane_index)
        # longitudinal, lateral = lane.local_coordinates(self.vehicle.position)
        # lane_w = lane.width_at(longitudinal)
        # lane_offset = lateral
        # lane_heading = lane.heading_at(longitudinal)
        # vehicle_fea = obs[0][:5].tolist()
        # acc = self.ego.action['acceleration']
        # features = [lane_heading, lane_w, lane_offset, (lane_heading - vehicle_fea[4]) / math.pi * 180, vehicle_fea[4], acc]
        # ego_fea = [vehicle_fea[0] - self.ego.to_dict()['x'], vehicle_fea[1] - self.ego.to_dict()['y'],
        #            vehicle_fea[2] - self.ego.to_dict()['vx'],
        #            vehicle_fea[3] - self.ego.to_dict()['vy'], self.ego.to_dict()['heading']]
        # features.extend(ego_fea)
        #
        # # exp 2
        # # for i in range(5):
        # #     fea = obs[i][:5].tolist()
        # #     features.extend(fea)
        #
        # adv_features = np.array(features)

        return adv_features  # 返回特征数组

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