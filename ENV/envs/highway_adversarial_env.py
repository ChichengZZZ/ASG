from __future__ import division, print_function, absolute_import
from gym.envs.registration import register
import numpy as np

from NGSIM_env import utils
from NGSIM_env.envs.common.observation import observation_factory
from NGSIM_env import utils
from NGSIM_env.envs.common.abstract import AbstractEnv
from NGSIM_env.road.road import Road, RoadNetwork
from NGSIM_env.vehicle.behavior import IDMVehicle
from NGSIM_env.vehicle.humandriving import HumanLikeVehicle, NGSIMVehicle
from NGSIM_env.road.lane import LineType, StraightLane
from NGSIM_env.utils import *
import pickle
import os


class HighwayAdEnv(AbstractEnv):
    """
    A highway driving environment with NGSIM data.
    ego 为 Agent， vehicle 为被测AV
    """

    def __init__(self, scene, path, period, vehicle_id, IDM=False, adversarial_config=None):
        # f = open('NGSIM_env/data/trajectory_set.pickle', 'rb')
        self.path = path
        f = open(self.path, 'rb')
        self.trajectory_set = pickle.load(f)  # 从指定路径加载NGSIM数据集
        f.close()
        # self.trajectory_set = None
        self.vehicle_id = vehicle_id
        self.scene = scene
        # self.trajectory_set = build_trajecotry(scene, period, vehicle_id)
        self.ego_length = self.trajectory_set['ego']['length'] / 3.281  # 获取ego车辆的长度信息
        self.ego_width = self.trajectory_set['ego']['width'] / 3.281  # 获取ego车辆的宽度信息
        self.ego_trajectory = self.trajectory_set['ego']['trajectory']  # 获取ego车辆的轨迹信息
        self.duration = 200 #len(self.ego_trajectory) - 3  # 设置环境的持续时间
        self.surrounding_vehicles = list(self.trajectory_set.keys())  # 获取所有其他车辆的ID
        self.surrounding_vehicles.pop(0)  # 除去ego车辆的ID
        self.run_step = 0  # 初始化步数
        self.human = False  # 是否使用人类驾驶员的目标
        self.IDM = IDM  # 是否使用IDM模型
        self.gail = False  # 是否进行GAIL训练

        self.adversarial_sconfig = adversarial_config if adversarial_config is not None else { 'distance_limit_mi': 15, 'time_limit_steps': 20 }  # 设置对抗模型的配置参数
        super(HighwayAdEnv, self).__init__()

    def t_init(self, path):
        self.path = path
        # print(11, self.path)
        f = open(self.path, 'rb')
        self.trajectory_set = pickle.load(f)  # 从指定路径加载NGSIM数据集
        f.close()
        # self.trajectory_set = None
        # self.trajectory_set = build_trajecotry(scene, period, vehicle_id)
        self.ego_length = self.trajectory_set['ego']['length'] / 3.281  # 获取ego车辆的长度信息
        self.ego_width = self.trajectory_set['ego']['width'] / 3.281  # 获取ego车辆的宽度信息
        self.ego_trajectory = self.trajectory_set['ego']['trajectory']  # 获取ego车辆的轨迹信息
        self.duration = 100  # len(self.ego_trajectory) - 3  # 设置环境的持续时间
        self.surrounding_vehicles = list(self.trajectory_set.keys())  # 获取所有其他车辆的ID
        self.surrounding_vehicles.pop(0)  # 除去ego车辆的ID

    def process_raw_trajectory(self, trajectory):
        trajectory = np.array(trajectory)
        for i in range(trajectory.shape[0]):
            x = trajectory[i][0] - 6
            y = trajectory[i][1]
            speed = trajectory[i][2]
            trajectory[i][0] = y / 3.281  # 坐标变换
            trajectory[i][1] = x / 3.281  # 坐标变换
            trajectory[i][2] = speed / 3.281  # 坐标变换

        return trajectory  # 返回处理后的轨迹数据

    def default_config(self):
        config = super().default_config()  # 调用父类的默认配置
        config.update({
            "observation": {
                "type": "Kinematics",
                'see_behind': True,
                "features": ["x", 'y', "vx", 'vy', 'heading', 'vehicle_id'],
                "normalize": False,
                "absolute": True,
                "vehicles_count": 11},
            "vehicles_count": 10,
            "show_trajectories": True,
            "screen_width": 800,
            "screen_height": 300,
            "simulation_frequency": 15,
            "bv_collision_reward": -10,  # The reward received when colliding with a bv vehicle.
            "ego_collision_reward": 10,  # The reward received when colliding with ego vehicle.
            # "render_agent": True,
            # "offscreen_rendering": False, #os.environ.get("OFFSCREEN_RENDERING", "0") == "1",
            # "manual_control": False,
            # "real_time_rendering": False
        })

        return config  # 返回配置信息

    def reset(self, human=False, reset_time=1):
        '''
        Reset the environment at a given time (scene) and specify whether use human target
        '''
        self.human = human  # 是否使用人类驾驶员的目标
        self._create_road()  # 创建道路环境
        self._create_vehicles(reset_time)  # 创建车辆
        self.steps = 0  # 初始化步数

        return super(HighwayAdEnv, self).reset()  # 调用父类的重置方法

    def _create_road(self):
        """
        Create a road composed of NGSIM road network
        """
        net = RoadNetwork()  # 创建一个空的道路网络
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE  # 道路线类型

        if self.scene == 'us-101':
            length = 2150 / 3.281  # m
            width = 12 / 3.281  # m
            ends = [0, 560 / 3.281, (698 + 578 + 150) / 3.281, length]

            # first section
            line_types = [[c, n], [s, n], [s, n], [s, n], [s, c]]
            for lane in range(5):
                origin = [ends[0], lane * width]
                end = [ends[1], lane * width]
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
            0.基于强化学习的控制模型每次控制对抗NPCn秒，n=2。
            1.如果被测AV与对抗NPC距离超过D米，那么需要将Agent替换为AV距离最近的背景车辆。D=10
            2.如果选择失败，NPC则由IDM控制
        """
        # 检查车辆是否已崩溃
        if self.vehicle.crashed:
            return

        # 根据当前是否开启对抗模式，更新相关统计信息
        if self.open_adversarial_function:
            self.adversarial_time += 1
        else:
            self.no_adversarial_time += 1

        # 计算被测车辆与对抗NPC之间的距离
        dis = float(np.linalg.norm(self.vehicle.position - self.ego.position))
        vehicle = None

        # 判断是否需要改变对抗车辆
        if self.ego.crashed or (not self.ego.on_road) or (
                self.adversarial_time >= self.adversarial_sconfig['time_limit_steps']) or \
                (dis >= self.adversarial_sconfig['distance_limit_mi']):
            self.open_adversarial_function = False
            self.adversarial_time = 0
            dis = 1e6

            # 根据规定距离，获取附近的车辆
            neighbours = self.road.close_vehicles_to(self.vehicle, self.adversarial_sconfig['distance_limit_mi'], 5,
                                                     sort=True)
            lane = self.vehicle.lane_index

            # 如果被测车辆已崩溃或离开道路
            if self.ego.crashed or not self.ego.on_road:
                self.ego.color = (255, 100, 100)
                self.ego.velocity = 0

            # 遍历附近车辆，找到最近的可替换车辆
            for v in neighbours:
                if v.crashed or not v.on_road:
                    continue

                if abs(v.lane_index[2] - lane[2]) < 2:
                    tmp_dis = np.linalg.norm(v.position - self.vehicle.position)
                    if tmp_dis < dis:
                        dis = tmp_dis
                        vehicle = v

            # 如果距离最近的车辆是被测车辆，根据时间和距离判断是否进行对抗
            if vehicle is self.ego:
                if self.no_adversarial_time >= self.adversarial_sconfig['time_limit_steps'] and \
                        (float(np.linalg.norm(self.vehicle.position - self.ego.position)) < self.adversarial_sconfig['distance_limit_mi']):
                    self.ego.color = (0, 255, 0)
                    self.adversarial_time = 0
                    self.no_adversarial_time = 0
                    self.open_adversarial_function = True
                else:
                    self.ego.color = (100, 200, 255)
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
                    self.ego.color = (100, 200, 255)
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
                self.ego.color = (100, 200, 255)
                # self.road.vehicles.append(self.ego)
                self.adversarial_time = self.adversarial_sconfig['time_limit_steps']
            else: # 发生碰撞
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
                    neighbours[0].control_by_agent = True
                    self.ego = neighbours[0]
                    self.ego.color = (100, 200, 255)
                    self.adversarial_time = self.adversarial_sconfig['time_limit_steps']
                else:
                    self.done = True
                    return

        if self.adversarial_sconfig['time_limit_steps'] <= 0:
            self.ego.color = (100, 200, 255)
            self.open_adversarial_function = False

    def _create_vehicles(self, reset_time):
        """
        Create ego vehicle and NGSIM vehicles and add them on the road.
        创建自车和NGSIM车辆，并将它们添加到道路上。
        """
        whole_trajectory = self.process_raw_trajectory(self.ego_trajectory)
        # 对整个轨迹进行处理
        ego_trajectory = whole_trajectory[reset_time:]
        # 获取自车的轨迹
        ego_acc = (whole_trajectory[reset_time][2] - whole_trajectory[reset_time - 1][2]) / 0.1
        # 获取自车的加速度
        self.ego = HumanLikeVehicle.create(self.road, self.vehicle_id, ego_trajectory[0][:2], self.ego_length,
                                           self.ego_width,
                                           ego_trajectory, acc=ego_acc, velocity=ego_trajectory[0][2],
                                           human=self.human, IDM=True)
        # 创建自车，并设置车辆属性，如位置、长度、宽度等
        self.road.vehicles.append(self.ego)
        # 将自车添加到道路车辆列表中

        for veh_id in self.surrounding_vehicles:
            other_trajectory = self.process_raw_trajectory(self.trajectory_set[veh_id]['trajectory'])[reset_time:]

            self.road.vehicles.append(NGSIMVehicle.create(self.road, veh_id, other_trajectory[0][:2],
                                                          self.trajectory_set[veh_id]['length'] / 3.281,
                                                          self.trajectory_set[veh_id]['width'] / 3.281,
                                                          ngsim_traj=other_trajectory, velocity=other_trajectory[0][2]))
        # 创建其他周围车辆，并设置车辆属性，如位置、长度、宽度等，并将它们添加到道路车辆列表中

        self.open_adversarial_function = False
        # 初始化敌对功能属性为False
        self.adversarial_time = 0
        self.no_adversarial_time = 0
        lane = self.ego.lane_index
        dis = 1e6
        self.vehicle = 0
        # 初始化车辆对象为0
        self.neighbours = self.road.close_vehicles_to(self.ego, 50, 5, sort=True)
        # 获取与自车邻近的车辆列表
        for v in self.neighbours:
            if abs(v.lane_index[2] - lane[2]) < 2:
                tmp_dis = np.linalg.norm(v.position - self.ego.position)
                if tmp_dis < dis:
                    self.open_adversarial_function = True
                    dis = tmp_dis
                    self.vehicle = v
                    self.vehicle.color = (0, 0, 255)
        # 如果与自车邻近的车辆距离小于阈值，则设置敌对功能属性为True，并更新敌对车辆为距离最近的车辆

        if not self.vehicle or self.adversarial_sconfig['time_limit_steps'] <= 0:
            self.open_adversarial_function = False
            self.ego.IDM = True
            self.vehicle = self.neighbours[0]
            self.vehicle.color = (0, 0, 255)
        # 如果没有敌对车辆或者敌对车辆的时间限制步数小于等于0，则将敌对功能属性设置为False，并更新敌对车辆为邻近车辆列表的第一个车辆

        self.vehicle.is_ego = True
        self.vehicle.color = (0, 0, 255)
        self.l2_init = np.linalg.norm(self.ego.position - self.vehicle.position)
        # 设置敌对车辆为自车，并计算自车与敌对车辆的初始欧式距离

        self.v_sum = len(self.road.vehicles)
        spd_sum = 0
        for v in self.road.vehicles:
            spd_sum += v.velocity
        self.spd_mean = spd_sum / self.v_sum
        # 计算道路上所有车辆的平均速度

    def lane_offset(self):
        obs = self.observation.observe()
        # 获取观测数据
        lane_w = 12 / 3.281
        # 设置车道宽度
        lane_offset = obs[0][1] - lane_w * self.ego.lane_index[2]
        # 计算车道偏移量
        return abs(lane_offset)
        # 返回车道偏移量的绝对值

    def _reward_normal(self, action=None) -> float:
        """
        奖励定义为鼓励高速驾驶在最右侧车道上，并避免碰撞。
        :param action: 最后执行的动作
        :return: 对应的奖励
        """
        scaled_speed = utils.lmap(self.ego.velocity, [10, 20], [0, 1])  # 缩放自车速度，将其限制在0到1之间
        # 如果自车发生碰撞，则减去1
        reward = -1 * self.ego.crashed \
        + 0.5 * np.clip(scaled_speed, 0, 1) + 0.5 * self.time / self.duration  # 添加缩放后的速度和时间与持续时间之比

        reward = utils.lmap(reward, [-1, 0.5 + 0.5 * self.time / self.duration],
                            [0, 1])  # 将奖励从[-1, 0.5+0.5*self.time/self.duration]映射到[0, 1]之间
        reward = 0 if not self.ego.on_road else reward  # 如果自车不在道路上，则将奖励设置为0
        return reward

    def _reward(self, obs) -> float:
        rewards = []
        for vb in obs[1:2]:
            reward = 0
            distance = self.vehicle.lane_distance_to(self.ego)  # 计算自车和观测车辆之间的距离
            # print(distance)
            other_projected_speed = self.vehicle.velocity * np.dot(self.vehicle.direction,
                                                                   self.ego.direction)  # 计算观测车辆的投影速度
            # print(self.ego.speed - other_projected_speed)
            rr = self.ego.velocity - other_projected_speed  # 计算自车和观测车辆之间的相对速度
            ttc = distance / utils.not_zero(rr)  # 计算碰撞前估计时间（TTC）
            # print('ttc', ttc)
            if np.sign(distance * rr) < 0:  # 如果距离和相对速度的符号相反
                if distance < 0:
                    reward = 1 / abs(ttc) * 6  # 基于TTC的倒数乘以6来计算奖励
                else:
                    reward = ttc / 5  # 基于TTC除以5来计算奖励
            else:
                reward = -10  # 如果距离和相对速度的符号相同，则奖励设为-10
            # if distance < -2:
            #     reward += 12
            if self.vehicle.crashed:  # 如果观测车辆发生碰撞
                if self.ego.crashed and distance < -1:
                    reward += self.config["ego_collision_reward"]  # 添加与自车碰撞的奖励
                else:
                    reward += self.config["bv_collision_reward"]  # 添加与观测车辆碰撞的奖励
            if abs(self.ego.lane_index[2] - self.vehicle.lane_index[2]) > 1:  # 如果自车和观测车辆之间的车道索引差大于1
                reward += -5  # 减去5作为惩罚
            rewards.append(reward)  # 将奖励添加到奖励列表中
        reward = -10 if not self.vehicle.on_road else max(rewards)  # 如果观测车辆不在道路上，则将奖励设置为-10；否则，选择最大的奖励
        return reward

    def _reward_adv(self, action=None) -> float:

        reward = 0
        # l2 = np.linalg.norm(self.ego.position - self.vehicle.position)
        # rate = (self.l2_init-l2)/(self.l2_init)
        # reward += np.clip(rate, -1, 1)
        # if abs(self.ego.lane_index[2] - self.vehicle.lane_index[2]) > 1:
        #     reward = -1
        # print(self.l2_init, l2)
        # print('rate', rate)
        # reward += np.clip(10 / l2, 0, 2) - 1 # np.linalg.norm(self.ego.position - self.vehicle.position)

        # scaled_speed = utils.lmap(self.vehicle.velocity, [10, 20], [0, 1])
        # if self.ego.crashed:
        #     reward += 1

        l2 = np.linalg.norm(self.ego.position - self.vehicle.position)  # 计算自车和观测车辆之间的欧几里德距离
        rate = (self.l2_init - l2) / (self.l2_init)  # 计算距离变化率
        reward += np.clip(rate, -1, 1)  # 将限制在-1和1之间的距离变化率添加到奖励中
        if abs(self.ego.lane_index[2] - self.vehicle.lane_index[2]) > 1:  # 如果自车和观测车辆之间的车道索引差大于1
            reward += -1  # 减去1以作为惩罚
        if self.ego.crashed:
            if self.vehicle.crashed:
                reward += 1
            else:
                reward += -2
            # print(np.linalg.n
            # orm(self.ego.position - self.vehicle.position))
            # print(self.vehicle.lane_distance_to(self.ego))

            # reward += 0.5 * self.time / self.duration
            # reward = utils.lmap(reward, [-2, 2], [-1, 1])
        reward = -2 if not self.ego.on_road else reward
        return reward

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
            "velocity": self.ego.velocity,
            "crashed": self.ego.crashed,
            'offroad': not self.ego.on_road,
            "action": action,
            "time": self.time
        }

        return (gail_features, adv_fea), reward, terminal, info

    def statistic_data_step_before(self):
        # 统计数据，每步之前的操作
        if self.steps == 0:
            try:
                self.vehicle_position = self.ego.position.copy()
                # 尝试获取机动车的位置信息，并赋值给self.vehicle_position
            except:
                pass
            if not self.gail:
                self.ego_position = self.vehicle.position.copy()
                # 如果不是gail模式，则获取自车的位置信息，并赋值给self.ego_position

    def statistic_data_step_after(self):
        # 统计数据，每步之后的操作
        if self.open_adversarial_function or self.adversarial_sconfig["time_limit_steps"] == 0:
            # 如果开启了对抗功能或者对抗配置的time_limit_steps为0
            data = self.calculate_thw_and_ttc(self.ego)
            # 计算自车的THW和TTC
            data.extend(self.calculate_thw_and_ttc(self.vehicle))
            # 计算机动车的THW和TTC
            self.TTC_THW.append(data)
            # 将THW和TTC追加到TTC_THW列表中
            if self.steps >= 1:
                self.distance = np.linalg.norm(self.vehicle_position - self.ego.position)
                # 如果步数大于等于1，则计算机动车和自车之间的距离
                self.ego_distance = np.linalg.norm(self.ego_position - self.vehicle.position)
                # 计算自车和机动车之间的距离
        else:
            self.ego_distance = 0
            # 否则，将自车和机动车之间的距离置0
            self.distance = 0
            # 将机动车和自车之间的距离置0
            self.TTC_THW.append([100, 100, 100, 100])
            # 将[100, 100, 100, 100]追加到TTC_THW列表中

        if self.ego is not None:
            self.vehicle_position = self.ego.position.copy()
            # 如果自车存在，则将自车的位置信息赋值给self.vehicle_position

        self.ego_position = self.vehicle.position.copy()
        # 将机动车的位置信息赋值给self.ego_position

    def _simulate(self, action):
        # 模拟函数
        """
        Perform several steps of simulation with the planned trajectory
        执行计划轨迹的模拟步骤
        """
        trajectory_features = []
        # 轨迹特征
        self.distance = 0.0
        # 距离置0
        self.ego_distance = 0.0
        # 自车和机动车之间的距离置0
        # T = action[2] if action is not None else 5
        self.TTC_THW = []
        # THW和TTC设置为空列表
        for i in range(1):
            # for循环，range(1)表示循环一次
            # if self.steps > 0:
            #     self.change_adversarial_vehicle()
            if i == 0:
                if action is not None:  # sampled goal 若action不为空，则表示采样目标
                    pass
                    # self.vehicle.trajectory_planner(action[0], action[1], action[2])
                else:  # human goal 如果action为空，则表示人类目标
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

            self.statistic_data_step_before()
            # 统计数据，每步之前的操作
            self.road.act(self.run_step)
            # 路口行为
            if self.open_adversarial_function and self.adversarial_sconfig['time_limit_steps'] != 0:
                self.ego.action['steering'] = action[0]
                # 如果开启对抗功能且对抗配置的time_limit_steps不为0，则将action的第一个元素赋值给自车的转向动作
                self.ego.action['acceleration'] = action[1]
                # 将action的第二个元素赋值给自车的加速度动作
                self.ego.color = (0, 255, 0)
                # 将自车的颜色设置为(0, 255, 0)
            # print(self.vehicle.action)
            self.road.step(1 / self.SIMULATION_FREQUENCY)
            # 路口步骤
            self.time += 1
            # 时间加1
            self.run_step += 1
            # 运行步骤加1
            self.steps += 1
            # 步骤加1
            features_adv = self.adv_features()
            # 对抗特征

            action = [self.ego.action['steering'], self.ego.action['acceleration']]
            # 将自车的转向动作和加速度动作赋值给action
            # print(features, action)
            # features += action
            self.statistic_data_step_after()
            # 统计数据，每步之后的操作

            # self.nearest = self.road.close_vehicles_to(self.ego, 50, 10, sort=True)
            # print(self.vehicle)
            # print(self.nearest)
            # if self.nearest:
            #     for v in self.nearest:
            #         # print(v.position)
            #         v.color = (255, 0, 0)

            self._automatic_rendering()
            # 自动渲染

            # for v in self.road.vehicles:
            #     if hasattr(v, 'color'):
            #         delattr(v, 'color')
            self.change_adversarial_vehicle()
            # 改变对抗车辆

            # Stop at terminal states
            # 在终端状态时停止
            if self.done or self._is_terminal():
                break

        self.enable_auto_render = False
        # 自动渲染设置为False

        return None, features_adv
        # 返回None和对抗特征

    def adv_features(self):

        # 获取当前车辆所在车道的索引
        lane_index = self.road.network.get_closest_lane_index(self.ego.position, self.ego.heading)
        # 根据车道索引获取车道对象
        lane = self.road.network.get_lane(lane_index)
        # 获取车辆在车道上的纵向和横向坐标
        longitudinal, lateral = lane.local_coordinates(self.ego.position)
        # 获取车道在当前纵向位置上的宽度
        lane_w = lane.width_at(longitudinal)
        # 获取车道在当前纵向位置上的横向偏移量
        lane_offset = lateral
        # 获取车道在当前纵向位置上的方向
        lane_heading = lane.heading_at(longitudinal)

        # 获取当前车辆和前车的相关特征值
        features = [lane_heading, lane_w, lane_offset, self.ego.heading]
        ego_fea = [self.ego.to_dict()['x'] - self.vehicle.to_dict()['x'],
                   self.ego.to_dict()['y'] - self.vehicle.to_dict()['y'],
                   self.ego.to_dict()['vx'] - self.vehicle.to_dict()['vx'],
                   self.ego.to_dict()['vy'] - self.vehicle.to_dict()['vy'], self.vehicle.to_dict()['heading']]
        features.extend(ego_fea)

        # 构造高级特征向量
        adv_features = np.array(features)

        return adv_features

    def calculate_thw_and_ttc(self, vehicle):
        # 初始化最小THW和TTC值
        THWs = [100]
        TTCs = [100]
        for v in self.road.vehicles:
            if v is vehicle:
                continue

            # 如果前车在当前车道前方且与当前车辆间的横向距离小于当前车辆的宽度，且当前车辆的速度大于等于1，则进行计算
            if v.position[0] > vehicle.position[0] and abs(
                    v.position[1] - vehicle.position[1]) < vehicle.WIDTH and vehicle.velocity >= 1:
                v_speed = v.velocity * np.cos(v.heading)
                vehicle_speed = vehicle.velocity * np.cos(vehicle.heading)

                # 计算THW值
                if vehicle_speed > 0:
                    THW = (v.position[0] - vehicle.position[0]) / utils.not_zero(vehicle_speed)
                else:
                    THW = 100

                # 计算TTC值
                if v_speed > vehicle_speed:
                    TTC = 100
                else:
                    TTC = (v.position[0] - vehicle.position[0]) / utils.not_zero(vehicle_speed - v_speed)

                THWs.append(THW)
                TTCs.append(TTC)

        # 返回最小的TTC和THW值
        return [min(TTCs), min(THWs)]

    def _is_terminal(self):
        """
        判断当前状态是否为终止状态，终止条件包括：时间超过限制、道路上的车辆数小于2、车辆超出道路限制、车辆碰撞、车辆离开道路。
        """

        return self.time >= self.duration or len(self.road.vehicles) < 2 or self.vehicle.position[0] >= 2150 / 3.281 or \
            self.ego.position[0] >= 2150 / 3.281 or self.vehicle.crashed or not self.vehicle.on_road