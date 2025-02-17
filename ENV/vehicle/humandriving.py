from __future__ import division, print_function
import math
import numpy as np
import matplotlib.pyplot as plt
from NGSIM_env.vehicle.control import ControlledVehicle
from NGSIM_env import utils
from NGSIM_env.vehicle.dynamics import Vehicle
from NGSIM_env.vehicle.behavior import IDMVehicle
from NGSIM_env.vehicle.control import MDPVehicle
from NGSIM_env.vehicle.planner import planner
from NGSIM_env.road.lane import PolyLaneFixedWidth, StraightLane
import numpy as np
from Utils.math import filter_similar_points, init_linear
from scipy.interpolate import UnivariateSpline


"""
高速公路参数标定结果
目标速度 v_0 : 27.31 m/s
安全时距 T: 1.09 s
最大加速度 a:5.99 m/s^2
舒适减速度 b:5.96 m/s^2
安全车距 s_0: 2.91 m
"""


class NGSIMVehicle(IDMVehicle):
    """
    使用NGSIM人类驾驶轨迹。
    """

    # 纵向策略参数
    ACC_MAX = 5.99 # [m/s2]  """最大加速度"""
    COMFORT_ACC_MAX = 5.96 # [m/s2]  """期望最大加速度"""
    COMFORT_ACC_MIN = -5.96 # [m/s2] """期望最大减速度"""
    DISTANCE_WANTED = 2.91 # [m] """与前方车辆的期望距离"""
    TIME_WANTED = 1.09 # [s]  """与前方车辆的期望时间间隔"""
    DELTA = 4.0  # [] """速度项的指数"""
    # # 雨天
    # DISTANCE_WANTED = 2.0 # [m] """与前方车辆的期望距离"""
    # TIME_WANTED = 1 # [s]  """与前方车辆的期望时间间隔"""
    # DELTA = 2  # [] """速度项的指数"""

    # 横向策略参数[MOBIL]
    POLITENESS = 0.1  # 在[0, 1]之间
    # POLITENESS = 0.2  # 在[0, 1]之间
    LANE_CHANGE_MIN_ACC_GAIN = 0.2 # [m/s2]
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 2.0 # [m/s2]
    LANE_CHANGE_DELAY = 1.0  # [s]
    # LANE_CHANGE_DELAY = 2.0  # [s]

    # 驾驶场景
    SCENE = 'us-101'

    def __init__(self, road, position,
                 heading=0,
                 velocity=0,
                 target_lane_index=None,
                 target_velocity=27.31,
                 route=None,
                 enable_lane_change=True, # 只在这里修改
                 timer=None,
                 vehicle_ID=None, v_length=None, v_width=None, ngsim_traj=None):
        super(NGSIMVehicle, self).__init__(road, position, heading, velocity, target_lane_index, target_velocity, route, enable_lane_change, timer)

        self.ngsim_traj = ngsim_traj
        self.traj = np.array(self.position)
        self.vehicle_ID = vehicle_ID
        self.sim_steps = 0
        self.overtaken = False
        self.appear = True if self.position[0] != 0 else False
        self.velocity_history = []
        self.heading_history = []
        self.crash_history = []
        self.overtaken_history = []

        # 车辆长度[m]
        self.LENGTH = v_length
        # 车辆宽度[m]
        self.WIDTH = v_width
        self.control_by_agent = False
        # self.make_linear()

    def make_linear(self):
        self.linear, self.unique_arr = init_linear(self.ngsim_traj, PolyLaneFixedWidth)

    @classmethod
    def create(cls, road, vehicle_ID, position, v_length, v_width, ngsim_traj, heading=0, velocity=15):
        """
        创建一个新的NGSIM车辆。

        :param road: 车辆行驶的道路
        :param vehicle_id: NGSIM车辆ID
        :param position: 车辆在道路上的起始位置
        :param v_length: 车辆长度
        :param v_width: 车辆宽度
        :param ngsim_traj: NGSIM轨迹
        :param velocity: 初始速度[m/s]。如果为None，则随机选择
        :param heading: 初始方向

        :return: 带有NGSIM位置和速度的车辆
        """

        v = cls(road, position, heading, velocity, vehicle_ID=vehicle_ID, v_length=v_length, v_width=v_width, ngsim_traj=ngsim_traj)

        return v

    def act(self):
        """
        当NGSIM车辆被超越时执行一个动作。

        :param action: 动作
        """
        if not self.overtaken:
            return

        if self.crashed:
            return

        action = {}
        front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self)

        # 横向: MOBIL
        self.follow_road()
        if self.enable_lane_change:
            self.change_lane_policy()
        action['steering'] = self.steering_control(self.target_lane_index)

        # 纵向: IDM
        action['acceleration'] = self.acceleration(ego_vehicle=self, front_vehicle=front_vehicle, rear_vehicle=rear_vehicle)
        action['acceleration'] = np.clip(action['acceleration'], -self.ACC_MAX, self.ACC_MAX)
        self.action = action

    def step(self, dt):
        # print('ngsim step start')
        """
        更新NGSIM车辆的状态。
        如果前方车辆过近，则使用IDM模型覆盖NGSIM车辆。
        """

        self.appear = True if self.ngsim_traj[self.sim_steps][0] != 0 else False
        self.timer += dt
        self.sim_steps += 1
        self.heading_history.append(self.heading)
        self.velocity_history.append(self.velocity)
        self.crash_history.append(self.crashed)
        self.overtaken_history.append(self.overtaken)

        if self.control_by_agent:
            self.overtaken = True
            super(NGSIMVehicle, self).step(dt)
            self.traj = np.append(self.traj, self.position, axis=0)
            return

        # 检查是否需要超车
        front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self)
        if front_vehicle is not None and isinstance(front_vehicle, NGSIMVehicle) and front_vehicle.overtaken:
            gap = self.lane_distance_to(front_vehicle)
            desired_gap = self.desired_gap(self, front_vehicle)
        elif front_vehicle is not None and (isinstance(front_vehicle, HumanLikeVehicle) or isinstance(front_vehicle, MDPVehicle)):
            gap = self.lane_distance_to(front_vehicle)
            desired_gap = self.desired_gap(self, front_vehicle)
        else:
            gap = 100
            desired_gap = 50
        # print(gap, desired_gap)
  
        if gap >= desired_gap and not self.overtaken:
            self.position = self.ngsim_traj[self.sim_steps][:2]
            lateral_velocity = (self.ngsim_traj[self.sim_steps+1][1] - self.position[1])/0.1
            heading = np.arcsin(np.clip(lateral_velocity/utils.not_zero(self.velocity), -1, 1))
            self.heading = np.clip(heading, -np.pi/4, np.pi/4)     
            self.velocity = (self.ngsim_traj[self.sim_steps+1][0] - self.position[0])/0.1 if self.position[0] != 0 else 0
            self.target_velocity = self.velocity                    
            self.lane_index = self.road.network.get_closest_lane_index(self.position)
            self.lane = self.road.network.get_lane(self.lane_index)
            # print('if gap >= desired_gap and not self.overtaken')
        elif int(self.ngsim_traj[self.sim_steps][3]) == 0 and self.overtaken:          
            self.position = self.ngsim_traj[self.sim_steps][:2]
            self.velocity = self.ngsim_traj[self.sim_steps][2]
        else:
            self.overtaken = True
            # print('self.overtaken = True')

            # Determine the target lane
            target_lane = int(self.ngsim_traj[self.sim_steps][3])
            if self.SCENE == 'us-101':
                if target_lane <= 5:
                    if 0 < self.position[0] <= 560/3.281:
                        self.target_lane_index = ('s1', 's2', target_lane-1)
                    elif 560/3.281 < self.position[0] <= (698+578+150)/3.281:
                        self.target_lane_index = ('s2', 's3', target_lane-1)
                    else:
                        self.target_lane_index = ('s3', 's4', target_lane-1)
                elif target_lane == 6:
                    self.target_lane_index = ('s2', 's3', -1)
                elif target_lane == 7:
                    self.target_lane_index = ('merge_in', 's2', -1)
                elif target_lane == 8:
                    self.target_lane_index = ('s3', 'merge_out', -1)
            elif self.SCENE == 'i-80':
                if target_lane <= 6:
                    if 0 < self.position[0] <= 600/3.281:
                        self.target_lane_index = ('s1','s2', target_lane-1)
                    elif 600/3.281 < self.position[0] <= 700/3.281:
                        self.target_lane_index = ('s2','s3', target_lane-1)
                    elif 700/3.281 < self.position[0] <= 900/3.281:
                        self.target_lane_index = ('s3','s4', target_lane-1)
                    else:
                        self.target_lane_index = ('s4','s5', target_lane-1)
                elif target_lane == 7:
                    self.target_lane_index = ('s1', 's2', -1)

            super(NGSIMVehicle, self).step(dt)

        self.traj = np.append(self.traj, self.position, axis=0)
        # print('ngsim step end')

    def check_collision(self, other):
        """
        检查与另一辆车的碰撞情况。

        :param other: 另一辆车
        """
        if not self.COLLISIONS_ENABLED or not other.COLLISIONS_ENABLED or self.crashed or other is self:
            return

        # Fast spherical pre-check
        if np.linalg.norm(other.position - self.position) > self.LENGTH:
            return

        # if both vehicles are NGSIM vehicles and have not been overriden
        # if isinstance(self, NGSIMVehicle) and not self.overtaken and isinstance(other, NGSIMVehicle) and not other.overtaken:
        #     return
        # print('111', self, other)

        # Accurate rectangular check
        if utils.rotated_rectangles_intersect((self.position, 0.9*self.LENGTH, 0.9*self.WIDTH, self.heading),
                                              (other.position, 0.9*other.LENGTH, 0.9*other.WIDTH, other.heading)) and self.appear:
            self.velocity = other.velocity = min([self.velocity, other.velocity], key=abs)
            self.crashed = other.crashed = True

    def calculate_human_likeness(self):
        original_traj = self.ngsim_traj[:self.sim_steps + 1, :2]
        ego_traj = self.traj.reshape(-1, 2)
        ADE = np.mean([np.linalg.norm(original_traj[i] - ego_traj[i]) for i in
                       range(ego_traj.shape[0])])  # Average Displacement Error (ADE)
        FDE = np.linalg.norm(original_traj[-1] - ego_traj[-1])  # Final Displacement Error (FDE)

        return (ADE, FDE)


class HumanLikeVehicle(IDMVehicle):
    """
    创建一个类似人类驾驶（真实驾驶）的代理。
    """    
    TAU_A = 0.2 # [s]
    TAU_DS = 0.1 # [s]
    PURSUIT_TAU = 1.5*TAU_DS # [s]
    KP_A = 1 / TAU_A
    KP_HEADING = 1 / TAU_DS
    KP_LATERAL = 1 / 0.2 # [1/s]
    MAX_STEERING_ANGLE = np.pi / 3  # [rad]
    MAX_VELOCITY = 30 # [m/s]

    # Longitudinal policy parameters
    ACC_MAX = 5.99 # [m/s2]  """Maximum acceleration."""
    COMFORT_ACC_MAX = 5.96 # [m/s2]  """Desired maximum acceleration."""
    COMFORT_ACC_MIN = -5.96 # [m/s2] """Desired maximum deceleration."""
    DISTANCE_WANTED = 2.91 # [m] """Desired jam distance to the front vehicle."""
    TIME_WANTED = 1.09 # [s]  """Desired time gap to the front vehicle."""
    DELTA = 4.0  # [] """Exponent of the velocity term."""
    # # 雨天
    # DISTANCE_WANTED = 2.0 # [m] """Desired jam distance to the front vehicle."""
    # TIME_WANTED = 1 # [s]  """Desired time gap to the front vehicle."""
    # DELTA = 2  # [] """Exponent of the velocity term."""

    # Lateral policy parameters [MOBIL]
    POLITENESS = 0.1  # in [0, 1]
    # POLITENESS = 0.2  # in [0, 1]
    LANE_CHANGE_MIN_ACC_GAIN = 0.2 # [m/s2]
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 2.0 # [m/s2]
    LANE_CHANGE_DELAY = 1.0  # [s]
    # LANE_CHANGE_DELAY = 2.0  # [s]

    def __init__(self, road, position,
                 heading=0,
                 velocity=0,
                 acc=0,
                 target_lane_index=None,
                 target_velocity=15, # Speed reference
                 route=None,
                 timer=None,
                 vehicle_ID=None, v_length=None, v_width=None, ngsim_traj=None, human=False, IDM=False):
        super(HumanLikeVehicle, self).__init__(road, position, heading, velocity, target_lane_index, target_velocity, route, timer)

        self.ngsim_traj = ngsim_traj
        self.traj = np.array(self.position)
        self.sim_steps = 0
        self.vehicle_ID = vehicle_ID
        self.planned_trajectory = None
        self.human = human
        self.IDM = IDM
        self.velocity_history = []
        self.heading_history = []
        self.crash_history = []
        self.acc = acc
        self.steering_noise = None
        self.acc_noise = None

        self.LENGTH = v_length # Vehicle length [m]
        self.WIDTH = v_width # Vehicle width [m]

    @classmethod
    def create(cls, road, vehicle_ID, position, v_length, v_width, ngsim_traj, heading=0, velocity=0, acc=0, target_velocity=15, human=False, IDM=False):
        """
        Create a human-like (IRL) driving vehicle in replace of a NGSIM vehicle.
        """
        v = cls(road, position, heading, velocity, acc, target_velocity=target_velocity, 
                vehicle_ID=vehicle_ID, v_length=v_length, v_width=v_width, ngsim_traj=ngsim_traj, human=human, IDM=IDM)
       
        return v

    def make_linear(self):
        self.linear, self.unique_arr = init_linear(self.ngsim_traj, PolyLaneFixedWidth)

    def trajectory_planner(self, target_point, target_speed, time_horizon):
        """
        Plan a trajectory for the human-like (IRL) vehicle.
        """
        s_d, s_d_d, s_d_d_d = self.position[0], self.velocity * np.cos(self.heading), self.acc # Longitudinal
        c_d, c_d_d, c_d_dd = self.position[1], self.velocity * np.sin(self.heading), 0 # Lateral
        target_area, speed, T = target_point, target_speed, time_horizon

        if not self.human:
            target_area += np.random.normal(0, 0.2)

        path = planner(s_d, s_d_d, s_d_d_d, c_d, c_d_d, c_d_dd, target_area, speed, T)
        
        self.planned_trajectory = np.array([[x, y] for x, y in zip(path[0].x, path[0].y)])

        if self.IDM:
            self.planned_trajectory = None

        # if constant velocity:
        #time = np.arange(0, T*10, 1)
        #path_x = self.position[0] + self.velocity * np.cos(self.heading) * time/10
        #path_y = self.position[1] + self.velocity * np.sin(self.heading) * time/10
        #self.planned_trajectory = np.array([[x, y] for x, y in zip(path_x, path_y)])

    def act(self, step):
        if self.planned_trajectory is not None and self.linear is None:
            self.action = {'steering': self.steering_control_hum(self.planned_trajectory, step),
                           'acceleration': self.velocity_control(self.planned_trajectory, step)}
        elif self.IDM:
            super(HumanLikeVehicle, self).act()
            # print('self.vehicle.action', self.action)
        else:
            return

    def steering_control_hum(self, trajectory, step):
        """
        将车辆转向以跟随给定的轨迹。

        1. 横向位置由比例控制器控制，得到一个横向速度指令
        2. 横向速度指令转换为航向参考值
        3. 航向由比例控制器控制，得到一个航向速率指令
        4. 航向速率指令转换为转向角度

        :param trajectory: 要跟随的轨迹
        :return: 方向盘转角指令 [弧度]
        """
        target_coords = trajectory[step]
        # Lateral position control
        lateral_velocity_command = self.KP_LATERAL * (target_coords[1] - self.position[1])

        # Lateral velocity to heading
        heading_command = np.arcsin(np.clip(lateral_velocity_command/utils.not_zero(self.velocity), -1, 1))
        heading_ref = np.clip(heading_command, -np.pi/4, np.pi/4)

        # Heading control
        heading_rate_command = self.KP_HEADING * utils.wrap_to_pi(heading_ref - self.heading)

        # Heading rate to steering angle
        steering_angle = np.arctan(self.LENGTH / utils.not_zero(self.velocity) * heading_rate_command)
        steering_angle = np.clip(steering_angle, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)

        return steering_angle

    def velocity_control(self, trajectory, step):
        """
        控制车辆的速度。

        使用简单的比例控制器。

        :param trajectory: 要跟随的轨迹 :return: 加速度指令 [m/s2]

        """
        target_velocity = (trajectory[step][0] - trajectory[step-1][0]) / 0.1
        acceleration = self.KP_A * (target_velocity - self.velocity)

        return acceleration

    def step(self, dt):
        self.sim_steps += 1
        self.heading_history.append(self.heading)
        self.velocity_history.append(self.velocity)
        self.crash_history.append(self.crashed)

        super(HumanLikeVehicle, self).step(dt)
        
        self.traj = np.append(self.traj, self.position, axis=0)
    
    def calculate_human_likeness(self):
        original_traj = self.ngsim_traj[:self.sim_steps+1, :2]
        ego_traj = self.traj.reshape(-1, 2)
        ADE = np.mean([np.linalg.norm(original_traj[i] - ego_traj[i]) for i in range(ego_traj.shape[0])]) # Average Displacement Error (ADE)
        FDE = np.linalg.norm(original_traj[-1] - ego_traj[-1]) # Final Displacement Error (FDE)
        
        return (ADE, FDE)


class TrajRecovery(IDMVehicle):
    """
    Create a human-like (IRL) driving agent.
    """
    TAU_A = 0.2  # [s]
    TAU_DS = 0.1  # [s]
    PURSUIT_TAU = 1.5 * TAU_DS  # [s]
    KP_A = 1 / TAU_A
    KP_HEADING = 1 / TAU_DS
    KP_LATERAL = 1 / 0.2  # [1/s]
    MAX_STEERING_ANGLE = np.pi / 3  # [rad]
    MAX_VELOCITY = 30  # [m/s]

    # Longitudinal policy parameters
    ACC_MAX = 5.0  # [m/s2]  """Maximum acceleration."""
    COMFORT_ACC_MAX = 3.0  # [m/s2]  """Desired maximum acceleration."""
    COMFORT_ACC_MIN = -3.0  # [m/s2] """Desired maximum deceleration."""
    DISTANCE_WANTED = 1.0  # [m] """Desired jam distance to the front vehicle."""
    TIME_WANTED = 0.5  # [s]  """Desired time gap to the front vehicle."""
    DELTA = 4.0  # [] """Exponent of the velocity term."""
    # # 雨天
    # DISTANCE_WANTED = 2.0  # [m] """Desired jam distance to the front vehicle."""
    # TIME_WANTED = 1  # [s]  """Desired time gap to the front vehicle."""
    # DELTA = 2  # [] """Exponent of the velocity term."""

    # Lateral policy parameters [MOBIL]
    POLITENESS = 0.1  # in [0, 1]
    # POLITENESS = 0.2  # in [0, 1]
    LANE_CHANGE_MIN_ACC_GAIN = 0.2  # [m/s2]
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 2.0  # [m/s2]
    LANE_CHANGE_DELAY = 1.0  # [s]

    # LANE_CHANGE_DELAY = 2.0  # [s]

    def __init__(self, road, position,
                 heading=0,
                 velocity=0,
                 acc=0,
                 target_lane_index=None,
                 target_velocity=15,  # Speed reference
                 route=None,
                 timer=None,
                 vehicle_ID=None, v_length=None, v_width=None, ngsim_traj=None, human=False, IDM=False):
        super(TrajRecovery, self).__init__(road, position, heading, velocity, target_lane_index, target_velocity,
                                               route, timer)

        self.ngsim_traj = ngsim_traj
        self.traj = np.array(self.position)
        self.sim_steps = 0
        self.vehicle_ID = vehicle_ID
        self.planned_trajectory = None
        self.planned_heading = None
        self.planned_speed = None
        self.human = human
        self.IDM = IDM
        self.velocity_history = []
        self.heading_history = []
        self.crash_history = []
        self.acc = acc
        self.steering_noise = None
        self.acc_noise = None

        self.LENGTH = v_length  # Vehicle length [m]
        self.WIDTH = v_width  # Vehicle width [m]

    @classmethod
    def create(cls, road, vehicle_ID, position, v_length, v_width, ngsim_traj, heading=0, velocity=0, acc=0,
               target_velocity=15, human=False, IDM=False):
        """
        Create a human-like (IRL) driving vehicle in replace of a NGSIM vehicle.
        """
        v = cls(road, position, heading, velocity, acc, target_velocity=target_velocity,
                vehicle_ID=vehicle_ID, v_length=v_length, v_width=v_width, ngsim_traj=ngsim_traj, human=human, IDM=IDM)

        return v

    def trajectory_planner(self, target_point, target_speed, time_horizon):
        """
        Plan a trajectory for the human-like (IRL) vehicle.
        """
        s_d, s_d_d, s_d_d_d = self.position[0], self.velocity * np.cos(self.heading), self.acc  # Longitudinal
        c_d, c_d_d, c_d_dd = self.position[1], self.velocity * np.sin(self.heading), 0  # Lateral
        target_area, speed, T = target_point, target_speed, time_horizon

        if not self.human:
            target_area += np.random.normal(0, 0.2)

        path = planner(s_d, s_d_d, s_d_d_d, c_d, c_d_d, c_d_dd, target_area, speed, T)

        self.planned_trajectory = np.array([[x, y] for x, y in zip(path[0].x, path[0].y)])

        if self.IDM:
            self.planned_trajectory = None

        # if constant velocity:
        # time = np.arange(0, T*10, 1)
        # path_x = self.position[0] + self.velocity * np.cos(self.heading) * time/10
        # path_y = self.position[1] + self.velocity * np.sin(self.heading) * time/10
        # self.planned_trajectory = np.array([[x, y] for x, y in zip(path_x, path_y)])

    def act(self, step):

        if self.planned_trajectory is not None:
            try:
                if step < self.planned_trajectory.shape[0] - 1:
                    self.next_position = self.planned_trajectory[step+1]

                control_heading, acceleration = self.control_vehicle(self.planned_trajectory[step], self.planned_speed[step], self.planned_heading[step])
                self.action = {'steering': control_heading,
                               'acceleration': acceleration}
            except Exception as e:
                super(TrajRecovery, self).act()

        elif self.IDM:
            super(TrajRecovery, self).act()
            # print('self.vehicle.action', self.action)
        else:
            return

    def control_vehicle(self, next_position, next_speed, next_heading):

        # Heading control
        heading_rate_command = self.KP_HEADING * utils.wrap_to_pi(next_heading - self.heading)

        # Heading rate to steering angle
        steering_angle = np.arctan(self.LENGTH / utils.not_zero(self.velocity) * heading_rate_command)
        # steering_angle = np.clip(steering_angle, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)

        acceleration = 10 * (next_speed - self.velocity)
        return steering_angle, acceleration

    def steering_control_hum(self, trajectory, step):
        """
        将车辆转向以跟随给定的轨迹。

        1. 横向位置由比例控制器控制，得到一个横向速度指令
        2. 横向速度指令转换为航向参考值
        3. 航向由比例控制器控制，得到一个航向速率指令
        4. 航向速率指令转换为转向角度

        :param trajectory: 要跟随的轨迹
        :return: 方向盘转角指令 [弧度]
        """
        target_coords = trajectory[step]
        # Lateral position control
        lateral_velocity_command = self.KP_LATERAL * (target_coords[1] - self.position[1])

        # Lateral velocity to heading
        heading_command = np.arcsin(np.clip(lateral_velocity_command / utils.not_zero(self.velocity), -1, 1))
        heading_ref = np.clip(heading_command, -np.pi / 4, np.pi / 4)

        # Heading control
        heading_rate_command = self.KP_HEADING * utils.wrap_to_pi(heading_ref - self.heading)

        # Heading rate to steering angle
        steering_angle = np.arctan(self.LENGTH / utils.not_zero(self.velocity) * heading_rate_command)
        steering_angle = np.clip(steering_angle, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)

        return steering_angle

    def velocity_control(self, trajectory, step):
        """
        控制车辆的速度。
        使用简单的比例控制器。
        :param trajectory: 要跟随的轨迹 :return: 加速度指令 [m/s2]
        """
        target_velocity = (trajectory[step][0] - trajectory[step - 1][0]) / 0.1
        acceleration = self.KP_A * (target_velocity - self.velocity)

        return acceleration

    def step(self, dt):
        self.sim_steps += 1
        self.heading_history.append(self.heading)
        self.velocity_history.append(self.velocity)
        self.crash_history.append(self.crashed)

        super(TrajRecovery, self).step(dt)

        self.traj = np.append(self.traj, self.position, axis=0)


# class InterActionVehicle(IDMVehicle):
#     """
#     Create a human-like (IRL) driving agent.
#     """
#     TAU_A = 0.2  # [s]
#     TAU_DS = 0.1  # [s]
#     PURSUIT_TAU = 1.5 * TAU_DS  # [s]
#     KP_A = 1 / TAU_A
#     KP_HEADING = 1 / TAU_DS
#     KP_LATERAL = 1 / 0.2  # [1/s]
#     MAX_STEERING_ANGLE = np.pi / 3  # [rad]
#     MAX_VELOCITY = 30  # [m/s]
#     # # Longitudinal policy parameters
#     # ACC_MAX = 5.0 # [m/s2]  """Maximum acceleration."""
#     # COMFORT_ACC_MAX = 3.0 # [m/s2]  """Desired maximum acceleration."""
#     # COMFORT_ACC_MIN = -3.0 # [m/s2] """Desired maximum deceleration."""
#     # DISTANCE_WANTED = 1.0 # [m] """Desired jam distance to the front vehicle."""
#     # TIME_WANTED = 0.5 # [s]  """Desired time gap to the front vehicle."""
#     # DELTA = 4.0  # [] """Exponent of the velocity term."""
#     #
#     # # Lateral policy parameters [MOBIL]
#     # POLITENESS = 0.1  # in [0, 1]
#     # LANE_CHANGE_MIN_ACC_GAIN = 0.2 # [m/s2]
#     # LANE_CHANGE_MAX_BRAKING_IMPOSED = 2.0 # [m/s2]
#     # LANE_CHANGE_DELAY = 1.0  # [s]
#
#     def __init__(self, road, position,
#                  heading=0,
#                  velocity=0,
#                  acc=0,
#                  target_lane_index=None,
#                  target_velocity=15,  # Speed reference
#                  route=None,
#                  timer=None,
#                  vehicle_ID=None, v_length=None, v_width=None, ngsim_traj=None, human=False, IDM=False):
#         super(InterActionVehicle, self).__init__(road, position, heading, velocity, target_lane_index, target_velocity,
#                                                route, timer)
#
#         self.ngsim_traj = ngsim_traj
#         self.traj = np.array(self.position)
#         self.sim_steps = 0
#         self.vehicle_ID = vehicle_ID
#         self.planned_trajectory = None
#         self.planned_heading = None
#         self.planned_speed = None
#         self.human = human
#         self.IDM = IDM
#         self.velocity_history = []
#         self.heading_history = []
#         self.crash_history = []
#         self.position_history = []
#         self.acc = acc
#         self.steering_noise = None
#         self.acc_noise = None
#         self.next_position = None
#         self.MARGIN = 5
#         self.LENGTH = v_length  # Vehicle length [m]
#         self.WIDTH = v_width  # Vehicle width [m]
#         unique_arr = np.unique(ngsim_traj[:, :2].copy(), axis=0)
#         mask = np.where(np.sum(unique_arr, axis=1) == 0, False, True)
#         self.linear = PolyLaneFixedWidth(unique_arr[mask].tolist())
#
#     @classmethod
#     def create(cls, road, vehicle_ID, position, v_length, v_width, ngsim_traj, heading=0, velocity=0, acc=0,
#                target_velocity=15, human=False, IDM=False):
#         """
#         Create a human-like (IRL) driving vehicle in replace of a NGSIM vehicle.
#         """
#         v = cls(road, position, heading, velocity, acc, target_velocity=target_velocity,
#                 vehicle_ID=vehicle_ID, v_length=v_length, v_width=v_width, ngsim_traj=ngsim_traj, human=human, IDM=IDM)
#
#         return v
#
#     def act(self, step):
#         if self.planned_trajectory is not None and not self.IDM:
#             try:
#                 if step < self.planned_trajectory.shape[0] - 1:
#                     self.next_position = self.planned_trajectory[step+1]
#
#                 control_heading, acceleration = self.control_vehicle(self.planned_trajectory[step], self.planned_speed[step], self.planned_heading[step])
#                 self.action = {'steering': control_heading,
#                                'acceleration': acceleration}
#             except Exception as e:
#                 super(InterActionVehicle, self).act()
#
#         elif self.IDM:
#             super(InterActionVehicle, self).act()
#             # print('self.vehicle.action', self.action)
#         else:
#             return
#
#     def control_vehicle(self, next_position, next_speed, next_heading):
#
#         # Heading control
#         heading_rate_command = self.KP_HEADING * utils.wrap_to_pi(next_heading - self.heading)
#
#         # Heading rate to steering angle
#         steering_angle = np.arctan(self.LENGTH / utils.not_zero(self.velocity) * heading_rate_command)
#         # steering_angle = np.clip(steering_angle, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)
#
#         acceleration = 10 * (next_speed - self.velocity)
#         return steering_angle, acceleration
#
#     def fix_angle(self, angle):
#         while angle <= -math.pi:
#             angle += 2 * math.pi
#
#         while angle > math.pi:
#             angle -= 2 * math.pi
#
#         return angle
#
#     def check_collision(self, other):
#         """
#         Check for collision with another vehicle.
#
#         :param other: the other vehicle
#         """
#         if not self.COLLISIONS_ENABLED or not other.COLLISIONS_ENABLED or self.crashed or other is self:
#             return
#
#         # Fast spherical pre-check
#         if np.linalg.norm(other.position - self.position) > (other.LENGTH + self.LENGTH) / 2:
#             return
#
#         # if both vehicles are NGSIM vehicles and have not been overriden
#         # if isinstance(self, NGSIMVehicle) and not self.overtaken and isinstance(other, NGSIMVehicle) and not other.overtaken:
#         #     return
#         # print('111', self, other)
#
#         # Accurate rectangular check
#         if utils.rotated_rectangles_intersect((self.position, 0.9*self.LENGTH, 0.9*self.WIDTH, self.heading),
#                                               (other.position, 0.9*other.LENGTH, 0.9*other.WIDTH, other.heading)):
#             self.velocity = other.velocity = min([self.velocity, other.velocity], key=abs)
#             self.crashed = other.crashed = True
#
#         # if utils.judge_car_collision(float(self.position[0]), float(self.position[1]), self.LENGTH, self.WIDTH, self.heading,
#         #                              float(other.position[0]), float(other.position[1]), other.LENGTH, other.WIDTH, other.heading):
#         #     self.velocity = other.velocity = min([self.velocity, other.velocity], key=abs)
#         #     self.crashed = other.crashed = True
#
#     def step(self, dt):
#         self.sim_steps += 1
#         self.heading_history.append(self.heading)
#         self.velocity_history.append(self.velocity)
#         self.crash_history.append(self.crashed)
#         self.position_history.append(self.position)
#         if not self.IDM:
#             front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self, margin=self.MARGIN)
#             if front_vehicle is not None:
#                 gap = self.lane_distance_to(front_vehicle)
#                 desired_gap = self.desired_gap(self, front_vehicle)
#             else:
#                 gap = 100
#                 desired_gap = 50
#
#             if gap < desired_gap:
#                 self.IDM = True
#                 self.action['steering'] = self.steering_control(self.target_lane_index)
#
#                 # Longitudinal: IDM
#                 self.action['acceleration'] = self.acceleration(ego_vehicle=self, front_vehicle=front_vehicle,
#                                                            rear_vehicle=rear_vehicle)
#                 self.action['acceleration'] = np.clip(self.action['acceleration'], -self.ACC_MAX, self.ACC_MAX)
#
#         super(InterActionVehicle, self).step(dt)
#
#         self.traj = np.append(self.traj, self.position, axis=0)
#
#     def calculate_human_likeness(self):
#         original_traj = self.ngsim_traj[:self.sim_steps + 1, :2]
#         ego_traj = self.traj.reshape(-1, 2)
#         ADE = np.mean([np.linalg.norm(original_traj[i] - ego_traj[i]) for i in
#                        range(ego_traj.shape[0])])  # Average Displacement Error (ADE)
#         FDE = np.linalg.norm(original_traj[-1] - ego_traj[-1])  # Final Displacement Error (FDE)
#
#         return (ADE, FDE)


class InterActionVehicle(IDMVehicle):
    """
    Create a human-like (IRL) driving agent.
    """
    TAU_A = 0.2  # [s]
    TAU_DS = 0.1  # [s]
    PURSUIT_TAU = 1.5 * TAU_DS  # [s]
    KP_A = 1 / TAU_A
    KP_HEADING = 1 / TAU_DS
    KP_LATERAL = 1 / 0.2  # [1/s]
    MAX_STEERING_ANGLE = np.pi / 3  # [rad]
    MAX_VELOCITY = 30  # [m/s]

    def __init__(self, road, position,
                 heading=0,
                 velocity=0,
                 acc=0,
                 target_lane_index=None,
                 target_velocity=15,  # Speed reference
                 route=None,
                 timer=None,
                 vehicle_ID=None, v_length=None, v_width=None, ngsim_traj=None, human=False, IDM=False):
        super(InterActionVehicle, self).__init__(road, position, heading, velocity, target_lane_index, target_velocity,
                                               route, timer)

        self.ngsim_traj = ngsim_traj
        self.traj = np.array(self.position)
        self.sim_steps = 0
        self.vehicle_ID = vehicle_ID
        self.planned_trajectory = None
        self.planned_heading = None
        self.planned_speed = None
        self.human = human
        self.IDM = IDM
        self.velocity_history = []
        self.heading_history = []
        self.crash_history = []
        self.position_history = []
        self.acc = acc
        self.steering_noise = None
        self.acc_noise = None
        self.next_position = None
        self.MARGIN = 5
        self.LENGTH = v_length  # Vehicle length [m]
        self.WIDTH = v_width  # Vehicle width [m]

    @classmethod
    def create(cls, road, vehicle_ID, position, v_length, v_width, ngsim_traj, heading=0, velocity=0, acc=0,
               target_velocity=15, human=False, IDM=False):
        """
        Create a human-like (IRL) driving vehicle in replace of a NGSIM vehicle.
        """
        v = cls(road, position, heading, velocity, acc, target_velocity=target_velocity,
                vehicle_ID=vehicle_ID, v_length=v_length, v_width=v_width, ngsim_traj=ngsim_traj, human=human, IDM=IDM)

        return v

    def act(self, step):
        if self.planned_trajectory is not None and not self.IDM:
            control_heading, acceleration = self.control_vehicle(self.planned_trajectory[step], self.planned_speed[step], self.planned_heading[step])
            self.action = {'steering': control_heading,
                           'acceleration': acceleration}
            # except Exception as e:
            #     super(InterActionVehicle, self).act()

        elif self.IDM:
            super(InterActionVehicle, self).act()
            # print('self.vehicle.action', self.action)
        else:
            return

    def control_vehicle(self, next_position, next_speed, next_heading):

        # Heading control
        heading_rate_command = self.KP_HEADING * utils.wrap_to_pi(next_heading - self.heading)

        # Heading rate to steering angle
        steering_angle = np.arctan(self.LENGTH / utils.not_zero(self.velocity) * heading_rate_command)
        # steering_angle = np.clip(steering_angle, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)

        acceleration = 10 * (next_speed - self.velocity)
        return steering_angle, acceleration

    def fix_angle(self, angle):
        while angle <= -math.pi:
            angle += 2 * math.pi

        while angle > math.pi:
            angle -= 2 * math.pi

        return angle

    def check_collision(self, other):
        """
        Check for collision with another vehicle.

        :param other: the other vehicle
        """
        if not self.COLLISIONS_ENABLED or not other.COLLISIONS_ENABLED or self.crashed or other is self:
            return

        # Fast spherical pre-check
        if np.linalg.norm(other.position - self.position) > (other.LENGTH + self.LENGTH) / 2:
            return

        # if both vehicles are NGSIM vehicles and have not been overriden
        # if isinstance(self, NGSIMVehicle) and not self.overtaken and isinstance(other, NGSIMVehicle) and not other.overtaken:
        #     return
        # print('111', self, other)

        # Accurate rectangular check
        if utils.rotated_rectangles_intersect((self.position, 0.9*self.LENGTH, 0.9*self.WIDTH, self.heading),
                                              (other.position, 0.9*other.LENGTH, 0.9*other.WIDTH, other.heading)):
            self.velocity = other.velocity = min([self.velocity, other.velocity], key=abs)
            self.crashed = other.crashed = True

        # if utils.judge_car_collision(float(self.position[0]), float(self.position[1]), self.LENGTH, self.WIDTH, self.heading,
        #                              float(other.position[0]), float(other.position[1]), other.LENGTH, other.WIDTH, other.heading):
        #     self.velocity = other.velocity = min([self.velocity, other.velocity], key=abs)
        #     self.crashed = other.crashed = True

    def step(self, dt):
        self.sim_steps += 1
        self.heading_history.append(self.heading)
        self.velocity_history.append(self.velocity)
        self.crash_history.append(self.crashed)
        self.position_history.append(self.position)
        if not self.IDM:
            front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self, margin=self.MARGIN)
            if front_vehicle is not None:
                gap = self.lane_distance_to(front_vehicle)
                desired_gap = self.desired_gap(self, front_vehicle)
            else:
                gap = 100
                desired_gap = 50

            if gap < desired_gap:
                self.IDM = True
                self.action['steering'] = self.steering_control(self.target_lane_index)

                # Longitudinal: IDM
                self.action['acceleration'] = self.acceleration(ego_vehicle=self, front_vehicle=front_vehicle,
                                                           rear_vehicle=rear_vehicle)
                self.action['acceleration'] = np.clip(self.action['acceleration'], -self.ACC_MAX, self.ACC_MAX)

        super(InterActionVehicle, self).step(dt)

        self.traj = np.append(self.traj, self.position, axis=0)

    def calculate_human_likeness(self):
        original_traj = self.ngsim_traj[:self.sim_steps + 1, :2]
        ego_traj = self.traj.reshape(-1, 2)
        ADE = np.mean([np.linalg.norm(original_traj[i] - ego_traj[i]) for i in
                       range(ego_traj.shape[0])])  # Average Displacement Error (ADE)
        FDE = np.linalg.norm(original_traj[-1] - ego_traj[-1])  # Final Displacement Error (FDE)

        return (ADE, FDE)


class IntersectionHumanLikeVehicle(IDMVehicle):
    """
    Create a human-like (IRL) driving agent.
    """
    TAU_A = 0.2  # [s]
    TAU_DS = 0.1  # [s]
    PURSUIT_TAU = 1.5 * TAU_DS  # [s]
    KP_A = 1 / TAU_A
    KP_HEADING = 1 / TAU_DS
    KP_LATERAL = 1 / 0.2  # [1/s]
    MAX_STEERING_ANGLE = np.pi / 3  # [rad]
    MAX_VELOCITY = 30  # [m/s]

    def __init__(self, road, position,
                 heading=0,
                 velocity=0,
                 acc=0,
                 target_lane_index=None,
                 target_velocity=15,  # Speed reference
                 route=None,
                 timer=None,
                 start_step=0,
                 vehicle_ID=None, v_length=None, v_width=None, ngsim_traj=None, human=False, IDM=False):
        super(IntersectionHumanLikeVehicle, self).__init__(road, position, heading, velocity,
                                                           target_lane_index, target_velocity, route, timer)

        self.ngsim_traj = ngsim_traj
        self.traj = np.array(self.position)
        self.sim_steps = 0
        self.vehicle_ID = vehicle_ID
        self.planned_trajectory = None
        self.human = human
        self.IDM = IDM
        self.velocity_history = []
        self.heading_history = []
        self.crash_history = []
        self.action_history = []
        self.acc = acc
        self.steering_noise = None
        self.acc_noise = None
        self.MARGIN = 5
        self.LENGTH = v_length  # Vehicle length [m]
        self.WIDTH = v_width  # Vehicle width [m]
        self.start_step = start_step - 1

    @classmethod
    def create(cls, road, vehicle_ID, position, v_length, v_width, ngsim_traj, heading=0, velocity=0, acc=0,
               target_velocity=15, human=False, IDM=False, start_step=0):
        """
        Create a human-like (IRL) driving vehicle in replace of a NGSIM vehicle.
        """
        v = cls(road, position, heading, velocity, acc, target_velocity=target_velocity,
                vehicle_ID=vehicle_ID, v_length=v_length, v_width=v_width, ngsim_traj=ngsim_traj, human=human, IDM=IDM, start_step=start_step)

        return v

    def make_linear(self):
        self.linear, self.unique_arr = init_linear(self.planned_trajectory, PolyLaneFixedWidth)

    def act(self, step):
        if self.planned_trajectory is not None and not self.IDM:
            try:
                control_heading, acceleration = self.control_vehicle(self.planned_trajectory[step-self.start_step], self.planned_speed[step - self.start_step], self.planned_heading[step - self.start_step])
                self.action = {'steering': control_heading,
                               'acceleration': acceleration}
            except Exception as e:
                print(e)

        elif self.IDM:
            super(IntersectionHumanLikeVehicle, self).act()
            # print('self.vehicle.action', self.action)
        else:
            return

    def control_vehicle(self, next_position, next_speed, next_heading):

        # Heading control
        heading_rate_command = self.KP_HEADING * utils.wrap_to_pi(next_heading - self.heading)

        # Heading rate to steering angle
        steering_angle = np.arctan(self.LENGTH / utils.not_zero(self.velocity) * heading_rate_command)
        # steering_angle = np.clip(steering_angle, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)

        acceleration = 10 * (next_speed - self.velocity)
        return steering_angle, acceleration

    def step(self, dt):
        self.sim_steps += 1
        self.heading_history.append(self.heading)
        self.velocity_history.append(self.velocity)
        self.crash_history.append(self.crashed)
        self.action_history.append(self.action)

        super(IntersectionHumanLikeVehicle, self).step(dt)

        self.traj = np.append(self.traj, self.position, axis=0)

    def calculate_human_likeness(self):
        original_traj = self.ngsim_traj[:self.sim_steps + 1, :2]
        ego_traj = self.traj.reshape(-1, 2)
        ADE = np.mean([np.linalg.norm(original_traj[i] - ego_traj[i]) for i in
                       range(ego_traj.shape[0])])  # Average Displacement Error (ADE)
        FDE = np.linalg.norm(original_traj[-1] - ego_traj[-1])  # Final Displacement Error (FDE)

        return (ADE, FDE)

# class IntersectionHumanLikeVehicle(IDMVehicle):
#     """
#     Create a human-like (IRL) driving agent.
#     """
#     TAU_A = 0.2  # [s]
#     TAU_DS = 0.1  # [s]
#     PURSUIT_TAU = 1.5 * TAU_DS  # [s]
#     KP_A = 1 / TAU_A
#     KP_HEADING = 1 / TAU_DS
#     KP_LATERAL = 1 / 0.2  # [1/s]
#     MAX_STEERING_ANGLE = np.pi / 3  # [rad]
#     MAX_VELOCITY = 30  # [m/s]
#     # # Longitudinal policy parameters
#     # ACC_MAX = 5.0 # [m/s2]  """Maximum acceleration."""
#     # COMFORT_ACC_MAX = 3.0 # [m/s2]  """Desired maximum acceleration."""
#     # COMFORT_ACC_MIN = -3.0 # [m/s2] """Desired maximum deceleration."""
#     # DISTANCE_WANTED = 1.0 # [m] """Desired jam distance to the front vehicle."""
#     # TIME_WANTED = 0.5 # [s]  """Desired time gap to the front vehicle."""
#     # DELTA = 4.0  # [] """Exponent of the velocity term."""
#     #
#     # # Lateral policy parameters [MOBIL]
#     # POLITENESS = 0.1  # in [0, 1]
#     # LANE_CHANGE_MIN_ACC_GAIN = 0.2 # [m/s2]
#     # LANE_CHANGE_MAX_BRAKING_IMPOSED = 2.0 # [m/s2]
#     # LANE_CHANGE_DELAY = 1.0  # [s]
#
#     # Longitudinal policy parameters
#     ACC_MAX = 5 # [m/s2]  """Maximum acceleration."""
#     COMFORT_ACC_MAX = 5.99 # [m/s2]  """Desired maximum acceleration."""
#     COMFORT_ACC_MIN = -5.99 # [m/s2] """Desired maximum deceleration."""
#     DISTANCE_WANTED = 0.99 # [m] """Desired jam distance to the front vehicle."""
#     TIME_WANTED = 0.21 # [s]  """Desired time gap to the front vehicle."""
#     DELTA = 4.0  # [] """Exponent of the velocity term."""
#
#     # Lateral policy parameters [MOBIL]
#     POLITENESS = 0.1  # in [0, 1]
#     LANE_CHANGE_MIN_ACC_GAIN = 0.2 # [m/s2]
#     LANE_CHANGE_MAX_BRAKING_IMPOSED = 2.0 # [m/s2]
#     LANE_CHANGE_DELAY = 1.0  # [s]
#
#     def __init__(self, road, position,
#                  heading=0,
#                  velocity=0,
#                  acc=0,
#                  target_lane_index=None,
#                  target_velocity=15,  # Speed reference
#                  route=None,
#                  timer=None,
#                  vehicle_ID=None, v_length=None, v_width=None, ngsim_traj=None, human=False, IDM=False):
#         super(IntersectionHumanLikeVehicle, self).__init__(road, position, heading, velocity,
#                                                            target_lane_index, target_velocity, route, timer)
#
#         self.ngsim_traj = ngsim_traj
#         self.traj = np.array(self.position)
#         self.sim_steps = 0
#         self.vehicle_ID = vehicle_ID
#         self.planned_trajectory = None
#         self.human = human
#         self.IDM = IDM
#         self.velocity_history = []
#         self.heading_history = []
#         self.crash_history = []
#         self.acc = acc
#         self.steering_noise = None
#         self.acc_noise = None
#         self.MARGIN = 5
#         self.LENGTH = v_length  # Vehicle length [m]
#         self.WIDTH = v_width  # Vehicle width [m]
#
#     def init_linear(self):
#         unique_arr = np.unique(self.ngsim_traj[:, :2].copy(), axis=0)
#         mask = np.where(np.sum(unique_arr, axis=1) == 0, False, True)
#         unique_arr = unique_arr[mask]
#         self.unique_arr = unique_arr
#
#         if self.unique_arr.shape[0] > 2 and (np.linalg.norm(self.unique_arr[-1] - self.unique_arr[0]) > 0.5):
#             # self.unique_arr = filter_similar_points(self.unique_arr, min_cluster_size=2, min_samples=10)
#             self.unique_arr = filter_similar_points(self.unique_arr, threshold=0.5)
#             # 原始数据点
#             fig, ax = plt.subplots()
#             x = sorted(unique_arr[:, 0], reverse=unique_arr[0, 0] > unique_arr[-1, 0])
#             y = sorted(unique_arr[:, 1], reverse=unique_arr[0, 1] > unique_arr[-1, 1])
#             ax.plot(x, y, color='r', alpha=0.5)
#             ax.plot(unique_arr[:, 0], unique_arr[:, 1], color='b', alpha=0.5)
#             plt.savefig('./spline.png')
#             self.unique_arr = np.stack([x, y], axis=1).tolist()
#             try:
#                 self.linear = PolyLaneFixedWidth(self.unique_arr)
#             except Exception as e:
#                 print(e)
#
#     @classmethod
#     def create(cls, road, vehicle_ID, position, v_length, v_width, ngsim_traj, heading=0, velocity=0, acc=0,
#                target_velocity=15, human=False, IDM=False):
#         """
#         Create a human-like (IRL) driving vehicle in replace of a NGSIM vehicle.
#         """
#         v = cls(road, position, heading, velocity, acc, target_velocity=target_velocity,
#                 vehicle_ID=vehicle_ID, v_length=v_length, v_width=v_width, ngsim_traj=ngsim_traj, human=human, IDM=IDM)
#
#         return v
#
#     def act(self, step):
#         if self.planned_trajectory is not None and not self.IDM:
#             try:
#                 if step < self.planned_trajectory.shape[0] - 1:
#                     self.next_position = self.planned_trajectory[step+1]
#
#                 control_heading, acceleration = self.control_vehicle(self.planned_trajectory[step], self.planned_speed[step], self.planned_heading[step])
#                 self.action = {'steering': control_heading,
#                                'acceleration': acceleration}
#             except Exception as e:
#                 super(IntersectionHumanLikeVehicle, self).act()
#
#         elif self.IDM:
#             super(IntersectionHumanLikeVehicle, self).act()
#             # print('self.vehicle.action', self.action)
#         else:
#             return
#
#     def control_vehicle(self, next_position, next_speed, next_heading):
#
#         # Heading control
#         heading_rate_command = self.KP_HEADING * utils.wrap_to_pi(next_heading - self.heading)
#
#         # Heading rate to steering angle
#         steering_angle = np.arctan(self.LENGTH / utils.not_zero(self.velocity) * heading_rate_command)
#         # steering_angle = np.clip(steering_angle, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)
#
#         acceleration = 10 * (next_speed - self.velocity)
#         return steering_angle, acceleration
#
#     def step(self, dt):
#         self.sim_steps += 1
#         self.heading_history.append(self.heading)
#         self.velocity_history.append(self.velocity)
#         self.crash_history.append(self.crashed)
#
#         super(IntersectionHumanLikeVehicle, self).step(dt)
#
#         self.traj = np.append(self.traj, self.position, axis=0)
#
#     def calculate_human_likeness(self):
#         original_traj = self.ngsim_traj[:self.sim_steps + 1, :2]
#         ego_traj = self.traj.reshape(-1, 2)
#         ADE = np.mean([np.linalg.norm(original_traj[i] - ego_traj[i]) for i in
#                        range(ego_traj.shape[0])])  # Average Displacement Error (ADE)
#         FDE = np.linalg.norm(original_traj[-1] - ego_traj[-1])  # Final Displacement Error (FDE)
#
#         return (ADE, FDE)
#

class Pedestrian(IDMVehicle):

    TAU_A = 0.2  # [s]
    TAU_DS = 0.1  # [s]
    PURSUIT_TAU = 1.5 * TAU_DS  # [s]
    KP_A = 1 / TAU_A
    KP_HEADING = 1 / TAU_DS
    KP_LATERAL = 1 / 0.2  # [1/s]
    MAX_STEERING_ANGLE = np.pi / 3  # [rad]
    MAX_VELOCITY = 1  # [m/s]
    # Longitudinal policy parameters
    ACC_MAX = 1.0 # [m/s2]  """Maximum acceleration."""
    COMFORT_ACC_MAX = 1.0 # [m/s2]  """Desired maximum acceleration."""
    COMFORT_ACC_MIN = -1.0 # [m/s2] """Desired maximum deceleration."""
    DISTANCE_WANTED = 1.0 # [m] """Desired jam distance to the front vehicle."""
    TIME_WANTED = 0.5 # [s]  """Desired time gap to the front vehicle."""
    DELTA = 4.0  # [] """Exponent of the velocity term."""

    def __init__(self, road, position,
                 heading=0,
                 velocity=0,
                 acc=0,
                 target_lane_index=None,
                 target_velocity=15,  # Speed reference
                 route=None,
                 timer=None,
                 vehicle_ID=None, v_length=None, v_width=None, ngsim_traj=None, human=False, IDM=False):
        super(Pedestrian, self).__init__(road, position, heading, velocity,
                                                           target_lane_index, target_velocity, route, timer)

        self.ngsim_traj = ngsim_traj
        self.traj = np.array(self.position)
        self.sim_steps = 0
        self.vehicle_ID = vehicle_ID
        self.planned_trajectory = None
        self.human = human
        self.IDM = IDM
        self.velocity_history = []
        self.heading_history = []
        self.crash_history = []
        self.acc = acc
        self.steering_noise = None
        self.acc_noise = None
        self.MARGIN = 5
        self.LENGTH = v_length  # Vehicle length [m]
        self.WIDTH = v_width  # Vehicle width [m]
        unique_arr = np.unique(ngsim_traj[:, :2].copy(), axis=0)
        mask = np.where(np.sum(unique_arr, axis=1) == 0, False, True)
        self.linear = PolyLaneFixedWidth(unique_arr[mask].tolist())

    @classmethod
    def create(cls, road, vehicle_ID, position, v_length, v_width, ngsim_traj, heading=0, velocity=0, acc=0,
               target_velocity=15, human=False, IDM=False):
        """
        Create a human-like (IRL) driving vehicle in replace of a NGSIM vehicle.
        """
        v = cls(road, position, heading, velocity, acc, target_velocity=target_velocity,
                vehicle_ID=vehicle_ID, v_length=v_length, v_width=v_width, ngsim_traj=ngsim_traj, human=human, IDM=IDM)

        return v

    def act(self, step):
        if self.planned_trajectory is not None and not self.IDM:
            try:
                if step < self.planned_trajectory.shape[0] - 1:
                    self.next_position = self.planned_trajectory[step+1]

                control_heading, acceleration = self.control_vehicle(self.planned_trajectory[step], self.planned_speed[step], self.planned_heading[step])
                self.action = {'steering': control_heading,
                               'acceleration': acceleration}
            except Exception as e:
                super(Pedestrian, self).act()

        elif self.IDM:
            super(Pedestrian, self).act()
            # print('self.vehicle.action', self.action)
        else:
            return

    def control_vehicle(self, next_position, next_speed, next_heading):

        # Heading control
        heading_rate_command = self.KP_HEADING * utils.wrap_to_pi(next_heading - self.heading)

        # Heading rate to steering angle
        steering_angle = np.arctan(self.LENGTH / utils.not_zero(self.velocity) * heading_rate_command)
        # steering_angle = np.clip(steering_angle, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)

        acceleration = 10 * (next_speed - self.velocity)
        return steering_angle, acceleration

    def step(self, dt):
        self.sim_steps += 1
        self.heading_history.append(self.heading)
        self.velocity_history.append(self.velocity)
        self.crash_history.append(self.crashed)

        super(Pedestrian, self).step(dt)

        self.traj = np.append(self.traj, self.position, axis=0)

    def check_collision(self, other):

        if not self.COLLISIONS_ENABLED or not other.COLLISIONS_ENABLED or self.crashed or other is self:
            return

        # Fast spherical pre-check
        if np.linalg.norm(other.position - self.position) > self.LENGTH * 0.5:
            return

        # Accurate rectangular check
        if utils.rotated_rectangles_intersect((self.position, 0.5*self.LENGTH, 0.5*self.WIDTH, self.heading),
                                              (other.position, 0.9*other.LENGTH, 0.9*other.WIDTH, other.heading)):
            self.velocity = other.velocity = min([self.velocity, other.velocity], key=abs)
            self.crashed = other.crashed = True


