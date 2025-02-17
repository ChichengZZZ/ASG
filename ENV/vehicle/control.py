from __future__ import division, print_function
import numpy as np
import copy
from NGSIM_env import utils
from NGSIM_env.vehicle.dynamics import Vehicle


class ControlledVehicle(Vehicle):
    """
    由两个低级控制器驾驶的车辆，可以执行高级动作，如巡航控制和车道变更。

        纵向控制器是一个速度控制器；
        横向控制器是一个航向控制器，串联着一个横向位置控制器。

    """

    TAU_A = 0.6  # [s]
    TAU_DS = 0.2  # [s]
    TAU_LATERAL = 3 # [s]

    PURSUIT_TAU = 0.5*TAU_DS  # [s]
    KP_A = 1 / TAU_A
    KP_HEADING = 1 / TAU_DS
    KP_LATERAL = 1 / TAU_LATERAL # [1/s]
    MAX_STEERING_ANGLE = np.pi / 3  # [rad]
    DELTA_VELOCITY = 2  # [m/s]

    def __init__(self,
                 road,
                 position,
                 heading=0,
                 velocity=0,
                 target_lane_index=None,
                 target_velocity=None,
                 route=None):
        super(ControlledVehicle, self).__init__(road, position, heading, velocity)
        self.target_lane_index = target_lane_index or self.lane_index
        self.target_velocity = target_velocity or self.velocity
        self.route = route

    @classmethod
    def create_from(cls, vehicle):
        """
        从现有的车辆创建一个新的车辆。 车辆动力学和目标动力学被复制，其他属性默认。

        :param vehicle: 一辆车辆
        :return: 一个在相同动力学状态下的新车辆

        """
        v = cls(vehicle.road, vehicle.position, heading=vehicle.heading, velocity=vehicle.velocity,
                target_lane_index=vehicle.target_lane_index, target_velocity=vehicle.target_velocity, route=vehicle.route)
        return v

    def plan_route_to(self, destination):
        """
        在道路网络中规划到达目的地的路径。

        :param destination: 道路网络中的一个节点
        """
        path = self.road.network.shortest_path(self.lane_index[1], destination)
        if path:
            self.route = [self.lane_index] + [(path[i], path[i + 1], None) for i in range(len(path) - 1)]
        else:
            self.route = [self.lane_index]
        return self

    def act(self, action=None):
        """
        执行高级动作来改变期望的车道或速度。

        - 如果提供了高级动作，更新目标速度和车道；
        - 然后执行纵向和横向控制。

        :param action: 一个高级动作
        """
        if self.linear is None:
            self.follow_road()

            if action == "FASTER":
                self.target_velocity += self.DELTA_VELOCITY

            elif action == "SLOWER":
                self.target_velocity -= self.DELTA_VELOCITY

            elif action == "LANE_RIGHT":
                _from, _to, _id = self.target_lane_index
                target_lane_index = _from, _to, np.clip(_id + 1, 0, len(self.road.network.graph[_from][_to]) - 1)
                if self.road.network.get_lane(target_lane_index).is_reachable_from(self.position):
                    self.target_lane_index = target_lane_index

            elif action == "LANE_LEFT":
                _from, _to, _id = self.target_lane_index
                target_lane_index = _from, _to, np.clip(_id - 1, 0, len(self.road.network.graph[_from][_to]) - 1)
                if self.road.network.get_lane(target_lane_index).is_reachable_from(self.position):
                    self.target_lane_index = target_lane_index

            action = {'steering': self.steering_control(self.target_lane_index), 'acceleration': self.velocity_control(self.target_velocity)}

        super(ControlledVehicle, self).act(action)

    def follow_road(self):
        """
        在车道的末端，自动切换到下一个车道。
        """
        if self.road.network.get_lane(self.target_lane_index).after_end(self.position):
            # print('-------')
            # temp = self.target_lane_index
            self.target_lane_index = self.road.network.next_lane(self.target_lane_index, route=self.route, position=self.position, np_random=self.road.np_random)
            # if temp != self.target_lane_index and self.target_lane_index==('s3', 'merge_out', 0):
            #     print(temp, 's3', 'merge_out', 0)

    def steering_control(self, target_lane_index):
        """
        操纵车辆以跟随给定车道的中心。

        1. 横向位置由比例控制器控制，产生横向速度指令
        2. 横向速度指令转换为航向参考值
        3. 航向由比例控制器控制，产生航向角速度指令
        4. 航向角速度指令转换为转向角度

        :param target_lane_index: 要跟随的车道的索引
        :return: 转向盘角度指令 [rad]
        """
        target_lane = self.road.network.get_lane(target_lane_index)
        lane_coords = target_lane.local_coordinates(self.position)
        lane_next_coords = lane_coords[0] + self.velocity * self.PURSUIT_TAU
        lane_future_heading = target_lane.heading_at(lane_next_coords)

        # Lateral position control
        lateral_velocity_command = - self.KP_LATERAL * lane_coords[1]

        # Lateral velocity to heading
        heading_command = np.arcsin(np.clip(lateral_velocity_command/utils.not_zero(self.velocity), -1, 1))
        heading_ref = lane_future_heading + np.clip(heading_command, -np.pi/4, np.pi/4)

        # Heading control
        heading_rate_command = self.KP_HEADING * utils.wrap_to_pi(heading_ref - self.heading)

        # Heading rate to steering angle
        steering_angle = np.arctan(self.LENGTH / utils.not_zero(self.velocity) * heading_rate_command)
        steering_angle = np.clip(steering_angle, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)

        return steering_angle

    def velocity_control(self, target_velocity):
        """
        控制车辆的速度。

        使用一个简单的比例控制器。

        :param target_velocity: 期望的速度
        :return: 加速度指令 [m/s2]
        """
        return self.KP_A * (target_velocity - self.velocity)

    def set_route_at_intersection(self, _to):
        """
        设置在下一个路口要跟随的道路。 删除当前计划的路线。
        :param _to: 在道路网络中下一个路口要跟随的道路的索引
        """

        if not self.route:
            return      
        for index in range(min(len(self.route), 3)):
            try:
                next_destinations = self.road.network.graph[self.route[index][1]]
            except KeyError:
                continue
            if len(next_destinations) >= 2:
                break
        else:
            return
        
        next_destinations_from = list(next_destinations.keys())
        if _to == "random":
            _to = self.road.np_random.randint(0, len(next_destinations_from))
        next_index = _to % len(next_destinations_from)
        self.route = self.route[0:index+1] + [(self.route[index][1], next_destinations_from[next_index], self.route[index][2])]

    def predict_trajectory_constant_velocity(self, times):
        """
        预测车辆沿着规划路线的未来位置，在恒定速度下
        :param times: 预测的时间步长
        :return: 位置，航向
        """
        coordinates = self.lane.local_coordinates(self.position)
        route = self.route or [self.lane_index]
        return zip(*[self.road.network.position_heading_along_route(route, coordinates[0] + self.velocity * t, 0) for t in times])


class MDPVehicle(ControlledVehicle):
    """
    具有指定离散范围内允许的目标速度的受控车辆。
    """

    SPEED_COUNT = 21 # []
    SPEED_MIN = 0  # [m/s]
    SPEED_MAX = 20  # [m/s]

    def __init__(self,
                 road,
                 position,
                 heading=0,
                 velocity=0,
                 target_lane_index=None,
                 target_velocity=None,
                 route=None,
                 ngsim_traj=None,
                 vehicle_ID=None, v_length=None, v_width=None):
        super(MDPVehicle, self).__init__(road, position, heading, velocity, target_lane_index, target_velocity, route)
        self.velocity_index = self.speed_to_index(self.target_velocity)
        self.target_velocity = self.index_to_speed(self.velocity_index)
        self.ngsim_traj = ngsim_traj
        self.traj = np.array(self.position)
        self.sim_steps = 0
        self.vehicle_ID = vehicle_ID
        self.LENGTH = v_length # Vehicle length [m]
        self.WIDTH = v_width # Vehicle width [m]

    @classmethod
    def create(cls, road, vehicle_ID, position, v_length, v_width, ngsim_traj, heading=0, velocity=0, target_velocity=10):
        """
        创建一个类似人类驾驶的车辆以取代 NGSIM 车辆。

        :param road: 车辆行驶的道路
        :param position: 车辆在道路上的起始位置
        :param velocity: 初始速度 [m/s]。如果为 None，则随机选择
        :return: 具有随机位置和/或速度的车辆
        """
        v = cls(road, position, heading, velocity, target_velocity=target_velocity, 
                vehicle_ID=vehicle_ID, v_length=v_length, v_width=v_width, ngsim_traj=ngsim_traj)
       
        return v

    def act(self, action=None):
        """
        执行高级动作。
        如果动作是速度变化，请从允许的离散范围中选择速度。
        否则，将动作转发给受控车辆处理程序。

        :param action: 一个高级动作
        """
        # print(action)

        if action == "FASTER":
            self.velocity_index = self.speed_to_index(self.velocity) + 1
        elif action == "SLOWER":
            self.velocity_index = self.speed_to_index(self.velocity) - 1
        else:
            super(MDPVehicle, self).act(action)
            return
        #print(self.velocity_index)
        self.velocity_index = np.clip(self.velocity_index, 0, self.SPEED_COUNT - 1)
        self.target_velocity = self.index_to_speed(self.velocity_index)
        super().act()

    def step(self, dt):
        self.sim_steps += 1
        super(MDPVehicle, self).step(dt)
        
        self.traj = np.append(self.traj, self.position, axis=0)

    @classmethod
    def index_to_speed(cls, index):
        """
        将在允许的速度中的索引转换为相应的速度
        :param index: 速度索引 []
        :return: 相应的速度 [m/s]
        """
        if cls.SPEED_COUNT > 1:
            return cls.SPEED_MIN + index * (cls.SPEED_MAX - cls.SPEED_MIN) / (cls.SPEED_COUNT - 1)
        else:
            return cls.SPEED_MIN

    @classmethod
    def speed_to_index(cls, speed):
        """
        找到离给定速度最接近的允许速度的索引。
        :param speed: 输入速度 [m/s]
        :return: 最接近的允许速度的索引 []
        """
        x = (speed - cls.SPEED_MIN) / (cls.SPEED_MAX - cls.SPEED_MIN)
        return int(np.clip(np.round(x * (cls.SPEED_COUNT - 1)), 0, cls.SPEED_COUNT - 1))

    def speed_index(self):
        """
        当前速度的索引
        """
        return self.speed_to_index(self.velocity)

    def predict_trajectory(self, actions, action_duration, trajectory_timestep, dt):
        """
        根据一系列动作预测车辆的未来轨迹。

        :param actions: 一系列未来的动作。
        :param action_duration: 每个动作的持续时间。
        :param trajectory_timestep: 保存车辆状态之间的持续时间。
        :param dt: 模拟的时间步长
        :return: 未来状态的序列
        """
        states = []
        v = copy.deepcopy(self)
        t = 0
        for action in actions:
            v.act(action)  # High-level decision
            for _ in range(int(action_duration / dt)):
                t += 1
                v.act()  # Low-level control action
                v.step(dt)
                if (t % int(trajectory_timestep / dt)) == 0:
                    states.append(copy.deepcopy(v))
                    
        return states
    
    def calculate_human_likeness(self):
        original_traj = self.ngsim_traj[:self.sim_steps+1,:2]
        ego_traj = self.traj.reshape(-1, 2)
        ADE = np.mean([np.linalg.norm(original_traj[i]-ego_traj[i]) for i in range(ego_traj.shape[0])]) # Average Displacement Error (ADE)
        FDE = np.linalg.norm(original_traj[-1]-ego_traj[-1]) # Final Displacement Error (FDE)
        
        return FDE
