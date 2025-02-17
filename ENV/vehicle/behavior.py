from __future__ import division, print_function
import numpy as np
from NGSIM_env.vehicle.control import ControlledVehicle
from NGSIM_env import utils


class IDMVehicle(ControlledVehicle):
    """
        同时使用纵向和横向决策策略的车辆。
        纵向：IDM模型根据前车的距离和速度计算加速度。
        横向：MOBIL模型通过最大化附近车辆的加速度来决定何时变道。
    """

    # 纵向策略参数

    ACC_MAX = 6.0  # [m/s2]
    """Maximum acceleration."""

    COMFORT_ACC_MAX = 3.0  # [m/s2]
    """Desired maximum acceleration."""

    COMFORT_ACC_MIN = -5.0  # [m/s2]
    """Desired maximum deceleration."""

    DISTANCE_WANTED = 5.0 + ControlledVehicle.LENGTH  # [m]
    """Desired jam distance to the front vehicle."""

    TIME_WANTED = 1.5  # [s]
    """Desired time gap to the front vehicle."""

    DELTA = 4.0  # []
    """Exponent of the velocity term."""

    DELTA_RANGE = [3.5, 4.5]
    """Range of delta when chosen randomly."""

    # Lateral policy parameters
    POLITENESS = 0.  # in [0, 1]
    LANE_CHANGE_MIN_ACC_GAIN = 0.2  # [m/s2]
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 2.0  # [m/s2]
    LANE_CHANGE_DELAY = 1.0  # [s]

    # # Longitudinal policy parameters
    # ACC_MAX = 1.5  # [m/s2]
    # COMFORT_ACC_MAX = 0.7 # [m/s2]
    # COMFORT_ACC_MIN = -0.7  # [m/s2]
    # DISTANCE_WANTED = 1.5 # [m]
    # TIME_WANTED = 1.2 # [s]
    # DELTA = 4.0  # []
    #
    # # Lateral policy parameters
    # POLITENESS = 0.01  # in [0, 1]
    # LANE_CHANGE_MIN_ACC_GAIN = 0.2  # [m/s2]
    # LANE_CHANGE_MAX_BRAKING_IMPOSED = 2.0  # [m/s2]
    # LANE_CHANGE_DELAY = 1.0  # [s]
    MARGIN = 1.0

    def __init__(self, road, position,
                 heading=0,
                 velocity=0,
                 target_lane_index=None,
                 target_velocity=None,
                 route=None,
                 enable_lane_change=True,
                 timer=None):
        super(IDMVehicle, self).__init__(road, position, heading, velocity, target_lane_index, target_velocity, route)
        self.enable_lane_change = enable_lane_change
        self.timer = timer or (np.sum(self.position)*np.pi) % self.LANE_CHANGE_DELAY
        self.linear = None

    def randomize_behavior(self):
        pass

    @classmethod
    def create_from(cls, vehicle):
        """
        根据现有车辆创建一个新车辆。 车辆的动力学和目标动力学会被复制，其他属性为默认值。

        参数：
        - vehicle：一个车辆
        返回值： - 一个与原车辆在相同动力学状态下的新车辆

        """
        v = cls(vehicle.road, vehicle.position, heading=vehicle.heading, velocity=vehicle.velocity,
                target_lane_index=vehicle.target_lane_index, target_velocity=vehicle.target_velocity,
                route=vehicle.route, timer=getattr(vehicle, 'timer', None))

        return v

    def act(self, action=None):
        """
        执行一个动作。

        目前，因为车辆根据IDM和MOBIL模型自行决策加速和变道，所以不支持任何动作。

        :param action: 动作

        """
        if self.crashed:
            return
        action = {}
        if self.linear is None:
            self.follow_road()

            front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self, margin=self.MARGIN)

            # Lateral: MOBIL
            # self.follow_road()
            if self.enable_lane_change:
                # print('self.change_lane_policy()')
                self.change_lane_policy()
            action['steering'] = self.steering_control(self.target_lane_index)

            # Longitudinal: IDM
            # print(self, self.ACC_MAX)
            action['acceleration'] = self.acceleration(ego_vehicle=self, front_vehicle=front_vehicle, rear_vehicle=rear_vehicle)
            if self.lane_index != self.target_lane_index:
                front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self, self.target_lane_index)
                target_idm_acceleration = self.acceleration(ego_vehicle=self,
                                                            front_vehicle=front_vehicle,
                                                            rear_vehicle=rear_vehicle)
                action['acceleration'] = min(action['acceleration'], target_idm_acceleration)

            action['acceleration'] = np.clip(action['acceleration'], -self.ACC_MAX, self.ACC_MAX)
        else:
            self.follow_road()
            front_vehicle, rear_vehicle = self.neighbour_vehicles(self.road.vehicles)
            action['steering'] = self.linear.steering_control(self.position, self.velocity, self.heading, self.LENGTH,
                                                              self.PURSUIT_TAU, self.KP_LATERAL, self.KP_HEADING, self.MAX_STEERING_ANGLE)

            action['acceleration'] = self.linear.acceleration(ego_vehicle=self, front_vehicle=front_vehicle, rear_vehicle=rear_vehicle, position=self.position,
                                                              DISTANCE_WANTED=self.DISTANCE_WANTED, TIME_WANTED=self.TIME_WANTED, DELTA=self.DELTA,
                                                              COMFORT_ACC_MAX=self.COMFORT_ACC_MAX, COMFORT_ACC_MIN=self.COMFORT_ACC_MIN
                                                              )
            action['acceleration'] = np.clip(action['acceleration'], -self.ACC_MAX, self.ACC_MAX)

        super(ControlledVehicle, self).act(action)

    def step(self, dt):
        """
        进行仿真步骤。

        增加一个用于决策策略的计时器，并进行车辆动力学步进。

        :param dt: 时间步长

        """
        self.timer += dt
        # if self.target_lane_index != self.lane_index:
        #      print(self.target_lane_index, self.lane_index)
        # print(super(IDMVehicle, self).step)
        super(IDMVehicle, self).step(dt)

    def acceleration(self, ego_vehicle, front_vehicle=None, rear_vehicle=None):
        """
        使用智能驾驶模型计算加速度指令。

        加速度的选择是为了：
        - 达到目标速度；
        - 保持与前车之间的最小安全距离（以及安全时间）。

        :param ego_vehicle: 需要计算期望加速度的车辆。它不一定是一个IDM车辆，这就是为什么这个方法是一个类方法的原因。这允许一个IDM车辆来推断其他车辆的行为，即使它们可能不是IDMs。
        :param front_vehicle: 在自车之前的车辆
        :param rear_vehicle: 在自车之后的车辆
        :return: 自车的加速度指令 [m/s2]

        """
        if not ego_vehicle:
            return 0

        ego_target_velocity = utils.not_zero(getattr(ego_vehicle, "target_velocity", 0))
        acceleration = self.COMFORT_ACC_MAX * (1 - np.power(max(ego_vehicle.velocity, 0) / ego_target_velocity, self.DELTA))
        # if acceleration.shape[0] > 1:
        #     print(acceleration)
        if front_vehicle:
            d = ego_vehicle.lane_distance_to(front_vehicle)
            acceleration -= self.COMFORT_ACC_MAX * np.power(self.desired_gap(ego_vehicle, front_vehicle) / utils.not_zero(d), 2)

        return acceleration

    def desired_gap(self, ego_vehicle, front_vehicle=None):
        """
        计算车辆与前车之间的期望距离。

        :param ego_vehicle: 被控制的车辆
        :param front_vehicle: 前面的车辆
        :return: 两辆车之间的期望距离
        """
        d0 = self.DISTANCE_WANTED + ego_vehicle.LENGTH / 2 + front_vehicle.LENGTH / 2
        tau = self.TIME_WANTED
        ab = -self.COMFORT_ACC_MAX * self.COMFORT_ACC_MIN
        dv = ego_vehicle.velocity - front_vehicle.velocity
        d_star = d0 + ego_vehicle.velocity * tau + ego_vehicle.velocity * dv / (2 * np.sqrt(ab))

        return d_star

    def maximum_velocity(self, front_vehicle=None):
        """
        计算为了避免不可避免的碰撞状态所允许的最大速度。

        假设前车将以全减速刹车，并且在一定延迟后被注意到，计算出允许自车足够减速以避开碰撞的最大速度。

        :param front_vehicle: 前车
        :return: 最大允许速度和建议的加速度。
        """

        if not front_vehicle:
            return self.target_velocity
        d0 = self.DISTANCE_WANTED
        a0 = self.COMFORT_ACC_MIN
        a1 = self.COMFORT_ACC_MIN
        tau = self.TIME_WANTED
        d = max(self.lane_distance_to(front_vehicle) - self.LENGTH / 2 - front_vehicle.LENGTH / 2 - d0, 0)
        v1_0 = front_vehicle.velocity
        delta = 4 * (a0 * a1 * tau) ** 2 + 8 * a0 * (a1 ** 2) * d + 4 * a0 * a1 * v1_0 ** 2
        v_max = -a0 * tau + np.sqrt(delta) / (2 * a1)

        # Velocity control
        self.target_velocity = min(self.maximum_velocity(front_vehicle), self.target_velocity)
        acceleration = self.velocity_control(self.target_velocity)

        return v_max, acceleration

    def change_lane_policy(self):
        """
        决定何时换道。

        根据以下因素：
        - 频率；
        - 目标车道的接近程度；
        - MOBIL模型。

        """
        # If a lane change already ongoing
        if self.lane_index != self.target_lane_index:
            # If we are on correct route but bad lane: abort it if someone else is already changing into the same lane
            if self.lane_index[:2] == self.target_lane_index[:2]:
                for v in self.road.vehicles:
                    if v is not self \
                            and v.lane_index != self.target_lane_index \
                            and isinstance(v, ControlledVehicle) \
                            and v.target_lane_index == self.target_lane_index:
                        d = self.lane_distance_to(v)
                        d_star = self.desired_gap(self, v)
                        if 0 < d < d_star:
                            self.target_lane_index = self.lane_index
                            break
            return

        # else, at a given frequency,
        if not utils.do_every(self.LANE_CHANGE_DELAY, self.timer):
            return
        self.timer = 0

        # decide to make a lane change
        for lane_index in self.road.network.side_lanes(self.lane_index):
            # Is the candidate lane close enough?
            if not self.road.network.get_lane(lane_index).is_reachable_from(self.position):
                continue
            # Does the MOBIL model recommend a lane change?
            if self.mobil(lane_index):
                self.target_lane_index = lane_index

    def mobil(self, lane_index):
        """
        MOBIL换道模型：通过换道来最小化因换道引起的整体制动

        只有在以下情况下，车辆才应该换道：
        - 换道后（和/或后续的车辆）能够更快加速；
        - 它不会对新的后续车辆造成不安全的制动。

        :param lane_index: 换道的候选车道
        :return: 是否应进行换道
        """
        # 对新的后续车辆来说，这个动作是否安全？
        new_preceding, new_following = self.road.neighbour_vehicles(self, lane_index)
        new_following_a = self.acceleration(ego_vehicle=new_following, front_vehicle=new_preceding)
        new_following_pred_a = self.acceleration(ego_vehicle=new_following, front_vehicle=self)

        if new_following_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
            return False

        # Do I have a planned route for a specific lane which is safe for me to access?
        old_preceding, old_following = self.road.neighbour_vehicles(self)
        self_pred_a = self.acceleration(ego_vehicle=self, front_vehicle=new_preceding)
        if self.route and self.route[0][2]:
            # Wrong direction
            if np.sign(lane_index[2] - self.target_lane_index[2]) != np.sign(self.route[0][2] - self.target_lane_index[2]):
                return False
            # Unsafe braking required
            elif self_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
                return False

        # Is there an acceleration advantage for me and/or my followers to change lane?
        else:
            self_a = self.acceleration(ego_vehicle=self, front_vehicle=old_preceding)
            old_following_a = self.acceleration(ego_vehicle=old_following, front_vehicle=self)
            old_following_pred_a = self.acceleration(ego_vehicle=old_following, front_vehicle=old_preceding)
            jerk = self_pred_a - self_a + self.POLITENESS * (new_following_pred_a - new_following_a + old_following_pred_a - old_following_a)
            if jerk < self.LANE_CHANGE_MIN_ACC_GAIN:
                return False

        # All clear, let's go!
        return True

    def recover_from_stop(self, acceleration):
        """
        如果停在错误的车道上，请尝试倒车操作。

        :param acceleration: 期望的加速度（来自IDM）
        :return: 建议的加速度以脱离困境
        """
        stopped_velocity = 2
        safe_distance = 200

        # Is the vehicle stopped on the wrong lane?
        if self.target_lane_index != self.lane_index and self.velocity < stopped_velocity:
            _, rear = self.road.neighbour_vehicles(self)
            _, new_rear = self.road.neighbour_vehicles(self, self.road.network.get_lane(self.target_lane_index))
            # Check for free room behind on both lanes
            if (not rear or rear.lane_distance_to(self) > safe_distance) and \
                    (not new_rear or new_rear.lane_distance_to(self) > safe_distance):
                # Reverse
                return -self.COMFORT_ACC_MAX / 2

        return acceleration

    def neighbour_vehicles(self, vehicles, margin=0.5):
        s = self.linear.local_coordinates(self.position)[0]
        s_front = s_rear = None
        v_front = v_rear = None
        for v in vehicles:
            # print(v.__class__.__name__)
            if v is not self and v.__class__.__name__ not in ['Pedestrian']:
                s_v, lat_v = self.linear.local_coordinates(v.position)
                if not self.linear.on_lane(v.position, s_v, lat_v, margin=margin):
                    continue
                if s <= s_v and (s_front is None or s_v <= s_front):
                    s_front = s_v
                    v_front = v
                if s_v < s and (s_rear is None or s_v > s_rear):
                    s_rear = s_v
                    v_rear = v

        return v_front, v_rear


class LinearVehicle(IDMVehicle):
    """
    车辆的纵向和横向控制器相对于参数是线性的。
    """
    ACCELERATION_PARAMETERS = [0.3, 0.14, 0.8]
    STEERING_PARAMETERS = [ControlledVehicle.KP_HEADING, ControlledVehicle.KP_HEADING * ControlledVehicle.KP_LATERAL]

    ACCELERATION_RANGE = np.array([0.5*np.array(ACCELERATION_PARAMETERS), 1.5*np.array(ACCELERATION_PARAMETERS)])
    STEERING_RANGE = np.array([np.array(STEERING_PARAMETERS) - np.array([0.07, 1.5]),
                               np.array(STEERING_PARAMETERS) + np.array([0.07, 1.5])])

    TIME_WANTED = 2.0

    def __init__(self, road, position,
                 heading=0,
                 velocity=0,
                 target_lane_index=None,
                 target_velocity=None,
                 route=None,
                 enable_lane_change=True,
                 timer=None):
        super(LinearVehicle, self).__init__(road,
                                            position,
                                            heading,
                                            velocity,
                                            target_lane_index,
                                            target_velocity,
                                            route,
                                            enable_lane_change,
                                            timer)

    def randomize_behavior(self):
        ua = self.road.np_random.uniform(size=np.shape(self.ACCELERATION_PARAMETERS))
        self.ACCELERATION_PARAMETERS = self.ACCELERATION_RANGE[0] + ua*(self.ACCELERATION_RANGE[1] - self.ACCELERATION_RANGE[0])
        ub = self.road.np_random.uniform(size=np.shape(self.STEERING_PARAMETERS))
        self.STEERING_PARAMETERS = self.STEERING_RANGE[0] + ub*(self.STEERING_RANGE[1] - self.STEERING_RANGE[0])

    def acceleration(self, ego_vehicle, front_vehicle=None, rear_vehicle=None):
        """
        选择加速度的目的是：
        - 达到目标速度；
        - 若前车速度低于自车速度，则达到与前车相同的速度；
        - 若后车速度高于自车速度，则达到与后车相同的速度；
        - 保持与前车的最小安全距离。

        :param ego_vehicle: 需要计算期望加速度的车辆。它不一定是一个线性车辆，因此该方法是一个类方法。这使得线性车辆能够推理其他车辆的行为，即使它们可能不是线性的。
        :param front_vehicle: 自车前方的车辆
        :param rear_vehicle: 自车后方的车辆
        :return: 自车的加速度命令 [m/s2]
        """
        return np.dot(self.ACCELERATION_PARAMETERS, self.acceleration_features(ego_vehicle, front_vehicle, rear_vehicle))

    def acceleration_features(self, ego_vehicle, front_vehicle=None, rear_vehicle=None):
        vt, dv, dp = 0, 0, 0
        if ego_vehicle:
            vt = ego_vehicle.target_velocity - ego_vehicle.velocity
            d_safe = self.DISTANCE_WANTED + np.max(ego_vehicle.velocity, 0) * self.TIME_WANTED + ego_vehicle.LENGTH
            if front_vehicle:
                d = ego_vehicle.lane_distance_to(front_vehicle)
                dv = min(front_vehicle.velocity - ego_vehicle.velocity, 0)
                dp = min(d - d_safe, 0)
        return np.array([vt, dv, dp])

    def steering_control(self, target_lane_index):
        """
            关于参数的线性控制器。 覆盖非线性控制器 ControlledVehicle.steering_control() 方法
            :param target_lane_index: 跟随的车道的索引
            :return: 方向盘角度命令 [弧度]
        """
        steering_angle = np.dot(np.array(self.STEERING_PARAMETERS), self.steering_features(target_lane_index))
        steering_angle = np.clip(steering_angle, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)
        return steering_angle

    def steering_features(self, target_lane_index):
        """
        用于追随车道的特征集合
        :param target_lane_index: 要追随的车道的索引
        :return: 特征数组

        """
        lane = self.road.network.get_lane(target_lane_index)
        lane_coords = lane.local_coordinates(self.position)
        lane_next_coords = lane_coords[0] + self.velocity * self.PURSUIT_TAU
        lane_future_heading = lane.heading_at(lane_next_coords)
        features = np.array([utils.wrap_to_pi(lane_future_heading - self.heading) *
                             self.LENGTH / utils.not_zero(self.velocity),
                             -lane_coords[1] * self.LENGTH / (utils.not_zero(self.velocity) ** 2)])
        return features


class AggressiveVehicle(LinearVehicle):
    LANE_CHANGE_MIN_ACC_GAIN = 1.0  # [m/s2]
    MERGE_ACC_GAIN = 0.8
    MERGE_VEL_RATIO = 0.75
    MERGE_TARGET_VEL = 30
    ACCELERATION_PARAMETERS = [MERGE_ACC_GAIN / ((1 - MERGE_VEL_RATIO) * MERGE_TARGET_VEL),
                               MERGE_ACC_GAIN / (MERGE_VEL_RATIO * MERGE_TARGET_VEL),
                               0.5]


class DefensiveVehicle(LinearVehicle):
    LANE_CHANGE_MIN_ACC_GAIN = 1.0  # [m/s2]
    MERGE_ACC_GAIN = 1.2
    MERGE_VEL_RATIO = 0.75
    MERGE_TARGET_VEL = 30
    ACCELERATION_PARAMETERS = [MERGE_ACC_GAIN / ((1 - MERGE_VEL_RATIO) * MERGE_TARGET_VEL),
                               MERGE_ACC_GAIN / (MERGE_VEL_RATIO * MERGE_TARGET_VEL),
                               2.0]
