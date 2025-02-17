from __future__ import division, print_function
from abc import ABCMeta, abstractmethod
import numpy as np

from NGSIM_env import utils
from NGSIM_env.vehicle.dynamics import Vehicle
from copy import deepcopy
from typing import Tuple, List, Optional, Union
import numpy as np
from NGSIM_env.cubic_spline import Spline2D
# from NGSIM_env.road.spline import LinearSpline2D
from NGSIM_env.utils import wrap_to_pi, Vector, get_class_path, class_from_path


class AbstractLane(object):
    """
        道路上的车道，由其中心曲线描述。
    """
    metaclass__ = ABCMeta
    DEFAULT_WIDTH = 5
    VEHICLE_LENGTH = 5
    length: float = 0
    line_type: List["LineType"]

    @abstractmethod
    def position(self, longitudinal, lateral):
        """
        将局部车道坐标转换为世界位置。

        :param longitudinal: 纵向车道坐标[m]
        :param lateral: 横向车道坐标[m]
        :return: 对应的世界位置[m]
        """
        raise NotImplementedError()

    @abstractmethod
    def local_coordinates(self, position):
        """
            将世界位置转换为局部车道坐标。

        :param position: 一个世界位置[m]
        :return: (纵向，横向) 车道坐标[m]
        """
        raise NotImplementedError()

    @abstractmethod
    def heading_at(self, longitudinal):
        """
            获取给定纵向车道坐标的车道航向角。

        :param longitudinal: 纵向车道坐标[m]
        :return: 车道航向角[rad]
        """
        raise NotImplementedError()

    @abstractmethod
    def width_at(self, longitudinal):
        """
            获取给定纵向车道坐标处的车道宽度。

        :param longitudinal: 纵向车道坐标[m]
        :return: 车道宽度[m]
        """
        raise NotImplementedError()

    def on_lane(self, position, longitudinal=None, lateral=None, margin=0):
        """
            判断给定的世界位置是否在车道上。

        :param position: 一个世界位置[m]
        :param longitudinal: (可选) 对应的纵向车道坐标，如果已知 [m]
        :param lateral: (可选) 对应的横向车道坐标，如果已知 [m]
        :param margin: (可选) 车道宽度周围的额外边距
        :return: 是否在车道上的判断结果
        """
        if not longitudinal or not lateral:
            longitudinal, lateral = self.local_coordinates(position)
        is_on = np.abs(lateral) <= self.width_at(longitudinal) / 2 + margin and \
            -Vehicle.LENGTH <= longitudinal < self.length + Vehicle.LENGTH
        return is_on

    def is_reachable_from(self, position):
        """
        判断从给定的世界位置是否可以到达车道。

        :param position: 世界位置 [m]
        :return: 是否可以到达车道的判断结果
        """
        if self.forbidden:
            return False
        longitudinal, lateral = self.local_coordinates(position)
        is_close = np.abs(lateral) <= 2 * self.width_at(longitudinal) and 0 <= longitudinal < self.length + Vehicle.LENGTH
        return is_close

    def after_end(self, position, longitudinal=None, lateral=None):
        if not longitudinal:
            longitudinal, _ = self.local_coordinates(position)
        return longitudinal > self.length - Vehicle.LENGTH / 2

    def distance(self, position):
        """
        计算从位置到车道的L1距离[m]。
        """
        s, r = self.local_coordinates(position)
        return abs(r) + max(s - self.length, 0) + max(0 - s, 0)

    def distance_with_heading(self, position: np.ndarray, heading: Optional[float], heading_weight: float = 1.0):
        if heading is None:
            return self.distance(position)
        else:
            s, r = self.local_coordinates(position)
            angle = np.abs(self.local_angle(heading, s))
            return abs(r) + max(s - self.length, 0) + max(0 - s, 0) + heading_weight * angle

    def local_angle(self, heading: float, long_offset: float):
        return wrap_to_pi(heading - self.heading_at(long_offset))

    def steering_control(self, position, velocity, heading, LENGTH, PURSUIT_TAU, KP_LATERAL, KP_HEADING, MAX_STEERING_ANGLE):

        lane_coords = self.local_coordinates(position)
        lane_next_coords = lane_coords[0] + velocity * PURSUIT_TAU
        lane_future_heading = self.heading_at(lane_next_coords)

        # Lateral position control
        lateral_velocity_command = - KP_LATERAL * lane_coords[1]

        # Lateral velocity to heading
        heading_command = np.arcsin(np.clip(lateral_velocity_command/utils.not_zero(velocity), -1, 1))
        heading_ref = lane_future_heading + np.clip(heading_command, -np.pi/4, np.pi/4)

        # Heading control
        heading_rate_command = KP_HEADING * utils.wrap_to_pi(heading_ref - heading)

        # Heading rate to steering angle
        steering_angle = np.arctan(LENGTH / utils.not_zero(velocity) * heading_rate_command)
        steering_angle = np.clip(steering_angle, -MAX_STEERING_ANGLE, MAX_STEERING_ANGLE)

        return steering_angle

    def velocity_control(self, velocity, target_velocity, KP_A):
        """
        控制车辆的速度。

        使用简单的比例控制器。

        :param target_velocity: 目标速度
        :return: 加速度命令[m/s²]
        """
        return KP_A * (target_velocity - velocity)

    def acceleration(self, ego_vehicle, front_vehicle=None, rear_vehicle=None, position=None, DISTANCE_WANTED=None, TIME_WANTED=None, DELTA=None,
                     COMFORT_ACC_MAX=None, COMFORT_ACC_MIN=None):
        """
        使用智能驾驶模型计算一个加速度命令。

        加速度的选择是为了：
        - 达到目标速度；
        - 与前车保持最小安全距离（和安全时间）。

        :param ego_vehicle: 需要计算期望加速度的车辆。它不一定是一个IDM车辆，因此这个方法是一个类方法。这使得IDM车辆能够考虑其他车辆的行为，即使它们不是IDM车辆。
        :param front_vehicle: 前车
        :param rear_vehicle: 后车
        :return: 车辆的加速度命令[m/s²]

        """
        if not ego_vehicle:
            return 0

        ego_target_velocity = utils.not_zero(getattr(ego_vehicle, "target_velocity", 0))
        acceleration = COMFORT_ACC_MAX * (1 - np.power(max(ego_vehicle.velocity, 0) / ego_target_velocity, DELTA))
        # if acceleration.shape[0] > 1:
        #     print(acceleration)
        if front_vehicle:
            # p1 = front_vehicle.position
            d = self.local_coordinates(front_vehicle.position)[0] - self.local_coordinates(position)[0]
            acceleration -= COMFORT_ACC_MAX * np.power(self.desired_gap(ego_vehicle, front_vehicle, DISTANCE_WANTED, TIME_WANTED,
                                                                        COMFORT_ACC_MAX, COMFORT_ACC_MIN) / utils.not_zero(d), 2)

        return acceleration

    def desired_gap(self, ego_vehicle, front_vehicle=None, DISTANCE_WANTED=None, TIME_WANTED=None, COMFORT_ACC_MAX=None, COMFORT_ACC_MIN=None):
        """
        计算车辆与其前车之间的期望距离。

        :param ego_vehicle: 正在控制的车辆
        :param front_vehicle: 它的前车
        :return: 两辆车之间的期望距离
        """
        d0 = DISTANCE_WANTED + ego_vehicle.LENGTH / 2 + front_vehicle.LENGTH / 2
        tau = TIME_WANTED
        ab = -COMFORT_ACC_MAX * COMFORT_ACC_MIN
        dv = ego_vehicle.velocity - front_vehicle.velocity
        d_star = d0 + ego_vehicle.velocity * tau + ego_vehicle.velocity * dv / (2 * np.sqrt(ab))

        return d_star


class LineType:
    """
    A lane side line type.
    """
    NONE = 0
    STRIPED = 1
    CONTINUOUS = 2
    CONTINUOUS_LINE = 3


class StraightLane(AbstractLane):
    """
    A lane going in straight line.
    """
    def __init__(self, start, end, width=AbstractLane.DEFAULT_WIDTH, line_types=None, forbidden=False, speed_limit=20, priority=0):
        """
        新建直行车道。

        :param start: 车道的起始位置 [m]
        :param end: 车道的结束位置 [m]
        :param width: 车道宽度 [m]
        :param line_types: 车道两侧线的类型
        :param forbidden: 是否禁止切换到这个车道
        :param priority: 车道的优先级，用于确定谁有优先权
        """
        super(StraightLane, self).__init__()
        self.start = np.array(start)
        self.end = np.array(end)
        self.width = width
        self.heading = np.arctan2(self.end[1] - self.start[1], self.end[0] - self.start[0])
        self.length = np.linalg.norm(self.end - self.start)
        self.line_types = line_types or [LineType.STRIPED, LineType.STRIPED]
        self.direction = (self.end - self.start) / self.length
        self.direction_lateral = np.array([-self.direction[1], self.direction[0]])
        self.forbidden = forbidden
        self.priority = priority
        self.speed_limit = speed_limit

    def position(self, longitudinal, lateral):
        return self.start + longitudinal * self.direction + lateral * self.direction_lateral

    def heading_at(self, s):
        return self.heading

    def width_at(self, s):
        return self.width

    def local_coordinates(self, position):
        delta = position - self.start
        longitudinal = np.dot(delta, self.direction)
        lateral = np.dot(delta, self.direction_lateral)
        return longitudinal, lateral


class SineLane(StraightLane):
    """
    A sinusoidal lane
    """

    def __init__(self, start, end, amplitude, pulsation, phase,
                 width=StraightLane.DEFAULT_WIDTH, line_types=None, forbidden=False, speed_limit=20, priority=0):
        """
            新的正弦波车道。

        :param start: 车道起始位置 [m]
        :param end: 车道结束位置 [m]
        :param amplitude: 车道振幅 [m]
        :param pulsation: 车道脉动 [rad/m]
        :param phase: 车道初始相位 [rad]
        """
        super(SineLane, self).__init__(start, end,  width, line_types, forbidden, speed_limit, priority)
        self.amplitude = amplitude
        self.pulsation = pulsation
        self.phase = phase

    def position(self, longitudinal, lateral):
        return super(SineLane, self).position(longitudinal, lateral
                                              + self.amplitude * np.sin(self.pulsation * longitudinal + self.phase))

    def heading_at(self, s):
        return super(SineLane, self).heading_at(s) + np.arctan(
            self.amplitude * self.pulsation * np.cos(self.pulsation * s + self.phase))

    def local_coordinates(self, position):
        longitudinal, lateral = super(SineLane, self).local_coordinates(position)
        return longitudinal, lateral - self.amplitude * np.sin(self.pulsation * longitudinal + self.phase)


class CircularLane(AbstractLane):
    """
        一个走圆弧的车道。
    """
    def __init__(self, center, radius, start_phase, end_phase, clockwise=True,
                 width=AbstractLane.DEFAULT_WIDTH, line_types=None, forbidden=False, speed_limit=20, priority=0):
        super(CircularLane, self).__init__()
        self.center = np.array(center)
        self.radius = radius
        self.start_phase = start_phase
        self.end_phase = end_phase
        self.direction = 1 if clockwise else -1
        self.width = width
        self.line_types = line_types or [LineType.STRIPED, LineType.STRIPED]
        self.forbidden = forbidden
        self.length = radius*(end_phase - start_phase) * self.direction
        self.priority = priority
        self.speed_limit = speed_limit

    def position(self, longitudinal, lateral):
        phi = self.direction * longitudinal / self.radius + self.start_phase
        return self.center + (self.radius - lateral * self.direction)*np.array([np.cos(phi), np.sin(phi)])

    def heading_at(self, s):
        phi = self.direction * s / self.radius + self.start_phase
        psi = phi + np.pi/2 * self.direction
        return psi

    def width_at(self, s):
        return self.width

    def local_coordinates(self, position):
        delta = position - self.center
        phi = np.arctan2(delta[1], delta[0])
        phi = self.start_phase + utils.wrap_to_pi(phi - self.start_phase)
        r = np.linalg.norm(delta)
        longitudinal = self.direction*(phi - self.start_phase)*self.radius
        lateral = self.direction*(self.radius - r)
        return longitudinal, lateral


class PolyLaneFixedWidth(AbstractLane):
    """
    由一组点定义的固定宽度车道，使用二维Hermite多项式进行近似。
    """

    def __init__(
        self,
        lane_points: List[Tuple[float, float]],
        width: float = AbstractLane.DEFAULT_WIDTH,
        line_types: Tuple[LineType, LineType] = None,
        forbidden: bool = False,
        speed_limit: float = 20,
        priority: int = 0,
    ) -> None:
        self.lane_points = lane_points
        x_list = np.array(lane_points)[:, 0]
        y_list = np.array(lane_points)[:, 1]
        self.curve = Spline2D(x_list, y_list)
        self.length = self.curve.s[-1]
        self.width = width
        self.line_types = line_types
        self.forbidden = forbidden
        self.speed_limit = speed_limit
        self.priority = priority

    def position(self, longitudinal: float, lateral: float) -> np.ndarray:
        x, y = self.curve.frenet_to_cartesian1D(longitudinal, 0)
        yaw = self.heading_at(longitudinal)
        # out = np.array([x - np.sin(yaw) * lateral, y + np.cos(yaw) * lateral])
        return np.array([x - np.sin(yaw) * lateral, y + np.cos(yaw) * lateral])

    def local_coordinates(self, position: np.ndarray) -> Tuple[float, float]:
        lon, lat = self.curve.cartesian_to_frenet1D(position[0], position[1])
        return lon, lat

    def heading_at(self, longitudinal: float) -> float:
        return self.curve.calc_yaw(longitudinal)

    def width_at(self, longitudinal: float) -> float:
        return self.width

    @classmethod
    def from_config(cls, config: dict):
        return cls(**config)

    def to_config(self) -> dict:
        return {
            "class_name": self.__class__.__name__,
            "config": {
                "lane_points": _to_serializable(
                    [_to_serializable(p.position) for p in self.curve.poses]
                ),
                "width": self.width,
                "line_types": self.line_types,
                "forbidden": self.forbidden,
                "speed_limit": self.speed_limit,
                "priority": self.priority,
            },
        }


class PolyLane(PolyLaneFixedWidth):
    """
    由一组点定义的车道，使用二维Hermite多项式进行近似。
    """

    def __init__(
        self,
        lane_points: List[Tuple[float, float]],
        left_boundary_points: List[Tuple[float, float]],
        right_boundary_points: List[Tuple[float, float]],
        line_types: Tuple[LineType, LineType] = None,
        forbidden: bool = False,
        speed_limit: float = 20,
        priority: int = 0
    ):
        super().__init__(
            lane_points=lane_points,
            line_types=line_types,
            forbidden=forbidden,
            speed_limit=speed_limit,
            priority=priority,
        )
        self.lane_points = lane_points
        self.left_boundary_points = left_boundary_points
        self.right_boundary_points = right_boundary_points
        right_boundary_points = np.array(right_boundary_points)
        self.right_boundary = Spline2D(right_boundary_points[:, 0], right_boundary_points[:, 1])

        left_boundary_points = np.array(left_boundary_points)
        self.left_boundary = Spline2D(left_boundary_points[:, 0], left_boundary_points[:, 1])
        self._init_width()

    def width_at(self, longitudinal: float) -> float:
        if longitudinal < 0:
            return self.width_samples[0]
        elif longitudinal > len(self.width_samples) - 1:
            return self.width_samples[-1]
        else:
            return self.width_samples[int(longitudinal)]

    def _width_at_s(self, longitudinal: float) -> float:
        """
        通过计算给定s值处中心线与每个边界之间的最小距离来计算宽度。这样可以弥补边界线上的凹痕。
        """

        center_x, center_y = self.position(longitudinal, 0)
        right_x, right_y = self.right_boundary.frenet_to_cartesian1D(
            self.right_boundary.cartesian_to_frenet1D(center_x, center_y)[0], 0
        )
        left_x, left_y = self.left_boundary.frenet_to_cartesian1D(
            self.left_boundary.cartesian_to_frenet1D(center_x, center_y)[0], 0
        )

        dist_to_center_right = np.linalg.norm(
            np.array([right_x, right_y]) - np.array([center_x, center_y])
        )
        dist_to_center_left = np.linalg.norm(
            np.array([left_x, left_y]) - np.array([center_x, center_y])
        )
        # if self.straight_line:
        return dist_to_center_left + dist_to_center_right
        # else:
        #     return max(
        #         min(dist_to_center_right, dist_to_center_left) * 2,
        #         AbstractLane.DEFAULT_WIDTH,
        #     )

    def _init_width(self):
        """
        预先计算在大约1m距离内采样的宽度值，以减少运行时的计算量。假设在1-2m范围内，宽度不会显著变化。
        使用numpy的linspace函数可以确保采样中包含最小和最大的s值。
        """
        s_samples = np.linspace(
            0,
            self.curve.s[-1],
            num=int(np.ceil(self.curve.s[-1])) + 1,
        )
        self.width_samples = [self._width_at_s(s) for s in s_samples]

    def to_config(self) -> dict:
        config = super().to_config()

        ordered_boundary_points = _to_serializable(
            [_to_serializable(p.position) for p in reversed(self.left_boundary.poses)]
        )
        ordered_boundary_points += _to_serializable(
            [_to_serializable(p.position) for p in self.right_boundary.poses]
        )

        config["class_name"] = self.__class__.__name__
        config["config"]["ordered_boundary_points"] = ordered_boundary_points
        del config["config"]["width"]

        return config


def _to_serializable(arg: Union[np.ndarray, List]) -> List:
    if isinstance(arg, np.ndarray):
        return arg.tolist()
    return arg


def lane_from_config(cfg: dict) -> AbstractLane:
    return class_from_path(cfg["class_path"])(**cfg["config"])