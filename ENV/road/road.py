from __future__ import division, print_function
import numpy as np
import pandas as pd
import logging
from typing import Optional
from NGSIM_env.logger import Loggable
from NGSIM_env.road.lane import LineType, StraightLane
from NGSIM_env.vehicle.dynamics import Obstacle
from NGSIM_env.vehicle.humandriving import HumanLikeVehicle, NGSIMVehicle, IntersectionHumanLikeVehicle, Pedestrian
from NGSIM_env.vehicle.control import MDPVehicle

logger = logging.getLogger(__name__)


class RoadNetwork(object):
    def __init__(self):
        self.graph = {}

    def add_node(self, node):
        """
        节点代表道路网络中的一个交叉口。
        :param node: 节点标签。
        """
        if node not in self.graph:
            self.graph[node] = []

    def add_lane(self, _from, _to, lane):
        """
        车道在道路网络中被编码为一条边。
        :param _from: 车道起始点的节点。
        :param _to: 车道结束点的节点。
        :param AbstractLane lane: 车道的几何形状。
        """
        if _from not in self.graph:
            self.graph[_from] = {}
        if _to not in self.graph[_from]:
            self.graph[_from][_to] = []
        self.graph[_from][_to].append(lane)

    def get_lane(self, index):
        """
        在道路网络中根据给定的索引获取车道的几何形状。
        :param index: 一个元组（起始节点，目标节点，车道在道路上的id）。
        :return: 相应的车道几何形状。
        """
        _from, _to, _id = index
        if _id is None and len(self.graph[_from][_to]) == 1:
            _id = 0
        return self.graph[_from][_to][_id]

    def get_closest_lane_index(self, position, heading: Optional[float] = None):
        """
        获取最接近世界位置的车道的索引。

        :param position: 世界位置 [m]。
        :param heading: 方向角 [rad]
        :return: 最接近车道的索引。
        """
        indexes, distances = [], []
        for _from, to_dict in self.graph.items():
            for _to, lanes in to_dict.items():
                for _id, l in enumerate(lanes):
                    distances.append(l.distance_with_heading(position, heading))
                    indexes.append((_from, _to, _id))
                    
        return indexes[int(np.argmin(distances))]

    def next_lane(self, current_index, route=None, position=None, np_random=np.random):
        """
        获取在完成当前车道后应该跟随的下一个车道的索引。

        如果有可用的计划并且与当前车道匹配，则跟随该计划。
        否则，随机选择下一条道路。
        如果它与当前道路具有相同数量的车道，则保持在同一车道。
        否则，选择下一条道路中最接近的车道。
        :param current_index: 当前车道的索引。
        :param route: 计划的路线，如果有的话。
        :param position: 车辆位置。
        :param np_random: 随机数源。
        :return: 当完成当前车道后要跟随的下一个车道的索引。
        """
        _from, _to, _id = current_index
        next_to = None
        # Pick next road according to planned route
        if route:
            if route[0][:2] == current_index[:2]:  # We just finished the first step of the route, drop it.
                route.pop(0)
            if route and route[0][0] == _to:  # Next road in route is starting at the end of current road.
                _, next_to, route_id = route[0]
            elif route:
                logger.warning("Route {} does not start after current road {}.".format(route[0], current_index))
        # Randomly pick next road
        if not next_to:
            try:
                keys = list(self.graph[_to].keys())
                next_to = list(self.graph[_to].keys())[np_random.randint(len(self.graph[_to]))]
                if 'merge_out' in keys:
                    next_to = 's4'
                    # print(keys)
            except KeyError:
                # logger.warning("End of lane reached.")
                return current_index

        # If next road has same number of lane, stay on the same lane
        if len(self.graph[_from][_to]) == len(self.graph[_to][next_to]):
            next_id = _id
        # Else, pick closest lane
        else:
            lanes = range(len(self.graph[_to][next_to]))
            next_id = min(lanes,  key=lambda l: self.get_lane((_to, next_to, l)).distance(position))

        return _to, next_to, next_id

    def bfs_paths(self, start, goal):
        """
        从起始节点到目标节点的广度优先搜索所有路径。

        :param start: 起始节点
        :param goal: 目标节点
        :return: 从起始节点到目标节点的路径列表。
        """
        queue = [(start, [start])]
        while queue:
            (node, path) = queue.pop(0)
            if node not in self.graph:
                yield []
            for _next in set(self.graph[node].keys()) - set(path):
                if _next == goal:
                    yield path + [_next]
                elif _next in self.graph:
                    queue.append((_next, path + [_next]))

    def shortest_path(self, start, goal):
        """
        从起点到目标节点的广度优先搜索最短路径。

        :param start: 起始节点
        :param goal: 目标节点
        :return: 从起点到目标节点的最短路径。
        """
        try:
            return next(self.bfs_paths(start, goal))
        except StopIteration:
            return None

    def all_side_lanes(self, lane_index):
        """
        :param lane_index: 车道的索引。
        :return: 属于同一条道路的所有车道的索引。
        """
        return self.graph[lane_index[0]][lane_index[1]]

    def side_lanes(self, lane_index):
        """
        :param lane_index: 车道的索引。
        :return: 输入车道旁边的车道的索引，即其左侧或右侧。
        """
        _from, _to, _id = lane_index
        lanes = []
        if _id > 0:
            lanes.append((_from, _to, _id - 1))
        if _id < len(self.graph[_from][_to]) - 1:
            lanes.append((_from, _to, _id + 1))
        return lanes

    @staticmethod
    def is_same_road(lane_index_1, lane_index_2, same_lane=False):
        """
        车道1和车道2是同一条道路上的吗？
        """
        return lane_index_1[:2] == lane_index_2[:2] and (not same_lane or lane_index_1[2] == lane_index_2[2])

    @staticmethod
    def is_leading_to_road(lane_index_1, lane_index_2, same_lane=False):
        """
        车道1是否通向车道2？
        """
        return lane_index_1[1] == lane_index_2[0] and (not same_lane or lane_index_1[2] == lane_index_2[2])

    def is_connected_road(self, lane_index_1, lane_index_2, route=None, same_lane=False, depth=0):
        """
        车道2是否通向车道1的路线内的道路？

        需要考虑这些车道上的车辆是否会碰撞。
        :param lane_index_1：起始车道
        :param lane_index_2：目标车道
        :param route：起始车道的路线（如果有的话）
        :param same_lane：比较车道id
        :param depth：沿着车道1的路线搜索的深度
        :return：道路是否连接
        """
        if RoadNetwork.is_same_road(lane_index_2, lane_index_1, same_lane) \
           or RoadNetwork.is_leading_to_road(lane_index_2, lane_index_1, same_lane):
            return True
        if depth > 0:
            if route and route[0][:2] == lane_index_1[:2]:
                # 路线从当前道路开始，跳过它。
                return self.is_connected_road(lane_index_1, lane_index_2, route[1:], same_lane, depth)
            elif route and route[0][0] == lane_index_1[1]:
                # 继续按照当前道路行驶，沿着它前行。
                return self.is_connected_road(route[0], lane_index_2, route[1:], same_lane, depth - 1)
            else:
                # 递归地搜索交叉口的所有道路。
                _from, _to, _id = lane_index_1
                return any([self.is_connected_road((_to, l1_to, _id), lane_index_2, route, same_lane, depth - 1)
                            for l1_to in self.graph.get(_to, {}).keys()])
        return False

    def lanes_list(self):
        return [lane for tos in self.graph.values() for ids in tos.values() for lane in ids]

    @staticmethod
    def straight_road_network(lanes=4, length=1000):
        net = RoadNetwork()
        for lane in range(lanes):
            origin = [0, lane * StraightLane.DEFAULT_WIDTH]
            end = [length, lane * StraightLane.DEFAULT_WIDTH]
            line_types = [LineType.CONTINUOUS_LINE if lane == 0 else LineType.STRIPED,
                          LineType.CONTINUOUS_LINE if lane == lanes - 1 else LineType.NONE]
            net.add_lane(0, 1, StraightLane(origin, end, line_types=line_types))

        return net

    def position_heading_along_route(self, route, longitudinal, lateral):
        """
        获取某些局部坐标系下由多条车道组成的路径上的绝对位置和朝向。
        :param route: 计划路径，车道索引的列表
        :param longitudinal: 纵向位置
        :param lateral: 横向位置
        :return: 位置，朝向
        """
        while len(route) > 1 and longitudinal > self.get_lane(route[0]).length:
            longitudinal -= self.get_lane(route[0]).length
            route = route[1:]
            
        return self.get_lane(route[0]).position(longitudinal, lateral), self.get_lane(route[0]).heading_at(longitudinal)


class Road(Loggable):
    """
    一条道路是一组车道，上面有一组车辆行驶。
    """

    def __init__(self, network=None, vehicles=None, np_random=None, record_history=False, lanelet=None):
        """
        新的道路。

        :param network: 描述车道的道路网络
        :param vehicles: 在道路上行驶的车辆
        :param np.random.RandomState np_random: 用于车辆行为的随机数生成器
        :param record_history: 是否记录车辆的最近轨迹以供显示

        """
        self.lanelet = lanelet
        self.network = network or []
        self.vehicles = vehicles or []
        self.np_random = np_random if np_random else np.random.RandomState()
        self.record_history = record_history

    def close_vehicles_to(self, vehicle, distance, count=None, sort=False, see_behind=True):
        vehicles = [v for v in self.vehicles
                    if np.linalg.norm(v.position - vehicle.position) < distance
                    and v is not vehicle
                    and (see_behind or -2*vehicle.LENGTH < vehicle.lane_distance_to(v))]

        if sort:
            # vehicles = sorted(vehicles, key=lambda v: abs(vehicle.lane_distance_to(v)))
            vehicles = sorted(vehicles, key=lambda v: np.linalg.norm(v.position - vehicle.position))
        if count:
            vehicles = vehicles[:count]

        return vehicles

    def act(self, step):
        """
        决定道路上每个实体的动作。
        """
        for vehicle in self.vehicles:
            if (not vehicle.is_ego) or not isinstance(vehicle, MDPVehicle):
                if vehicle.controlled_by_model:
                    pass
                if isinstance(vehicle, NGSIMVehicle):
                    vehicle.act()
                else:
                    vehicle.act(step)

    def step(self, dt):
        """
        更新道路上每个实体的动力学。

        :param dt: 时间步长 [秒]

        """
        for vehicle in self.vehicles:
            vehicle.step(dt)

        ego_vehicle = None
        test_vehicle = None
        for vehicle in self.vehicles:
            if vehicle.is_ego:
                ego_vehicle = vehicle
            if vehicle.controlled_by_model:
                test_vehicle = vehicle

        for other in self.vehicles:
            if not isinstance(other, Pedestrian):

                ego_vehicle.check_collision(other)
                if test_vehicle is not None:
                    test_vehicle.check_collision(other)

    def neighbour_vehicles(self, vehicle, lane_index=None, margin=1):
        """
        找到给定车辆的前车和后车。
        :param vehicle: 需要找到邻居的车辆
        :param lane_index: 在其中查找前车和后车的车道。 它不一定是当前车道，也可以是另一个车道，此时车辆将根据车道上的本地坐标在其上进行投影。
        :return: 前车，后车
        """
        lane_index = lane_index or vehicle.lane_index
        if not lane_index:
            return None, None
        lane = self.network.get_lane(lane_index)
        s = self.network.get_lane(lane_index).local_coordinates(vehicle.position)[0]
        s_front = s_rear = None
        v_front = v_rear = None
        for v in self.vehicles:
            if v is not vehicle and True: #self.network.is_connected_road(v.lane_index, lane_index, same_lane=True):
                s_v, lat_v = lane.local_coordinates(v.position)
                if not lane.on_lane(v.position, s_v, lat_v, margin=margin):
                    continue
                if s <= s_v and (s_front is None or s_v <= s_front):
                    s_front = s_v
                    v_front = v
                if s_v < s and (s_rear is None or s_v > s_rear):
                    s_rear = s_v
                    v_rear = v

        return v_front, v_rear

    def dump(self):
        """
        清除所有在道路上的实体的数据
        """
        for v in self.vehicles:
            if not isinstance(v, Obstacle):
                v.dump()

    def get_log(self):
        """
        将道路上所有实体的日志连接起来。
        :返回值: 连接后的日志。
        """
        return pd.concat([v.get_log() for v in self.vehicles])

    def __repr__(self):
        return self.vehicles.__repr__()

