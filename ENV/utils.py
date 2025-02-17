from __future__ import division, print_function

import copy
import importlib
import itertools
from typing import Tuple, Dict, Callable, List, Optional, Union, Sequence
from matplotlib.animation import FFMpegWriter
import matplotlib
matplotlib.use('Agg')
import numpy as np
from scipy.interpolate import interp1d
import cv2
# Useful types
Vector = Union[np.ndarray, Sequence[float]]
Matrix = Union[np.ndarray, Sequence[Sequence[float]]]
Interval = Union[np.ndarray,
                 Tuple[Vector, Vector],
                 Tuple[Matrix, Matrix],
                 Tuple[float, float],
                 List[Vector],
                 List[Matrix],
                 List[float]]

import importlib
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import math
EPSILON = 0.01
from NGSIM_env.cubic_spline import Spline2D

def constrain(x, a, b):
    return np.minimum(np.maximum(x, a), b)

def lmap(v: float, x, y) -> float:
    """Linear map of value v with range x to desired range y."""
    return y[0] + (v - x[0]) * (y[1] - y[0]) / (x[1] - x[0])

def not_zero(x):
    if abs(x) > EPSILON:
        return x
    elif x > 0:
        return EPSILON
    else:
        return -EPSILON


def wrap_to_pi(x):
    return ((x+np.pi) % (2*np.pi)) - np.pi


def point_in_rectangle(point, rect_min, rect_max):
    """
    检查一个点是否在一个矩形内部
    :param point: 一个点 (x, y)
    :param rect_min: 矩形左下角坐标 (x_min, y_min)
    :param rect_max: 矩形右上角坐标 (x_max, y_max)
    """
    return rect_min[0] <= point[0] <= rect_max[0] and rect_min[1] <= point[1] <= rect_max[1]


def point_in_rotated_rectangle(point, center, length, width, angle):
    """
    检查一个点是否在一个旋转矩形内部
    :param point: 一个点
    :param center: 矩形中心坐标
    :param length: 矩形长度
    :param width: 矩形宽度
    :param angle: 矩形角度 [弧度]
    """
    c, s = np.cos(angle), np.sin(angle)
    r = np.array([[c, -s], [s, c]])
    ru = r.dot(point - center)
    return point_in_rectangle(ru, [-length/2, -width/2], [length/2, width/2])


def point_in_ellipse(point, center, angle, length, width):
    """
    检查一个点是否在一个椭圆内部
    :param point: 一个点
    :param center: 椭圆中心坐标
    :param angle: 椭圆主轴角度
    :param length: 椭圆长轴长度
    :param width: 椭圆短轴长度 """
    c, s = np.cos(angle), np.sin(angle)
    r = np.matrix([[c, -s], [s, c]])
    ru = r.dot(point - center)
    return np.sum(np.square(ru / np.array([length, width]))) < 1


def rotated_rectangles_intersect(rect1, rect2):
    """
    判断两个旋转矩形是否相交？
    :param rect1: (中心点，长度，宽度，旋转角度)
    :param rect2: (中心点，长度，宽度，旋转角度)
    """
    return has_corner_inside(rect1, rect2) or has_corner_inside(rect2, rect1)


def has_corner_inside(rect1, rect2):
    """
    检查 rect1 是否有一个角落位于 rect2 的内部
    :param rect1: (中心点，长度，宽度，旋转角度)
    :param rect2: (中心点，长度，宽度，旋转角度)
    """
    (c1, l1, w1, a1) = rect1
    (c2, l2, w2, a2) = rect2
    c1 = np.array(c1)
    l1v = np.array([l1/2, 0])
    w1v = np.array([0, w1/2])
    r1_points = np.array([[0, 0],
                          - l1v, l1v, -w1v, w1v,
                          - l1v - w1v, - l1v + w1v, + l1v - w1v, + l1v + w1v])
    c, s = np.cos(a1), np.sin(a1)
    r = np.array([[c, -s], [s, c]])
    rotated_r1_points = r.dot(r1_points.transpose()).transpose()
    return any([point_in_rotated_rectangle(c1+np.squeeze(p), c2, l2, w2, a2) for p in rotated_r1_points])


def calculate_car_boundary(x, y, w, h, angle):
    cos_val = math.cos(angle)
    sin_val = math.sin(angle)
    x1 = x - (w/2) * cos_val - (h/2)*sin_val
    y1 = y + (w/2) * sin_val - (h/2)*cos_val

    x2 = x + (w / 2) * cos_val - (h / 2) * sin_val
    y2 = y - (w / 2) * sin_val - (h / 2) * cos_val

    x3 = x + (w / 2) * cos_val + (h / 2) * sin_val
    y3 = y - (w / 2) * sin_val + (h / 2) * cos_val

    x4 = x - (w / 2) * cos_val + (h / 2) * sin_val
    y4 = y + (w / 2) * sin_val + (h / 2) * cos_val

    return [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]


def is_intersect(p0, p1, q0, q1):
    def cross_product(a, b):
        return a[0]*b[1] - a[1]*b[0]

    vec1 = (p1[0] - p0[0], p1[1]-p0[1])
    vec2 = (q1[0] - q0[0], q1[1]-q0[1])
    vec3 = (p0[0] - q0[0], p0[1]-q0[1])

    if cross_product(vec1, vec2) == 0:
        return False
    else:
        r = cross_product(vec1, vec3) / cross_product(vec1, vec2)
        s = cross_product(vec2, vec3) / cross_product(vec1, vec2)
        return (0 <= r <= 1) and (0 <= s <= 1)


def judge_car_collision(cx1, cy1, w1, h1, angle1, cx2, cy2, w2, h2, angle2):
    boundary1 = calculate_car_boundary(cx1, cy1, w1, h1, angle1)
    boundary2 = calculate_car_boundary(cx2, cy2, w2, h2, angle2)
    for i in range(4):
        j = (i+1) % 4
        for k in range(4):
            l = (k+1) % 4
            if is_intersect(boundary1[i], boundary1[j], boundary2[k], boundary2[l]):
                return True
    return False

def do_every(duration, timer):
    return duration < timer


def remap(v, x, y):
    return y[0] + (v-x[0])*(y[1]-y[0])/(x[1]-x[0])


def class_from_path(path):
    module_name, class_name = path.rsplit(".", 1)
    class_object = getattr(importlib.import_module(module_name), class_name)
    return class_object


def is_straight_line(point: np.ndarray, threshold: float = 0.5):
    flag = True

    if point.shape[0] <= 2:
        flag = False
    else:
        x = point[:, 0]
        y = point[:, 1]
        A = np.vstack([x, np.ones_like(x)]).T
        m, c = np.linalg.lstsq(A, y, rcond=-1)[0]
        error = np.abs(y - (m*x + c)).mean()
        if error >= threshold:
            flag = False

    return flag


def smooth_line(line: np.ndarray) -> np.ndarray:

    if line.shape[0] < 4:
        return line

    x = line[:, 0]
    is_increasing = all(x[i] <= x[i + 1] for i in range(x.shape[0] - 1))
    is_decreasing = all(x[i] >= x[i + 1] for i in range(x.shape[0] - 1))
    if not (is_decreasing or is_increasing):
        return line

    y = line[:, 1]
    interp_func = interp1d(x, y, kind='cubic')
    x_smooth = np.linspace(x[0], x[-1], x.shape[0])
    y_smooth = interp_func(x)
    y_smooth[0] = y[0]
    y_smooth[-1] = y[-1]
    return np.vstack((x, y_smooth))


def lane_a_to_b(graph: dict, b: list):
    a = []
    for k, v in graph.items():
        if b in v:
            a.append(eval(k))
    return a


# 解析osm格式数据，并转化为highway地图格式
def load_lanelet_map(laneletmap):

    way = dict()
    for ls in laneletmap.lineStringLayer:

        if "type" not in ls.attributes.keys():
            raise RuntimeError("ID " + str(ls.id) + ": Linestring type must be specified")
        elif ls.attributes["type"] == "curbstone":
            type_dict = dict(color="black", linewidth=1, zorder=10)
        elif ls.attributes["type"] == "line_thin":
            if "subtype" in ls.attributes.keys() and ls.attributes["subtype"] == "dashed":
                type_dict = dict(color="white", linewidth=1, zorder=10, dashes=[10, 10])
            else:
                type_dict = dict(color="white", linewidth=1, zorder=10)
        elif ls.attributes["type"] == "line_thick":
            if "subtype" in ls.attributes.keys() and ls.attributes["subtype"] == "dashed":
                type_dict = dict(color="white", linewidth=2, zorder=10, dashes=[10, 10])
            else:
                type_dict = dict(color="white", linewidth=2, zorder=10)
        elif ls.attributes["type"] == "pedestrian_marking":
            type_dict = dict(color="white", linewidth=1, zorder=10, dashes=[5, 10])
        elif ls.attributes["type"] == "bike_marking":
            type_dict = dict(color="white", linewidth=1, zorder=10, dashes=[5, 10])
        elif ls.attributes["type"] == "stop_line":
            type_dict = dict(color="white", linewidth=3, zorder=10)
        elif ls.attributes["type"] == "virtual":
            type_dict = dict(color="blue", linewidth=1, zorder=10, dashes=[2, 5])
        elif ls.attributes["type"] == "road_border":
            type_dict = dict(color="black", linewidth=1, zorder=10)
        elif ls.attributes["type"] == "guard_rail":
            type_dict = dict(color="black", linewidth=1, zorder=10)
        elif ls.attributes["type"] == "traffic_sign":
            continue
        elif ls.attributes["type"] == "building":
            type_dict = dict(color="pink", zorder=1, linewidth=5)
        elif ls.attributes["type"] == "spawnline":
            if ls.attributes["spawn_type"] == "start":
                type_dict = dict(color="green", zorder=11, linewidth=2)
            elif ls.attributes["spawn_type"] == "end":
                type_dict = dict(color="red", zorder=11, linewidth=2)
        else:
            continue
        points = [(p.x, p.y) for p in ls]
        attr = dict(ls.attributes)
        type_dict.update(attr)
        x_list = [p[0] for p in points]
        y_list = [p[1] for p in points]
        way[ls.id] = {'type': type_dict, 'points': points, 'spline': Spline2D(x_list, y_list)}

    roads = dict()
    # fig, ax = plt.subplots()
    nodes = set()
    edegs = set()
    graph = dict()
    indegree = set()
    outdegree = set()

    left_right_lane_id = set()
    for ll in laneletmap.laneletLayer:

        left = [[pt.x, pt.y] for pt in ll.leftBound]
        right = [[pt.x, pt.y] for pt in ll.rightBound]
        center = [[pt.x, pt.y] for pt in ll.centerline]

        start_point = center[0]
        end_point = center[-1]
        nodes.add(tuple(start_point))
        nodes.add(tuple(end_point))
        edegs.add((tuple(start_point), tuple(end_point)))
        if graph.get(str(tuple(start_point))) is not None:
            point_list = graph[str(tuple(start_point))]
            point_list.append(end_point)
            graph[str(tuple(start_point))] = point_list.copy()
        else:
            graph[str(tuple(start_point))] = [end_point]

        left_line_type = way[ll.leftBound.id]
        right_line_type = way[ll.rightBound.id]
        roads[ll.id] = {
            'left': left,
            'right': right,
            'center': center,
            'left_line_type': left_line_type,
            'right_line_type': right_line_type,
        }
        start_point = str(tuple(center[0]))
        end_point = str(tuple(center[-1]))
        indegree.add(end_point)
        outdegree.add(start_point)

    # # plt.show()
    # # plt.savefig("road_network.png")
    # roads_copy = roads.copy()
    # for k, road in roads.items():
    #     start_point = road['center'][0]
    #     end_point = road['center'][-1]
    #
    #     rear_end_points = graph.get(str(tuple(end_point)))
    #     if rear_end_points is None or road['straight_line'] or len(road['center']) < 10:
    #         continue
    #
    #     front_points = lane_a_to_b(graph, end_point)
    #
    #     if len(rear_end_points) == len(front_points) == 1:
    #         for k2, road2 in roads.items():
    #             rear_start_point = road2['center'][0]
    #             rear_end_point = road2['center'][-1]
    #             if k == k2 or end_point not in [rear_start_point]:
    #                 continue
    #
    #             if rear_end_point in rear_end_points:
    #                 left = road['left']
    #                 right = road['right']
    #                 center = road['center']
    #                 left.extend(road2['left'][1:])
    #                 right.extend(road2['right'][1:])
    #                 center.extend(road2['center'][1:])
    #                 roads_copy[k]['left'] = left
    #                 roads_copy[k]['right'] = right
    #                 roads_copy[k]['center'] = center
    #                 roads_copy[k]['straight_line'] = is_straight_line(np.array(center))
    #                 roads_copy.pop(k2)
    #                 # print(f"merge {k2} to {k}.")

    return roads, graph, way, indegree, outdegree


def estimate_sine_lane(traj_points, k=1):
    """
    根据车道轨迹点，估计正弦车道的振幅、脉动和初始相位。

    Args:
        traj_points (ndarray): [n x 2] 轨迹点数组（x, y）。
        k (float): 用于估计车道振幅的曲率标准差的倍数。

    Returns:
        amplitude (float): 正弦车道的振幅。
        pulsation (float): 正弦车道的脉动。
        phase (float): 正弦车道的初始相位。
    """
    # 计算曲率和朝向角度
    dx = np.gradient(traj_points[:, 0])
    ddx = np.gradient(dx)
    dy = np.gradient(traj_points[:, 1])
    ddy = np.gradient(dy)
    curvature = (dx * ddy - dy * ddx) / np.power(dx * dx + dy * dy, 1.5)
    heading = np.arctan2(dy, dx)

    # 计算曲率标准差和平均朝向角度
    std_curvature = np.std(curvature)
    mean_heading = np.mean(heading)

    # 估计车道振幅、脉动和相位
    amplitude = k * std_curvature
    pulsation = 2 * np.pi / np.mean(np.abs(curvature))
    phase = traj_points[0, 1] - amplitude * np.sin(pulsation * traj_points[0, 0] + mean_heading)

    return amplitude, pulsation, phase


def residuals(params, x, y):
    xc, yc, r = params
    return (x - xc)**2 + (y - yc)**2 - r**2


def fit_spline(points, nums_points=10):
    x = points[:, 0]
    y = points[:, 1]

    z1 = np.polyfit(x, y, 5)
    p1 = np.poly1d(z1)
    # print(p1)  # 在屏幕上打印拟合多项式
    u_new = np.linspace(x.min(), x.max(), nums_points)
    yvals = p1(u_new)  # 也可以使用yvals=np.polyval(z1,x)

    # plot1 = plt.plot(x, y, '*', label='original values')
    # plot2 = plt.plot(u_new, yvals, 'r', label='polyfit values')
    # plt.xlabel('xaxis')
    # plt.ylabel('yaxis')
    # plt.legend(loc=4)
    # plt.title('polyfitting')
    # plt.show()

    return np.column_stack([u_new, yvals])


def fit_circle(x, y):
    x_m = np.mean(x)
    y_m = np.mean(y)

    params0 = [x_m, y_m, 1.0]
    res = least_squares(residuals, params0, args=(x, y))

    xc, yc, r = res.x
    return (xc, yc, r)


def compute_angle(position: list, center: list = None):
    """
    求单位向量与[1, 0]逆时针的夹角，取值范围0-360度
    """
    assert len(position) == 2 and len(center) == 2

    if center is not None:
        position[0] = position[0] - center[0]
        position[1] = position[1] - center[1]

    norm = math.sqrt(position[0] ** 2 + position[1] ** 2)
    position[0] /= norm
    position[1] /= norm

    # 计算与[1,0]的夹角
    theta = math.acos(position[0]) * 180 / math.pi

    # 确定角度方向
    if position[1] >= 0:
        angle = theta
    else:
        angle = 360 - theta

    if angle > 180:
        angle -= 360
    elif angle < -180:
        angle += 360

    return angle


def do_every(duration: float, timer: float) -> bool:
    return duration < timer


def lmap(v: float, x: Interval, y: Interval) -> float:
    """Linear map of value v with range x to desired range y."""
    return y[0] + (v - x[0]) * (y[1] - y[0]) / (x[1] - x[0])


def get_class_path(cls: Callable) -> str:
    return cls.__module__ + "." + cls.__qualname__


def class_from_path(path: str) -> Callable:
    module_name, class_name = path.rsplit(".", 1)
    class_object = getattr(importlib.import_module(module_name), class_name)
    return class_object


def constrain(x: float, a: float, b: float) -> np.ndarray:
    return np.clip(x, a, b)


def not_zero(x: float, eps: float = 1e-2) -> float:
    if abs(x) > eps:
        return x
    elif x >= 0:
        return eps
    else:
        return -eps


def wrap_to_pi(x: float) -> float:
    return ((x + np.pi) % (2 * np.pi)) - np.pi


def point_in_rectangle(point: Vector, rect_min: Vector, rect_max: Vector) -> bool:
    """
    检查一个点是否在矩形内
    :param point: 一个点 (x, y)
    :param rect_min: x_min, y_min
    :param rect_max: x_max, y_max
    """
    return rect_min[0] <= point[0] <= rect_max[0] and rect_min[1] <= point[1] <= rect_max[1]


def point_in_rotated_rectangle(point: np.ndarray, center: np.ndarray, length: float, width: float, angle: float) \
        -> bool:
    """
    检查一个点是否在一个旋转矩形内
    :param point: 一个点
    :param center: 矩形中心
    :param length: 矩形长度
    :param width: 矩形宽度
    :param angle: 矩形角度 [弧度]
    :return: 点是否在矩形内部
    """
    c, s = np.cos(angle), np.sin(angle)
    r = np.array([[c, -s], [s, c]])
    ru = r.dot(point - center)
    return point_in_rectangle(ru, (-length/2, -width/2), (length/2, width/2))


def point_in_ellipse(point: Vector, center: Vector, angle: float, length: float, width: float) -> bool:
    """
    检查一个点是否在椭圆内
    :param point: 一个点
    :param center: 椭圆中心
    :param angle: 椭圆主轴角度
    :param length: 椭圆大轴
    :param width: 椭圆小轴
    :return: 点是否在椭圆内部
    """
    c, s = np.cos(angle), np.sin(angle)
    r = np.matrix([[c, -s], [s, c]])
    ru = r.dot(point - center)
    return np.sum(np.square(ru / np.array([length, width]))) < 1


def rotated_rectangles_intersect(rect1: Tuple[Vector, float, float, float],
                                 rect2: Tuple[Vector, float, float, float]) -> bool:
    """
    两个旋转矩形是否相交？
    :param rect1: (中心点坐标，长度，宽度，角度)
    :param rect2: (中心点坐标，长度，宽度，角度)
    :return: 是否相交？
    """
    return has_corner_inside(rect1, rect2) or has_corner_inside(rect2, rect1)


def rect_corners(center: np.ndarray, length: float, width: float, angle: float,
                 include_midpoints: bool = False, include_center: bool = False) -> List[np.ndarray]:
    """
    返回矩形的角点位置。
    :param center: 矩形中心点
    :param length: 矩形长度
    :param width: 矩形宽度
    :param angle: 矩形角度
    :param include_midpoints: 是否包括边缘的中点
    :param include_center: 是否包括矩形的中心点
    :return: 位置列表 """
    center = np.array(center)
    half_l = np.array([length/2, 0])
    half_w = np.array([0, width/2])
    corners = [- half_l - half_w,
               - half_l + half_w,
               + half_l + half_w,
               + half_l - half_w]
    if include_center:
        corners += [[0, 0]]
    if include_midpoints:
        corners += [- half_l, half_l, -half_w, half_w]

    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s], [s, c]])
    return (rotation @ np.array(corners).T).T + np.tile(center, (len(corners), 1))


def has_corner_inside(rect1: Tuple[Vector, float, float, float],
                      rect2: Tuple[Vector, float, float, float]) -> bool:
    """
    检查 rect1 是否有一个角点在 rect2 内部
    :param rect1: (中心点坐标，长度，宽度，角度)
    :param rect2: (中心点坐标，长度，宽度，角度)
    """
    return any([point_in_rotated_rectangle(p1, *rect2)
                for p1 in rect_corners(*rect1, include_midpoints=True, include_center=True)])


def project_polygon(polygon: Vector, axis: Vector) -> Tuple[float, float]:
    min_p, max_p = None, None
    for p in polygon:
        projected = p.dot(axis)
        if min_p is None or projected < min_p:
            min_p = projected
        if max_p is None or projected > max_p:
            max_p = projected
    return min_p, max_p


def interval_distance(min_a: float, max_a: float, min_b: float, max_b: float):
    """ 计算 [minA, maxA] 和 [minB, maxB] 之间的距离 如果区间重叠，距离将为负数 """
    return min_b - max_a if min_a < min_b else min_a - max_b


def are_polygons_intersecting(a: Vector, b: Vector,
                              displacement_a: Vector, displacement_b: Vector) \
        -> Tuple[bool, bool, Optional[np.ndarray]]:
    """
    检查两个多边形是否相交。 参考链接：https://www.codeproject.com/Articles/15573/2D-Polygon-Collision-Detection
    :param a: 多边形 A，作为 [x, y] 点的列表
    :param b: 多边形 B，作为 [x, y] 点的列表
    :param displacement_a: 多边形 A 的速度
    :param displacement_b: 多边形 B 的速度
    :return: 相交与否、将要相交、平移向量
    """
    intersecting = will_intersect = True
    min_distance = np.inf
    translation, translation_axis = None, None
    for polygon in [a, b]:
        for p1, p2 in zip(polygon, polygon[1:]):
            normal = np.array([-p2[1] + p1[1], p2[0] - p1[0]])
            normal /= np.linalg.norm(normal)
            min_a, max_a = project_polygon(a, normal)
            min_b, max_b = project_polygon(b, normal)

            if interval_distance(min_a, max_a, min_b, max_b) > 0:
                intersecting = False

            velocity_projection = normal.dot(displacement_a - displacement_b)
            if velocity_projection < 0:
                min_a += velocity_projection
            else:
                max_a += velocity_projection

            distance = interval_distance(min_a, max_a, min_b, max_b)
            if distance > 0:
                will_intersect = False
            if not intersecting and not will_intersect:
                break
            if abs(distance) < min_distance:
                min_distance = abs(distance)
                d = a[:-1].mean(axis=0) - b[:-1].mean(axis=0)  # center difference
                translation_axis = normal if d.dot(normal) > 0 else -normal

    if will_intersect:
        translation = min_distance * translation_axis
    return intersecting, will_intersect, translation


def confidence_ellipsoid(data: Dict[str, np.ndarray], lambda_: float = 1e-5, delta: float = 0.1, sigma: float = 0.1,
                         param_bound: float = 1.0) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    计算参数 theta 上的置信椭球，其中 y = theta^T phi
    :param data: 一个字典，格式为 {"features": [phi_0,...,phi_N], "outputs": [y_0,...,y_N]}
    :param lambda_: l2 正则化参数
    :param delta: 置信水平
    :param sigma: 噪声协方差
    :param param_bound: 参数范数的上界
    :return: 估计的 theta，Gram 矩阵 G_N_lambda，半径 beta_N """
    phi = np.array(data["features"])
    y = np.array(data["outputs"])
    g_n_lambda = 1/sigma * np.transpose(phi) @ phi + lambda_ * np.identity(phi.shape[-1])
    theta_n_lambda = np.linalg.inv(g_n_lambda) @ np.transpose(phi) @ y / sigma
    d = theta_n_lambda.shape[0]
    beta_n = np.sqrt(2*np.log(np.sqrt(np.linalg.det(g_n_lambda) / lambda_ ** d) / delta)) + \
        np.sqrt(lambda_*d) * param_bound
    return theta_n_lambda, g_n_lambda, beta_n


def confidence_polytope(data: dict, parameter_box: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    计算参数 theta 上的置信多面体，其中 y = theta^T phi
    :param data: 一个字典，格式为 {"features": [phi_0,...,phi_N], "outputs": [y_0,...,y_N]}
    :param parameter_box: 一个包含参数 theta 的盒子 [theta_min, theta_max]
    :return: 估计的 theta，多面体的顶点，Gram 矩阵 G_N_lambda，半径 beta_N """
    param_bound = np.amax(np.abs(parameter_box))
    theta_n_lambda, g_n_lambda, beta_n = confidence_ellipsoid(data, param_bound=param_bound)

    values, pp = np.linalg.eig(g_n_lambda)
    radius_matrix = np.sqrt(beta_n) * np.linalg.inv(pp) @ np.diag(np.sqrt(1 / values))
    h = np.array(list(itertools.product([-1, 1], repeat=theta_n_lambda.shape[0])))
    d_theta = np.array([radius_matrix @ h_k for h_k in h])

    # Clip the parameter and confidence region within the prior parameter box.
    theta_n_lambda = np.clip(theta_n_lambda, parameter_box[0], parameter_box[1])
    for k, _ in enumerate(d_theta):
        d_theta[k] = np.clip(d_theta[k], parameter_box[0] - theta_n_lambda, parameter_box[1] - theta_n_lambda)
    return theta_n_lambda, d_theta, g_n_lambda, beta_n


def is_valid_observation(y: np.ndarray, phi: np.ndarray, theta: np.ndarray, gramian: np.ndarray,
                         beta: float, sigma: float = 0.1) -> bool:
    """
    根据 theta 上的置信椭球，检查新的观测值 (phi, y) 是否有效。
    :param y: 观测值
    :param phi: 特征
    :param theta: 估计的参数
    :param gramian: Gram 矩阵
    :param beta: 椭球半径
    :param sigma: 噪声协方差
    :return: 观测值的有效性 """
    y_hat = np.tensordot(theta, phi, axes=[0, 0])
    error = np.linalg.norm(y - y_hat)
    eig_phi, _ = np.linalg.eig(phi.transpose() @ phi)
    eig_g, _ = np.linalg.eig(gramian)
    error_bound = np.sqrt(np.amax(eig_phi) / np.amin(eig_g)) * beta + sigma
    return error < error_bound


def is_consistent_dataset(data: dict, parameter_box: np.ndarray = None) -> bool:
    """
    检查数据集 {phi_n, y_n} 是否一致。 最后一个观测值应该在通过前 N-1 个观测值得出的置信椭球体内。
    :param data: 包含特征 phi 和输出 y 的字典 {"features": [phi_0,...,phi_N], "outputs": [y_0,...,y_N]}
    :param parameter_box: 包含参数 theta 的区间 [theta_min, theta_max]
    :return: 数据集的一致性
    """
    train_set = copy.deepcopy(data)
    y, phi = train_set["outputs"].pop(-1), train_set["features"].pop(-1)
    y, phi = np.array(y)[..., np.newaxis], np.array(phi)[..., np.newaxis]
    if train_set["outputs"] and train_set["features"]:
        theta, _, gramian, beta = confidence_polytope(train_set, parameter_box=parameter_box)
        return is_valid_observation(y, phi, theta, gramian, beta)
    else:
        return True


def near_split(x, num_bins=None, size_bins=None):
    """ 将一个数字分成若干个大小近似均匀的容器。 您可以设置容器的数量或者它们的大小。 容器的总和始终等于该数字的总和。
    :param x: 需要分割的数字
    :param num_bins: 容器的数量
    :param size_bins: 容器的大小
    :return: 容器大小的列表
    """
    if num_bins:
        quotient, remainder = divmod(x, num_bins)
        return [quotient + 1] * remainder + [quotient] * (num_bins - remainder)
    elif size_bins:
        return near_split(x, num_bins=int(np.ceil(x / size_bins)))


def distance_to_circle(center, radius, direction):
    scaling = radius * np.ones((2, 1))
    a = np.linalg.norm(direction / scaling) ** 2
    b = -2 * np.dot(np.transpose(center), direction / np.square(scaling))
    c = np.linalg.norm(center / scaling) ** 2 - 1
    root_inf, root_sup = solve_trinom(a, b, c)
    if root_inf and root_inf > 0:
        distance = root_inf
    elif root_sup and root_sup > 0:
        distance = 0
    else:
        distance = np.infty
    return distance


def distance_to_rect(line: Tuple[np.ndarray, np.ndarray], rect: List[np.ndarray]):
    """
    计算线段与矩形的交点。 参考 https://math.stackexchange.com/a/2788041。
    :param line: 一个线段 [R, Q]
    :param rect: 一个矩形 [A, B, C, D]
    :return: R 点与线段 RQ 与矩形 ABCD 的交点之间的距离 """
    r, q = line
    a, b, c, d = rect
    u = b - a
    v = d - a
    u, v = u/np.linalg.norm(u), v/np.linalg.norm(v)
    rqu = (q - r) @ u
    rqv = (q - r) @ v
    interval_1 = [(a - r) @ u / rqu, (b - r) @ u / rqu]
    interval_2 = [(a - r) @ v / rqv, (d - r) @ v / rqv]
    interval_1 = interval_1 if rqu >= 0 else list(reversed(interval_1))
    interval_2 = interval_2 if rqv >= 0 else list(reversed(interval_2))
    if interval_distance(*interval_1, *interval_2) <= 0 \
            and interval_distance(0, 1, *interval_1) <= 0 \
            and interval_distance(0, 1, *interval_2) <= 0:
        return max(interval_1[0], interval_2[0]) * np.linalg.norm(q - r)
    else:
        return np.inf


def solve_trinom(a, b, c):
    delta = b ** 2 - 4 * a * c
    if delta >= 0:
        return (-b - np.sqrt(delta)) / (2 * a), (-b + np.sqrt(delta)) / (2 * a)
    else:
        return None, None


def img_2_video(path: str, imgs: list, use_opencv=False):
    if imgs is None:
        return

    h, w, c = imgs[0].shape
    if use_opencv:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 用于mp4格式的生成
        videowriter = cv2.VideoWriter(path, fourcc, 20, (w, h))  # 创建一个写入视频对象
        for i in range(len(imgs)):
            img_bgr = cv2.cvtColor(imgs[i], cv2.COLOR_RGB2BGR)
            videowriter.write(img_bgr)
        videowriter.release()

    else:
        plt.figsize = (w // 100, h // 100)
        fig, ax = plt.subplots()
        plt.axis('off')
        metadata = dict(title='01', artist='Highway', comment='depth prediiton')
        writer = FFMpegWriter(fps=20, metadata=metadata)
        with writer.saving(fig, path, 100):
            for image in imgs:
                plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                plt.margins(0, 0)
                # plt.ioff()
                plt.imshow(image)
                writer.grab_frame()