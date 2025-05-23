import itertools

import numpy as np


def intervals_product(a, b):
    """
        计算两个区间的乘积
        :param a: 区间 [a_min, a_max]
        :param b: 区间 [b_min, b_max]
        :return: 乘积ab的区间
    """
    p = lambda x: np.maximum(x, 0)
    n = lambda x: np.maximum(-x, 0)
    return np.array(
        [np.dot(p(a[0]), p(b[0])) - np.dot(p(a[1]), n(b[0])) - np.dot(n(a[0]), p(b[1])) + np.dot(n(a[1]), n(b[1])),
         np.dot(p(a[1]), p(b[1])) - np.dot(p(a[0]), n(b[1])) - np.dot(n(a[1]), p(b[0])) + np.dot(n(a[0]), n(b[0]))])


def intervals_diff(a, b):
    """
        计算两个区间的差值
        :param a: 区间 [a_min, a_max]
        :param b: 区间 [b_min, b_max]
        :return: 差值 a - b 的区间

    """
    return np.array([a[0] - b[1], a[1] - b[0]])


def interval_negative_part(a):
    """
        计算一个区间的负部分
        :param a: 区间 [a_min, a_max]
        :return: 其负部分 min(a, 0) 的区间
    """
    return np.minimum(a, 0)


def integrator_interval(x, k):
    """
        计算一个积分器系统的区间：dx = -k*x
        :param x: 状态区间
        :param k: 增益区间，必须为正值
        :return: dx 的区间
    """

    if x[0] >= 0:
        interval_gain = np.flip(-k, 0)
    elif x[1] <= 0:
        interval_gain = -k
    else:
        interval_gain = -np.array([k[0], k[0]])
    return interval_gain*x  # Note: no flip of x, contrary to using intervals_product(k,interval_minus(x))


def vector_interval_section(v_i, direction):
    corners = [[v_i[0, 0], v_i[0, 1]],
               [v_i[0, 0], v_i[1, 1]],
               [v_i[1, 0], v_i[0, 1]],
               [v_i[1, 0], v_i[1, 1]]]
    corners_dist = [np.dot(corner, direction) for corner in corners]
    return np.array([min(corners_dist), max(corners_dist)])


def polytope(parametrized_f, params_intervals):
    """

    :param parametrized_f: 参数化的矩阵函数
    :param params_intervals: 轴：[最小值，最大值]，参数
    :return: a0，d_a 多面体，表示矩阵区间
    """
    params_means = params_intervals.mean(axis=0)
    a0 = parametrized_f(params_means)
    vertices_id = itertools.product([0, 1], repeat=params_intervals.shape[1])
    d_a = []
    for vertex_id in vertices_id:
        params_vertex = params_intervals[vertex_id, np.arange(len(vertex_id))]
        d_a.append(parametrized_f(params_vertex) - parametrized_f(params_means))
    d_a = list({d_a_i.tostring(): d_a_i for d_a_i in d_a}.values())
    return a0, d_a


def is_metzler(matrix):
    return (matrix - np.diagonal(matrix) >= 0).all()


class LPV(object):
    def __init__(self, x0, a0, da, d=None, center=None, x_i=None):
        self.x0 = np.array(x0, dtype=float)
        self.a0 = np.array(a0, dtype=float)
        self.da = [np.array(da_i) for da_i in da]
        self.d = np.array(d) if d is not None else np.zeros(self.x0.shape)
        self.center = np.array(center) if center is not None else np.zeros(self.x0.shape)
        self.coordinates = None

        self.x_i = np.array(x_i) if x_i is not None else np.array([self.x0, self.x0])
        self.x_i_t = None

        self.update_coordinates_frame(self.a0)

    def update_coordinates_frame(self, a0):
        """
            确保动力学矩阵A0是Metzler型的。

            如果不是，设计一个坐标转换并将其应用于模型和状态区间。
            :param a0: 动力学矩阵A0

        """
        self.coordinates = None
        # Rotation
        if not is_metzler(a0):
            eig_v, transformation = np.linalg.eig(a0)
            if np.isreal(eig_v).all():
                self.coordinates = (transformation, np.linalg.inv(transformation))
            else:
                print("Non Metzler A0 with complex eigenvalues: ", eig_v)
        else:
            self.coordinates = (np.eye(a0.shape[0]), np.eye(a0.shape[0]))

        # Forward coordinates change of states and models
        self.a0 = self.change_coordinates(self.a0, matrix=True)
        self.da = self.change_coordinates(self.da, matrix=True)
        self.d = self.change_coordinates(self.d, offset=False)
        self.x_i_t = self.change_coordinates(self.x_i)

    def change_coordinates(self, value, matrix=False, back=False, interval=False, offset=True):
        """
            执行坐标变换：旋转和居中。

            :param value: 要转换的对象
            :param matrix: 是矩阵还是向量？
            :param back: 如果为True，返回原始坐标
            :param interval: 当转换间隔时，必须使用有损的间隔算术来保持包含性质。
            :param offset: 是否应用居中
            :return: 转换后的对象
        """
        if self.coordinates is None:
            return value
        transformation, transformation_inv = self.coordinates
        if interval:
            value = intervals_product(
                [self.coordinates[0], self.coordinates[0]],
                value[:, :, np.newaxis]).squeeze() + offset * np.array([self.center, self.center])
            return value
        elif matrix:  # Matrix
            if back:
                return transformation @ value @ transformation_inv
            else:
                return transformation_inv @ value @ transformation
        elif isinstance(value, list):  # List
            return [self.change_coordinates(v, back) for v in value]
        elif len(value.shape) == 2:
                for t in range(value.shape[0]):  # Array of vectors
                    value[t, :] = self.change_coordinates(value[t, :], back=back)
                return value
        elif len(value.shape) == 1:  # Vector
            if back:
                return transformation @ value + offset * self.center
            else:
                return transformation_inv @ (value - offset * self.center)

    def step(self, dt):
        self.x_i_t = self.step_interval_predictor(self.x_i_t, dt)

    def step_interval_observer(self, x_i, dt):
        a0, da, d = self.a0, self.da, self.d
        a_i = a0 + sum(intervals_product([0, 1], [da_i, da_i]) for da_i in da)
        dx_i = intervals_product(a_i, x_i) + d
        return x_i + dx_i*dt

    def step_interval_predictor(self, x_i, dt):
        a0, da, d = self.a0, self.da, self.d
        p = lambda x: np.maximum(x, 0)
        n = lambda x: np.maximum(-x, 0)
        da_p = sum(p(da_i) for da_i in da)
        da_n = sum(n(da_i) for da_i in da)
        x_m, x_M = x_i[0, :, np.newaxis], x_i[1, :, np.newaxis]
        dx_m = a0 @ x_m - da_p @ n(x_m) - da_n @ p(x_M) + d[:, np.newaxis]
        dx_M = a0 @ x_M + da_p @ p(x_M) + da_n @ n(x_m) + d[:, np.newaxis]
        dx_i = np.array([dx_m.squeeze(axis=-1), dx_M.squeeze(axis=-1)])
        return x_i + dx_i * dt
