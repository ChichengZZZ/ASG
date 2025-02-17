import torch
import math
from pykalman import KalmanFilter
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import euclidean_distances
import hdbscan
from hdbscan import HDBSCAN
import matplotlib.pyplot as plt
import scipy
import scipy.special


def normal_entropy(std):
    """
    计算正态分布的熵
    参数：
    std: 标准差
    返回值：
    熵
    """

    var = std.pow(2)  # 计算方差
    entropy = 0.5 + 0.5 * torch.log(2 * var * math.pi)  # 计算熵
    return entropy.sum(1, keepdim=True)


def normal_log_density(x, mean, log_std, std):
    """
    计算给定数据在正态分布下的对数密度
    参数：
    x: 给定数据
    mean: 均值
    log_std: 标准差的对数
    std: 标准差
    返回值：
    对数密度
    """

    var = std.pow(2)  # 计算方差
    log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - log_std  # 计算对数密度
    return log_density.sum(1, keepdim=True)


def Kalman1D(observations, damping=0.1, tc=0.3):
    """
    使用一维Kalman滤波器对观测数据进行平滑处理
    参数：
    observations: 观测序列
    damping: 观测协方差
    tc: 过渡协方差
    返回值：
    平滑后的时间序列数据
    """
    # 设置Kalman滤波器的参数
    observation_covariance = damping
    initial_value_guess = observations[0]
    transition_matrix = 1
    transition_covariance = tc
    initial_value_guess

    # 创建Kalman滤波器对象
    kf = KalmanFilter(
            initial_state_mean=initial_value_guess,
            initial_state_covariance=observation_covariance,
            observation_covariance=observation_covariance,
            transition_covariance=transition_covariance,
            transition_matrices=transition_matrix
        )

    # 对观测数据进行平滑处理
    pred_state, state_cov = kf.smooth(observations)
    return pred_state


def calculate_angle(vector_a, vector_b):
    """
    计算两个向量之间的夹角
    参数：
    vector_a: 向量A
    vector_b: 向量B
    返回值：
    夹角的度数
    """
    try:
        dot_product = sum(a * b for a, b in zip(vector_a, vector_b))  # 计算向量点乘
        mod_a = math.sqrt(sum(a ** 2 for a in vector_a))  # 计算向量A的模
        mod_b = math.sqrt(sum(b ** 2 for b in vector_b))  # 计算向量B的模

        angle_rad = math.acos(np.clip(dot_product / (mod_a * mod_b + 1e-10), -1, 1))  # 计算夹角的弧度
        angle_deg = angle_rad * (180 / math.pi)  # 转换为度数

    except Exception as e:
        # 处理异常情况
        print('vector_a', vector_a)
        print('vector_b', vector_b)
        print(e)
        angle_deg = 181

    return angle_deg

def orientation(p, q, r):
    # 计算3个点的方向
    # 返回值 > 0 表示顺时针方向
    # 返回值 < 0 表示逆时针方向
    # 返回值 = 0 表示共线
    return (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])


def line_segments_intersect(p1, q1, p2, q2):

    x1, y1 = tuple(p1)
    x2, y2 = tuple(q1)
    x3, y3 = tuple(p2)
    x4, y4 = tuple(q2)
    # 计算两条线段的方向向量
    vector_ab = (x2 - x1, y2 - y1)
    vector_cd = (x4 - x3, y4 - y3)

    # 判断两条线段是否平行
    if vector_ab[0]*vector_cd[1] - vector_ab[1]*vector_cd[0] == 0:
        # 判断是否完全重叠
        if (x1, y1) == (x3, y3) and (x2, y2) == (x4, y4):
            return True
        # 判断是否部分重叠
        if (x1, y1) == (x3, y3) or (x1, y1) == (x4, y4) or (x2, y2) == (x3, y3) or (x2, y2) == (x4, y4):
            return True
        # 两条线段平行且不重叠，不相交
        return False

    # 使用线段相交公式计算t1和t2
    t1 = ((x3 - x1) * (y3 - y4) - (y3 - y1) * (x3 - x4)) / ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    t2 = ((x1 - x2) * (y3 - y1) - (y1 - y2) * (x3 - x1)) / ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))

    # 判断两条线段是否相交
    if 0 <= t1 <= 1 and 0 <= t2 <= 1:
        return True

    return False

def on_segment(p, q, r):
    # 判断点 r 是否在线段 pq 上
    return min(p[0], q[0]) <= r[0] <= max(p[0], q[0]) and min(p[1], q[1]) <= r[1] <= max(p[1], q[1])


def do_segments_intersect(p1, q1, p2, q2):
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # Case 1: 一般情况
    if o1 != o2 and o3 != o4:
        return True

    # Case 2: 特殊情况
    # p1、q1和p2共线，并且p2在p1-q1上
    if o1 == 0 and on_segment(p1, q1, p2):
        return True

    # p1、q1和q2共线，并且q2在p1-q1上
    if o2 == 0 and on_segment(p1, q1, q2):
        return True

    # p2、q2和p1共线，并且p1在p2-q2上
    if o3 == 0 and on_segment(p2, q2, p1):
        return True

    # p2、q2和q1共线，并且q1在p2-q2上
    if o4 == 0 and on_segment(p2, q2, q1):
        return True

    # 如果以上情况都不满足，则两条线段不相交
    return False


# def filter_similar_points(points, min_cluster_size, min_samples):
#     # 使用HDBSCAN算法进行聚类和过滤
#     clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
#     labels = clusterer.fit_predict(points)
#
#     # 获取过滤后的轨迹点
#     filtered_points = []
#     for i in range(max(labels) + 1):
#         cluster_points = [point for j, point in enumerate(points) if labels[j] == i]
#         filtered_points.extend(cluster_points)
#     if len(filtered_points)>0:
#         return np.array(filtered_points)
#
#     return points

def filter_similar_points(points, threshold):
    n_cluster = 10 if len(points) > 10 else len(points)
    # 使用K-means算法将轨迹点分为不同的簇
    kmeans = KMeans(n_clusters=n_cluster)   # 设置簇的数量，根据需要进行调整
    kmeans.fit(points)

    # 获取每个点所属的簇
    labels = kmeans.labels_

    # 计算每个簇中所有点之间的距离
    distances = pairwise_distances(points)

    # 过滤相近的轨迹点
    filtered_points = []
    for i in range(max(labels)+1):
        # 获取属于当前簇的点的索引
        indices = [j for j, label in enumerate(labels) if label == i]

        # 仅保留距离大于阈值的点
        filtered_indices = []
        for j in indices:
            if all(distances[j, k] > threshold for k in indices):
                filtered_indices.append(j)

        # 将保留的点添加到过滤后的轨迹点中
        filtered_points.extend(points[filtered_indices])
    if len(filtered_points)>0:
        return np.array(filtered_points)

    return points


def polynomial_func(x, coeffs):
    """
    计算多项式函数值
    参数：
    x: 自变量 x
    coeffs: 多项式的系数
    返回值：
    计算得到的函数值
    """
    return np.dot(np.vander(x, len(coeffs)), coeffs)  # 使用 numpy 中的 vander 函数生成多项式 Vandermonde 矩阵，并进行系数相乘得到函数值


def calculate_bezier_point(t, control_points):
    """
    计算贝塞尔曲线上的点
    参数：
    t: 参数 t，表示曲线上的位置
    control_points: 控制点的坐标列表
    返回值：
    计算得到的贝塞尔曲线上的点的坐标
    """
    n = len(control_points) - 1  # 控制点数量减一
    point = np.zeros(2)  # 初始化点的坐标为零向量
    for i in range(n + 1):  # 遍历每个控制点
        binomial_co = scipy.special.comb(n, i)  # 计算二项式系数
        point += binomial_co * ((1 - t) ** (n - i)) * (t ** i) * control_points[i]  # 计算对应t下的点坐标并累加
    return point


def generate_bezier_curve(control_points):
    """
    生成贝塞尔曲线
    参数：
    control_points: 控制点的坐标列表
    返回值：
    计算得到的贝塞尔曲线上的点的坐标列表
    """
    t = np.linspace(0, 1, 100)  # 在[0, 1]区间内均匀取100个点
    curve = np.array([calculate_bezier_point(ti, control_points) for ti in t])  # 计算每个t对应的贝塞尔曲线上的点的坐标
    return curve

def init_linear(ngsim_traj, PolyLaneFixedWidth):
    _, idx = np.unique(ngsim_traj[:, :2].copy(), axis=0, return_index=True)
    unique_arr = ngsim_traj[np.sort(idx)]

    if unique_arr.shape[0] > 2 and (np.linalg.norm(unique_arr[-1] - unique_arr[0]) > 0.5):
        # self.unique_arr = filter_similar_points(self.unique_arr, min_cluster_size=2, min_samples=10)
        # unique_arr = filter_similar_points(unique_arr, threshold=0.5)
        # 原始数据点
        # fig, ax = plt.subplots()
        # x = sorted(unique_arr[:, 0], reverse=unique_arr[0, 0] > unique_arr[-1, 0])
        # y = sorted(unique_arr[:, 1], reverse=unique_arr[0, 1] > unique_arr[-1, 1])
        # ax.plot(x, y, color='r', alpha=0.5)
        # ax.plot(unique_arr[:, 0], unique_arr[:, 1], color='b', alpha=0.5)
        # plt.savefig('./spline.png')
        # unique_arr = np.stack([x, y], axis=1).tolist()

        # degree = 7
        x = unique_arr[:, 0]
        y = unique_arr[:, 1]
        #
        # # 构造多项式系数矩阵
        # X = np.vander(x, degree + 1, increasing=True)
        #
        # # 使用最小二乘法拟合多项式曲线
        # coefficients = np.linalg.lstsq(X, y, rcond=None)[0]
        #
        # # 生成拟合曲线的 x 坐标
        # x_fit = np.linspace(min(x), max(x), 100)
        #
        # # 计算拟合曲线的 y 坐标
        # y_fit = polynomial_func(x_fit, coefficients)
        #
        # # 绘制原始轨迹点和拟合曲线
        # plt.scatter(x, y, label='Data Points')
        # plt.plot(x_fit, y_fit, 'r-', label='Fitted Curve')
        # plt.xlabel('x')
        # plt.ylabel('y')
        # plt.legend()
        # plt.show()

        # 控制点
        control_points = np.column_stack((x, y))

        # 生成贝塞尔曲线
        bezier_curve = generate_bezier_curve(control_points)
        #
        # 绘制轨迹点和贝塞尔曲线
        # fig, ax = plt.subplots()
        # ax.scatter(x, y, label='Data Points')
        # ax.plot(bezier_curve[:, 0], bezier_curve[:, 1], 'r-', label='Bezier Curve')
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # plt.legend()
        # plt.savefig('./spline.png')
        # plt.show()

        try:
            linear = PolyLaneFixedWidth(bezier_curve)
        except Exception as e:
            linear = None
    else:
        linear = None

    return linear, unique_arr



# # 轨迹点的坐标
# x = [1, 2, 4, 6, 8]
# y = [1, 3, 5, 7, 9]

