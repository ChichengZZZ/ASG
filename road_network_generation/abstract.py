from abc import ABC
from abc import abstractmethod, abstractproperty, abstractstaticmethod, abstractclassmethod
import copy
import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os.path as osp
import os
import gc
from sklearn.cluster import KMeans, k_means
import alphashape
from sklearn.cluster import OPTICS, DBSCAN
import hdbscan


class GeneratedRoadNet(ABC):
    def __init__(self, config: dict) -> None:
        self.config = self.default_config()
        self.configure(config)
        # 存储需要处理的数据路径
        self.data_path = list()
        # 对数据路径根据路网类别进行分类, key为类别id，value：list
        self.class_path = dict()
        # key:document path, value:类别id.
        self.class_path_reverse = dict()
        # key 是 路网类别id , value: kmeans
        self.direction = dict()

    @classmethod
    def default_config(cls) -> dict:
        return {

            "vehicle_id_name": "",
            "coordinate_name": [],
            "sample_point_num": 10,
            "save_road_kwargs": dict(),
            # 转向索引
            "through_idx": 1,
            "left_turn_idx": 2,
            "right_turn_idx": 3,
            # 数据集存储路径
            "dataset_path": None,
            # 智能体在数据中的列名
            "agent_type_name": "agent_type",
            # 智能体的类别名
            "agent_type": ["car"],
            # 数据集中路网类别总数
            "road_net_categories_num": -1,
            # 每类路网的路口总数
            "crossing_center_num": [],
            # 数据清洗：轨迹起始点、终止点与路口中心的最大距离
            "max_distance_with_center": 10.0,
            # 数据清洗：静止车辆最大移动距离
            "max_distance_of_stationary_vehicle": 0.5,
            # 数据清洗：最小轨迹点数目
            "min_trajectory_point_num": 20,
            # 生成的路网使用的最大文件数
            "max_file_num": 5,
            # 项目输出路径
            "output_path": None,
            # visualization
            # 用于绘图
            "masker": ['.', ',', 'o', 'v', '^', '<', '>', '8',
                       's', 'p', '*', '+', 'D', 'd', '_', '|', 'x'],
            # show crossing center
            "show_img": False,
            "save_img": True

        }

    @classmethod
    def separate_data_by_key_value(cls, data, key, value):
        try:
            data_copy = data[data[key] == value].copy()
        except Exception as e:
            raise e
        else:
            return data_copy

    def configure(self, config: dict) -> None:
        if config:
            self.config.update(config)

    @abstractmethod
    def parse_dataset_path(self) -> None:
        """将需要处理的数据的路径保存在self.data_path"""
        raise NotImplementedError

    @abstractmethod
    def classify_path(self) -> None:
        raise NotImplementedError

    def delete_the_stationary_traj(self, df_to_dict: dict) -> dict:
        """删除静止的车辆"""

        df_to_dict_after = dict()

        for k, df in df_to_dict.items():
            trajectory = df[self.config["coordinate_name"]].to_numpy()
            # 如果轨迹点数量小于等于最小轨迹点数，则跳过
            if trajectory.shape[0] <= self.config["min_trajectory_point_num"]:
                continue
            # 如果轨迹中任一维度的最大值与最小值之差大于静止车辆的最大距离，保留该车辆的轨迹
            if ((np.max(trajectory[:, 0]) - np.min(trajectory[:, 0])) >
                self.config['max_distance_of_stationary_vehicle']) or \
                    ((np.max(trajectory[:, 1]) - np.min(trajectory[:, 1])) >
                     self.config['max_distance_of_stationary_vehicle']):
                df_to_dict_after[k] = df.copy()

        return df_to_dict_after

    @classmethod
    def compute_dis(cls, point1: np.ndarray, point2: np.ndarray) -> np.ndarray:
        return np.linalg.norm(point2 - point1)

    def cluster_junction_centers(self):
        """聚类出所有路口中心"""
        for k, v in self.class_path.items():
            data = []  # 路口中心点的集合
            total_point = []  # 所有轨迹点的集合
            for i, p in enumerate(v):
                df = self.read_data(p)  # 读取数据
                data_dict = self.make_dataframe_become_dict(df)  # 将数据转换为字典形式
                data_dict = self.delete_the_stationary_traj(data_dict)  # 删除静止车辆的轨迹
                for vh_id, traj in data_dict.items():
                    traj_copy = traj.copy().reset_index(drop=True)
                    if traj.shape[0] < 2:
                        continue
                    total_point.extend(traj_copy[self.config['coordinate_name']].to_numpy().tolist())  # 将轨迹点加入总集合
                    first_point = np.array(traj_copy.loc[0, self.config['coordinate_name']])
                    last_point = np.array(traj_copy.loc[traj_copy.shape[0] - 1, self.config['coordinate_name']])
                    data.extend([first_point, last_point])  # 将轨迹的起点和终点加入路口中心点集合

            else:
                # 使用KMeans聚类算法找到路口中心点
                kmeans = KMeans(n_clusters=self.config['crossing_center_num'][k], random_state=0).fit(np.array(data))
                self.direction[k] = kmeans

                # 绘制散点图显示聚类结果
                fig, ax = plt.subplots()
                ax.scatter(*zip(*total_point), c='blue', alpha=0.1)  # 绘制所有轨迹点
                for i in range(kmeans.cluster_centers_.shape[0]):
                    ax.scatter(kmeans.cluster_centers_[i, 0], kmeans.cluster_centers_[i, 1],
                               marker=self.config["masker"][i], label=' label-{}'.format(i))  # 绘制路口中心点
                plt.legend()
                plt.title("crossing center clustering")
                plt.ylabel("Y(m)")
                plt.xlabel("X(m)")
                if self.config['save_img']:
                    plt.savefig(os.path.join(self.config["output_path"], f"crossing_center_{k}.png"))  # 保存图片
                if self.config['show_img']:
                    plt.show()  # 显示图片

    # 根据轨迹的起始位置与终止位置，对轨迹进行分类，每个轨迹集合对应一条道路。
    def all_road(self, idx: int) -> dict:
        road = dict()
        for i, p in enumerate(self.class_path[idx]):

            if i >= self.config['max_file_num']:
                break

            df = self.read_data(p)
            data_dict = self.make_dataframe_become_dict(df)
            data_dict = self.delete_the_stationary_traj(data_dict)
            for vh_id, traj in data_dict.items():
                traj_copy = traj.copy().reset_index(drop=True)
                if traj.shape[0] < 2:
                    continue
                first_point = traj_copy.loc[0, self.config['coordinate_name']].to_numpy()
                last_point = traj_copy.loc[traj_copy.shape[0] - 1, self.config['coordinate_name']].to_numpy()
                labels = tuple(self.direction[idx].predict(np.array([first_point, last_point])).tolist())
                point1 = self.direction[idx].cluster_centers_[labels[0], :]
                point2 = self.direction[idx].cluster_centers_[labels[1], :]
                dis1 = self.compute_dis(first_point, point1)
                dis2 = self.compute_dis(last_point, point2)

                # 剔除掉头的轨迹
                if labels[0] == labels[1]:
                    continue

                if float(dis2) > self.config["max_distance_with_center"] or \
                        float(dis1) > self.config["max_distance_with_center"]:
                    continue

                traj_coordinate = traj_copy[self.config['coordinate_name']].to_numpy()
                if road.get(str(labels)) is None:
                    # 如果不存在某种组合，则进行创建新的键值对
                    road[str(labels)] = [traj_coordinate]
                else:
                    traj_coordinate_list = road[str(labels)]
                    traj_coordinate_list.append(traj_coordinate)
                    # 对道路的轨迹集合进行更新
                    road[str(labels)] = traj_coordinate_list

        return road

    def generated_road_network(self, idx: int, road: dict) -> dict:
        lines = dict()  # 存储道路网络的边界线

        for k, v in road.items():
            data = []  # 所有轨迹点的集合
            start_point = set()  # 起点的集合
            end_point = set()  # 终点的集合

            # 遍历每条道路的轨迹数据
            for trej in v:
                data.extend(trej.tolist())  # 将轨迹点加入总集合
                start_point.add(tuple(trej[0, :].tolist()))  # 将起点加入起点集合
                end_point.add(tuple(trej[-1, :].tolist()))  # 将终点加入终点集合

            data = np.array(data)

            # 使用DBSCAN聚类算法对轨迹点进行聚类，生成聚类标签
            clustering = DBSCAN(eps=0.5, min_samples=(data.shape[0] // 4)).fit(data)
            cluster_labels = clustering.labels_

            # 随机生成颜色
            codebook = np.random.randint(255, size=(500, 3)) / 255
            color = [codebook[i] for i in cluster_labels]

            # 绘制散点图显示聚类结果和路口中心点
            fig, ax = plt.subplots()
            ax.scatter(data[:, 0], data[:, 1], c=color)  # 绘制聚类结果点
            for i in range(self.direction[idx].cluster_centers_.shape[0]):
                ax.scatter(self.direction[idx].cluster_centers_[i, 0], self.direction[idx].cluster_centers_[i, 1],
                           marker=self.config["masker"][i], label=' center-{}'.format(i))  # 绘制路口中心点
            plt.legend()
            plt.title("Road Network Clustering")
            plt.ylabel("Y(m)")
            plt.xlabel("X(m)")
            if self.config['save_img']:
                plt.savefig(os.path.join(self.config["output_path"], f"{idx}_road_network_clustering_{k}.png"))
                # 保存图片
            if self.config['show_img']:
                plt.show()  # 显示图片

            # 对聚类出的道路提取边界线
            line = []
            for c in range(np.max(cluster_labels) + 2):
                cond_vector = np.where(cluster_labels == c - 1)[0]
                conda_data = data[cond_vector].copy()
                line.append(self.acquire_boundary(conda_data, (start_point, end_point), k, idx))
            lines[k] = line

        return lines

    def acquire_boundary(self, data: np.ndarray, start_end_point: tuple, key: tuple, idx: int) -> list:
        start_point, end_point = start_end_point

        # 使用alpha形状方法计算凸包
        hull = alphashape.alphashape(data, 0.5)
        hull_pts = hull.exterior.coords.xy
        x = hull_pts[0]
        y = hull_pts[1]

        # 初始化左右边界线
        left_line = []
        right_line = []

        # 将凸包的点绘制在蓝色散点图上
        edgs = [(x[i], y[i]) for i in range(len(x))]
        fig, ax = plt.subplots()
        ax.scatter(*zip(*edgs), c='blue')

        # 将中心点绘制在图上
        for i in range(self.direction[idx].cluster_centers_.shape[0]):
            ax.scatter(self.direction[idx].cluster_centers_[i, 0], self.direction[idx].cluster_centers_[i, 1],
                       marker=self.config["masker"][i], label=' center-{}'.format(i))

        plt.legend()
        plt.title("Road Network Boundary")
        plt.ylabel("Y(m)")
        plt.xlabel("X(m)")

        # 如果设置保存图像，则保存散点图
        if self.config['save_img']:
            plt.savefig(os.path.join(self.config["output_path"], f"{idx}_road_boundary_{key}.png"))

        # 如果设置展示图像，则展示散点图
        if self.config['show_img']:
            plt.show()

        print(f"key:{key}, :")

        # 初始化左右边界线的起点和终点
        right_line_start = -1
        right_line_end = -1
        left_line_start = -1
        left_line_end = -1

        # 遍历边界散点，找到起点和终点
        for j in range(len(edgs)):
            if (edgs[j] in start_point) and (edgs[(j + 1) % len(edgs)] in start_point) and (
                    edgs[(j - 1) % len(edgs)] not in start_point):
                print(f"right line start {j}.")
                right_line_start = j
            elif (edgs[(j - 1) % len(edgs)] in start_point) and (edgs[j] in start_point) and (
                    edgs[(j + 1) % len(edgs)] not in start_point):
                left_line_start = j
                print(f"left line start {j}.")

            if (edgs[j] in end_point) and (edgs[(j + 1) % len(edgs)] in end_point) and (
                    edgs[(j - 1) % len(edgs)] not in end_point):
                print(f"left line end {j}.")
                left_line_end = j
            elif (edgs[(j - 1) % len(edgs)] in end_point) and (edgs[j] in end_point) and (
                    edgs[(j + 1) % len(edgs)] not in end_point):
                right_line_end = j
                print(f"right line end {j}.")

        point_idx = left_line_start
        # 获取道路左边边界线
        while True:
            if point_idx == left_line_end:
                left_line.append(edgs[point_idx])
                break

            if left_line_start > right_line_start:
                point_idx += 1
            else:
                point_idx -= 1

            point_idx %= len(edgs)
            left_line.append(edgs[point_idx])

        point_idx = right_line_start
        # 获取道路右边边界线
        while True:
            if point_idx == right_line_end:
                right_line.append(edgs[point_idx])
                break

            if left_line_start < right_line_start:
                point_idx += 1
            else:
                point_idx -= 1

            point_idx %= len(edgs)
            right_line.append(edgs[point_idx])

        left_line_ = []
        right_line_ = []

        # 采样左右边界线的点
        for i in range(self.config["sample_point_num"]):
            left_line_.append(left_line[i * (len(left_line) // self.config["sample_point_num"])])
            right_line_.append(right_line[i * (len(right_line) // self.config["sample_point_num"])])
        else:
            left_line_.append(left_line[-self.config["sample_point_num"] // 2])
            left_line_.append(left_line[-1])
            right_line_.append(right_line[-self.config["sample_point_num"] // 2])
            right_line_.append(right_line[-1])

        fig, ax = plt.subplots()
        ax.scatter(*zip(*left_line_), c='blue')
        ax.scatter(*zip(*right_line_), c='red')

        # 将中心点绘制在图上
        for i in range(self.direction[idx].cluster_centers_.shape[0]):
            ax.scatter(self.direction[idx].cluster_centers_[i, 0], self.direction[idx].cluster_centers_[i, 1],
                       marker=self.config["masker"][i], label=' center-{}'.format(i))

        plt.legend()
        plt.title("Boundary Line")
        plt.ylabel("Y(m)")
        plt.xlabel("X(m)")

        # 如果设置保存图像，则保存边界线图像
        if self.config['save_img']:
            plt.savefig(os.path.join(self.config["output_path"], f"{idx}_boundary_line_{key}.png"))

        # 如果设置展示图像，则展示边界线图像
        if self.config['show_img']:
            plt.show()

        return [left_line_, right_line_]

    def get_center_line_of_road(self, roads: dict, idx: int):
        """
        这段代码的作用是根据给定的道路数据，获取道路的中心线。
        参数说明：
        - roads：道路数据，是一个字典结构，在字典中每个键值对表示一条道路，键是道路的名称，值是一个列表，表示道路的线段组成，每个线段由左右两条边界线组成。
        - idx：道路的索引，用于命名保存的图片文件。

        代码逻辑如下：
            设置一个空的列表Line来存储中心线的点。
            遍历道路字典，每次取出一条道路：
            - 创建一个空的列表Line来存储中心线的点。
            - 遍历该道路的每个线段，每次取出一条线段：
            获取线段的左边界线和右边界线。
            遍历左边界线或右边界线中的每个点，每次取出一个点：
            将左边界线和右边界线对应的点的坐标相加并除以2，得到当前点的中心点，存入Line列表中。
            将得到的中心线点列表转换为numpy数组fit_data。
            绘制图形：
            创建一个图形对象fig和一个坐标轴对象ax。
            将左边界线和右边界线分别绘制出来。
            将中心线点列表fit_data绘制出来。
            设置x轴和y轴的标签。
            添加图例。
            如果配置中设置了保存图片，则将图片保存到指定路径。
            如果配置中设置了显示图片，则显示图形。

        在代码中有一段被注释掉的代码，是使用多项式拟合数据并绘制拟合曲线的部分。如果需要使用多项式拟合，可以取消相应代码的注释，并调整拟合的阶数。

        """
        for k, road in roads.items():
            Line = []
            for line in road:
                left_line = line[0]
                right_line = line[1]
                for num in range(len(left_line)):

                    Line.append((np.array(left_line[num]) + np.array(right_line[num]))/2)

                fit_data = np.array(Line)
                # x = fit_data[:, 0]
                # z1 = np.polyfit(fit_data[:, 0], fit_data[:, 1], 5)
                # p1 = np.poly1d(z1)
                # print(p1)  # 在屏幕上打印拟合多项式
                # yvals = p1(x)  # 也可以使用yvals=np.polyval(z1,x)
                fig, ax = plt.subplots()
                left_line = np.array(left_line)
                right_line = np.array(right_line)
                ax.plot(left_line[:, 0], left_line[:, 1], '<', label='left line')
                ax.plot(np.array(right_line)[:, 0], np.array(right_line)[:, 1], '>', label='right line')
                ax.plot(fit_data[:, 0], fit_data[:, 1], '*', label='center line')
                # ax.plot(x, yvals, 'r', label='polyfit values')
                plt.xlabel('xaxis')
                plt.ylabel('yaxis')
                plt.legend(loc=4)
                plt.title('polyfitting')
                if self.config['save_img']:
                    plt.savefig(os.path.join(self.config["output_path"], f"{idx}_polyfitting_{k}.png"))
                if self.config['show_img']:
                    plt.show()

    # 将dataframe格式的数据转换成字典，其中字典的key为vehicle_id和total_frame组成的元组，value为该车所有信息
    def make_dataframe_become_dict(self, df: pd.DataFrame) -> dict:

        index = self.get_vehicles_set(df)
        dict_data = dict()
        for idx in index:
            vehicle_data = self.get_the_only_vehicle(df, idx)
            dict_data[idx] = vehicle_data
        return dict_data

    def get_the_only_vehicle(self, df: pd.DataFrame, vehicle_id: int) -> pd.DataFrame:
        return df[df[self.config["vehicle_id_name"]] == vehicle_id].copy()

    def get_vehicles_set(self, df) -> set:
        """"获取dataframe格式数据中车辆唯一标识"""
        index = df[[self.config["vehicle_id_name"]]].values.tolist()
        index = set([idx[0] for idx in index])
        return index

    @abstractmethod
    def read_data(self, path: str) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def save_road(self, roads: dict, idx: int):
        raise NotImplementedError

    def run(self, idx):
        # 将需要处理的数据的路径保存在self.data_path
        self.parse_dataset_path()
        # 根据路网类型对路径分类
        self.classify_path()
        # 路口聚类
        self.cluster_junction_centers()
        # for idx in list(self.class_path.keys()):
        # 获取道路
        road = self.all_road(idx)
        # 生成路网
        lines = self.generated_road_network(idx, road)
        # 获取道路中心线
        self.get_center_line_of_road(lines, idx)
        # 保存路网
        self.save_road(lines, idx)


