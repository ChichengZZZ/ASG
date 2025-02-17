import numpy as np
from scipy import interpolate
from typing import List, Tuple


class LinearSpline2D:
    """
    对一系列点进行分段线性曲线拟合。
    """

    PARAM_CURVE_SAMPLE_DISTANCE: int = 1  # curve samples are placed 1m apart

    def __init__(self, points: List[Tuple[float, float]]):
        x_values = np.array([pt[0] for pt in points])
        y_values = np.array([pt[1] for pt in points])
        # 计算横纵坐标差分
        x_values_diff = np.diff(x_values)
        x_values_diff = np.hstack((x_values_diff, x_values_diff[-1]))
        y_values_diff = np.diff(y_values)
        y_values_diff = np.hstack((y_values_diff, y_values_diff[-1]))
        # 计算每个点与曲线起点的累计弧长
        arc_length_cumulated = np.hstack(
            (0, np.cumsum(np.sqrt(x_values_diff[:-1] ** 2 + y_values_diff[:-1] ** 2)))
        )
        self.length = arc_length_cumulated[-1]
        # 使用线性插值生成X、Y曲线
        self.x_curve = interpolate.interp1d(
            arc_length_cumulated, x_values, fill_value="extrapolate"
        )
        self.y_curve = interpolate.interp1d(
            arc_length_cumulated, y_values, fill_value="extrapolate"
        )
        # 使用线性插值生成X差分曲线，Y差分曲线
        self.dx_curve = interpolate.interp1d(
            arc_length_cumulated, x_values_diff, fill_value="extrapolate"
        )
        self.dy_curve = interpolate.interp1d(
            arc_length_cumulated, y_values_diff, fill_value="extrapolate"
        )
        # 对于曲线上每个位置，进行采样并存储位置和切线信息
        (self.s_samples, self.poses) = self.sample_curve(
            self.x_curve, self.y_curve, self.length, self.PARAM_CURVE_SAMPLE_DISTANCE
        )

    # Frenet坐标系转笛卡儿坐标系
    def __call__(self, lon: float) -> Tuple[float, float]:
        return self.x_curve(lon), self.y_curve(lon)

    # 获取给定位置的切线方向
    def get_dx_dy(self, lon: float) -> Tuple[float, float]:
        idx_pose = self._get_idx_segment_for_lon(lon)
        pose = self.poses[idx_pose]
        return pose.normal

    def cartesian_to_frenet(self, position: Tuple[float, float]) -> Tuple[float, float]:
        """
        将笛卡尔坐标系中的点转换为曲线的弗雷内坐标系。
        """

        pose = self.poses[-1]
        projection = pose.project_onto_normal(position)
        if projection >= 0:
            lon = self.s_samples[-1] + projection
            lat = pose.project_onto_orthonormal(position)
            return lon, lat

        for idx in list(range(len(self.s_samples) - 1))[::-1]:
            pose = self.poses[idx]
            projection = pose.project_onto_normal(position)
            if projection >= 0:
                if projection < pose.distance_to_origin(position):
                    lon = self.s_samples[idx] + projection
                    lat = pose.project_onto_orthonormal(position)
                    return lon, lat
                else:
                    ValueError("No valid projection could be found")
        pose = self.poses[0]
        lon = pose.project_onto_normal(position)
        lat = pose.project_onto_orthonormal(position)
        return lon, lat

    def cartesian_to_frenet_with_heading(self, position: Tuple[float, float], heading=0) -> Tuple[float, float]:
        """
        将笛卡尔坐标系中的点转换为曲线的弗雷内坐标系。
        """

        pose = self.poses[-1]
        projection = pose.project_onto_normal(position)
        if projection >= 0:
            lon = self.s_samples[-1] + projection
            lat = pose.project_onto_orthonormal(position)
            return lon, lat

        for idx in list(range(len(self.s_samples) - 1))[::-1]:
            pose = self.poses[idx]
            projection = pose.project_onto_normal(position)
            if projection >= 0:
                if projection < pose.distance_to_origin(position):
                    lon = self.s_samples[idx] + projection
                    lat = pose.project_onto_orthonormal(position)
                    return lon, lat
                else:
                    ValueError("No valid projection could be found")
        pose = self.poses[0]
        lon = pose.project_onto_normal(position)
        lat = pose.project_onto_orthonormal(position)
        return lon, lat

    def frenet_to_cartesian(self, lon: float, lat: float) -> Tuple[float, float]:
        """
        将曲线的弗雷内坐标系中的点转换为笛卡尔坐标系。
        """
        idx_segment = self._get_idx_segment_for_lon(lon)
        s = lon - self.s_samples[idx_segment]
        pose = self.poses[idx_segment]
        point = pose.position + s * pose.normal
        point += lat * pose.orthonormal
        return point

    def _get_idx_segment_for_lon(self, lon: float) -> int:
        """
        返回对应于纵向坐标的曲线姿态的索引。
        """
        idx_smaller = np.argwhere(lon < self.s_samples)
        if len(idx_smaller) == 0:
            return len(self.s_samples) - 1
        if idx_smaller[0] == 0:
            return 0
        return int(idx_smaller[0]) - 1

    @staticmethod
    def sample_curve(x_curve, y_curve, length: float, CURVE_SAMPLE_DISTANCE=1):
        """
        创建距离为CURVE_SAMPLE_DISTANCE的曲线样本。这些样本用于弗雷内到笛卡尔的转换和反向转换。
        """
        num_samples = np.floor(length / CURVE_SAMPLE_DISTANCE)
        s_values = np.hstack(
            (CURVE_SAMPLE_DISTANCE * np.arange(0, int(num_samples) + 1), length)
        )
        x_values = x_curve(s_values)
        y_values = y_curve(s_values)
        dx_values = np.diff(x_values)
        dx_values = np.hstack((dx_values, dx_values[-1]))
        dy_values = np.diff(y_values)
        dy_values = np.hstack((dy_values, dy_values[-1]))

        poses = [
            CurvePose(x, y, dx, dy)
            for x, y, dx, dy in zip(x_values, y_values, dx_values, dy_values)
        ]

        return s_values, poses


class CurvePose:
    """
    在曲线上进行采样，用于进行弗雷内到笛卡尔的转换。
    """

    def __init__(self, x: float, y: float, dx: float, dy: float):
        self.length = np.sqrt(dx**2 + dy**2)
        self.position = np.array([x, y]).flatten()
        self.normal = np.array([dx, dy]).flatten() / self.length
        self.orthonormal = np.array([-self.normal[1], self.normal[0]]).flatten()

    def distance_to_origin(self, point: Tuple[float, float]) -> float:
        """
        计算点 [x, y] 与姿态原点之间的距离。
        """
        return np.sqrt(np.sum((self.position - point) ** 2))

    def project_onto_normal(self, point: Tuple[float, float]) -> float:
        """
        通过将点投影到姿态的法线向量上，计算从姿态原点到点的纵向距离。
        """
        return self.normal.dot(point - self.position)

    def project_onto_orthonormal(self, point: Tuple[float, float]) -> float:
        """
        通过将点投影到姿态的正交向量上，计算从姿态原点到点的横向距离。
        """
        return self.orthonormal.dot(point - self.position)


if __name__ == '__main__':
    point = [[1,1], [2,2], [2,3], [3,3], [3,2]]
    ls = LinearSpline2D(point)