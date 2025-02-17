import numpy as np
import matplotlib.pyplot as plt
import copy

# QuinticPolynomial

class QuinticPolynomial(object):
    """
    代表了一个五次多项式轨迹的类，用于描述机器人运动的路径。
    """

    def __init__(self, xs, vxs, axs, xe, vxe, axe, T):
        """
        初始化函数，传入初始位置、速度、加速度，结束位置、速度、加速度和时间。

        Args:
            xs (float): 初始位置
            vxs (float): 初始速度
            axs (float): 初始加速度
            xe (float): 结束位置
            vxe (float): 结束速度
            axe (float): 结束加速度
            T (float): 时间

        Returns:
            None
        """
        # 存储初始状态和结束状态
        self.xs = xs
        self.vxs = vxs
        self.axs = axs
        self.xe = xe
        self.vxe = vxe
        self.axe = axe

        # 计算五次多项式的系数
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        # 构建线性方程组Ax = b，求解x = [a3, a4, a5]
        A = np.array([[T**3, T**4, T**5],
                      [3*T**2, 4*T**3, 5*T**4],
                      [6*T, 12*T**2, 20*T**3]])
        b = np.array([xe - self.a0 - self.a1*T - self.a2*T**2,
                      vxe - self.a1 - 2*self.a2*T,
                      axe - 2*self.a2])
        x = np.linalg.solve(A, b)

        # 存储五次多项式的系数
        self.a3 = x[0]
        self.a4 = x[1]
        self.a5 = x[2]

    def calc_point(self, t):
        """
        根据五次多项式的理论计算在给定时间点t处的位置。

        Args:
            t (float): 时间点

        Returns:
            xt (float): 位置
        """
        xt = self.a0 + self.a1 * t + self.a2 * t**2 + self.a3 * t**3 + self.a4 * t**4 + self.a5 * t**5

        return xt

    # 以下是各阶导数的计算方法
    def calc_first_derivative(self, t):
        """
        计算一阶导数，在给定时间点t处的速度。

        Args:
            t (float): 时间点

        Returns:
            vxt (float): 速度
        """
        vxt = self.a1 + 2 * self.a2 * t + 3 * self.a3 * t**2 + 4 * self.a4 * t**3 + 5 * self.a5 * t**4

        return vxt

    def calc_second_derivative(self, t):
        """
        计算二阶导数，在给定时间点t处的加速度。

        Args:
            t (float): 时间点

        Returns:
            axt (float): 加速度
        """
        axt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t**2 + 20 * self.a5 * t**3

        return axt

    def calc_third_derivative(self, t):
        """
        计算三阶导数，在给定时间点t处的加加速度。

        Args:
            t (float): 时间点

        Returns:
            jxt (float): 加加速度
        """
        jxt = 6 * self.a3 + 24 * self.a4 * t + 60 * self.a5 * t**2

        return jxt

# QuarticPolynomial

class QuarticPolynomial:
    """
    代表了一个四次多项式轨迹的类，用于描述机器人运动的路径。
    """

    def __init__(self, xs, vxs, axs, vxe, axe, time):
        """
        初始化函数，传入初始位置、速度、加速度，结束速度、加速度和时间。

        Args:
            xs (float): 初始位置
            vxs (float): 初始速度
            axs (float): 初始加速度
            vxe (float): 结束速度
            axe (float): 结束加速度
            time (float): 时间

        Returns:
            None
        """
        # 计算四次多项式的系数
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        # 构建线性方程组Ax = b，求解x = [a3, a4]
        A = np.array([[3 * time ** 2, 4 * time ** 3],
                      [6 * time, 12 * time ** 2]])
        b = np.array([vxe - self.a1 - 2 * self.a2 * time, axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        # 存储四次多项式的系数
        self.a3 = x[0]
        self.a4 = x[1]

    def calc_point(self, t):
        """
        根据四次多项式的理论计算在给定时间点t处的位置。

        Args:
            t (float): 时间点

        Returns:
            xt (float): 位置
        """
        xt = self.a0 + self.a1*t + self.a2*t**2 + self.a3*t**3 + self.a4*t**4

        return xt

    def calc_first_derivative(self, t):
        """
        计算一阶导数，在给定时间点t处的速度。

        Args:
            t (float): 时间点

        Returns:
            vxt (float): 速度
        """
        vxt = self.a1 + 2*self.a2*t + 3*self.a3*t**2 + 4*self.a4*t**3

        return vxt

    def calc_second_derivative(self, t):
        """
        计算二阶导数，在给定时间点t处的加速度。

        Args:
            t (float): 时间点

        Returns:
            axt (float): 加速度
        """
        axt = 2*self.a2 + 6*self.a3*t + 12*self.a4*t**2

        return axt

    def calc_third_derivative(self, t):
        """
        计算三阶导数，在给定时间点t处的加加速度。

        Args:
            t (float): 时间点

        Returns:
            jxt (float): 加加速度
        """
        jxt = 6*self.a3 + 24*self.a4 * t

        return jxt

# Frenet Path
class FrenetPath:

    def __init__(self):
        # Frenet位置
        self.t = []    # 时间序列
        self.d = []    # 斜向位置
        self.d_d = []  # 斜向速度
        self.d_dd = [] # 斜向加速度
        self.d_ddd = [] # 斜向加加速度
        self.s = []    # 纵向位置
        self.s_d = []  # 纵向速度
        self.s_dd = [] # 纵向加速度
        self.s_ddd = [] # 纵向加加速度
        self.cd = 0.0  # 斜向加速度（常数）
        self.cv = 0.0  # 纵向加速度（常数）
        self.cf = 0.0  # 路线成本（常数）

        # 全局位置
        self.x = []    # x坐标
        self.y = []    # y坐标
        self.yaw = []  # 航向角
        self.ds = []   # 距离
        self.c = []    # 曲率

def calc_frenet_paths(s_d, s_d_d, s_d_d_d, c_d, c_d_d, c_d_dd, target_search_area, speed_search_area, T):
    frenet_paths = []     # Frenet路径列表
    DT = 0.1             # 时间间隔[s]

    # 横向运动规划（为每个偏离目标点生成路径）
    di = target_search_area
    fp = FrenetPath()    # 创建FrenetPath对象

    lat_qp = QuinticPolynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, T) # 横向路径

    fp.t = [t for t in np.arange(0.0, T, DT)]   # 时间序列
    fp.d = [lat_qp.calc_point(t) for t in fp.t]  # 斜向位置
    fp.d_d = [lat_qp.calc_first_derivative(t) for t in fp.t]  # 斜向速度
    fp.d_dd = [lat_qp.calc_second_derivative(t) for t in fp.t]  # 斜向加速度
    fp.d_ddd = [lat_qp.calc_third_derivative(t) for t in fp.t]  # 斜向加加速度

    tv = speed_search_area
    tfp = copy.deepcopy(fp)
    lon_qp = QuarticPolynomial(s_d, s_d_d, s_d_d_d, tv, 0.0, T) # 纵向路径

    tfp.s = [lon_qp.calc_point(t) for t in fp.t]  # 纵向位置
    tfp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t]  # 纵向速度
    tfp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp.t]  # 纵向加速度
    tfp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp.t]  # 纵向加加速度

    frenet_paths.append(tfp)

    return frenet_paths

def calc_global_paths(fplist):
    for fp in fplist:
        for i in range(len(fp.s)):
            fy = fp.d[i]
            fx = fp.s[i]
            fp.x.append(fx)
            fp.y.append(fy)

    return fplist


def planner(s_d, s_d_d, s_d_d_d, c_d, c_d_d, c_d_dd, target_search_area, speed_search_area, T):
    fplist = calc_frenet_paths(s_d, s_d_d, s_d_d_d, c_d, c_d_d, c_d_dd, target_search_area, speed_search_area, T)
    fplist = calc_global_paths(fplist)

    return fplist

if __name__ == "__main__":
    s_d, s_d_d, s_d_d_d = 0, 5, 0 # Longitudinal
    c_d, c_d_d, c_d_dd = 0, 0, 0 # Lateral
    target_search_area, speed_search_area, T  = [0, 12/3.281, -12/3.281], np.linspace(5, 30, 10), 5

    paths = []
    for target in target_search_area:
        for speed in speed_search_area:
            path = planner(s_d, s_d_d, s_d_d_d, c_d, c_d_d, c_d_dd, target, speed, T)[0]
            paths.append(path)

    plt.figure(figsize=(10,6))
    ax = plt.axes()
    ax.set_facecolor("grey")

    for i in range(len(paths)):
        plt.plot(paths[i].x, paths[i].y)

    plt.plot(np.linspace(0, 90, 50), np.ones(50)*6/3.281, 'w--', linewidth=2)
    plt.plot(np.linspace(0, 90, 50), np.ones(50)*-6/3.281, 'w--', linewidth=2)
    
    plt.show()