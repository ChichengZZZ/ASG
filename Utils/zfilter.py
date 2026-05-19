import numpy as np

# from https://github.com/joschu/modular_rl
# http://www.johndcook.com/blog/standard_deviation/


import numpy as np

class RunningStat(object):
    def __init__(self, shape):
        self._n = 0  # 数据的数量
        self._M = np.zeros(shape)  # 平均值的累加和
        self._S = np.zeros(shape)  # 方差的累加和

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape  # 断言x的形状和平均值的形状相同
        self._n += 1  # 数据数量加1
        if self._n == 1:
            self._M[...] = x  # 若为第一个数据，直接将x赋值给平均值
        else:
            oldM = self._M.copy()  # 复制平均值
            self._M[...] = oldM + (x - oldM) / self._n  # 更新平均值
            self._S[...] = self._S + (x - oldM) * (x - self._M)  # 更新方差

    @property
    def n(self):
        return self._n  # 返回数据的数量

    @property
    def mean(self):
        return self._M  # 返回平均值

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)  # 返回方差

    @property
    def std(self):
        return np.sqrt(self.var)  # 返回标准差

    @property
    def shape(self):
        return self._M.shape  # 返回平均值的形状


class ZFilter:
    """
    y = (x-mean)/std
    使用运行估计的均值和标准差
    """

    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        self.demean = demean  # 是否进行去均值处理
        self.destd = destd  # 是否进行标准化处理
        self.clip = clip  # 值的范围上下限

        self.rs = RunningStat(shape)  # 运行统计类的实例
        self.fix = False  # 是否固定运行统计参数

    def __call__(self, x, update=True):
        if update and not self.fix:
            self.rs.push(x)  # 若需要更新运行统计参数，则将数据传入运行统计类中
        if self.demean:
            x = x - self.rs.mean  # 进行均值去除操作
        if self.destd:
            x = x / (self.rs.std + 1e-8)  # 进行标准化操作
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)  # 将值裁剪至指定范围
        return x  # 返回处理后的数据

if __name__ == "__main__":
    running_state = ZFilter((5,), clip=5)
    running_state.fix = True
    s1 = [5, 5, 15, 5, -5]
    s2 = [5, 5, 15, 5, 5]
    print(running_state(s1))
    print(running_state.rs._n)
    print(running_state.rs.mean)
    print(running_state.rs.std)

    print(running_state(s2))
    print(running_state.rs._n)
    print(running_state.rs.mean)
    print(running_state.rs.std)
    print(running_state.fix)
