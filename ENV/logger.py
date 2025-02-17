from __future__ import division, print_function


class Loggable(object):
    """
        实现一个对象，可以通过时间记录其指标，并作为 pandas DataFrame 进行访问。
    """
    def dump(self):
        """
            更新对象数据的内部日志。
        """
        raise Exception('Not implemented.')

    def get_log(self):
        """
            将对象的内部日志转换为 pandas DataFrame。

            :return: 包含对象日志的 DataFrame
        """
        raise Exception('Not implemented.')

