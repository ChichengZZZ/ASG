from enum import Enum
from contextlib import contextmanager
import numpy as np
import os
import os.path as osp
import sys
import datetime
import dateutil.tz
import csv
import json
import pickle
import errno
from collections import OrderedDict
from numbers import Number
import os

from tabulate import tabulate
import dateutil.tz
import os.path as osp


def dict_to_safe_json(d):
    """
    将字典中的每个值转换为可转换为JSON的基本类型。
    :param d: 要转换的字典
    :return: 转换后的新字典
    """
    new_d = {}
    for key, item in d.items():
        if safe_json(item):
            new_d[key] = item
        else:
            if isinstance(item, dict):
                new_d[key] = dict_to_safe_json(item)
            else:
                new_d[key] = str(item)
    return new_d

def safe_json(data):
    """
    判断数据是否可以转换为JSON的基本类型
    :param data: 要判断的数据
    :return: 是否可以转换为JSON的基本类型
    """
    if data is None:
        return True
    elif isinstance(data, (bool, int, float)):
        return True
    elif isinstance(data, (tuple, list)):
        return all(safe_json(x) for x in data)
    elif isinstance(data, dict):
        return all(isinstance(k, str) and safe_json(v) for k, v in data.items())
    return False

def create_exp_name(exp_prefix, exp_id=0, seed=0):
    """
    创建一个半唯一实验名称，包含一个时间戳
    :param exp_prefix: 实验名称的前缀
    :param exp_id: 实验的编号
    :param seed: 种子值
    :return: 实验名称
    """
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    return "%s_%s_%04d--s-%d" % (exp_prefix, timestamp, exp_id, seed)

def create_log_dir(
        exp_prefix,
        exp_id=0,
        seed=0,
        base_log_dir=None,
        include_exp_prefix_sub_dir=True,
):
    """
    创建并返回一个唯一的日志目录
    :param exp_prefix: 所有具有该前缀的实验将在此目录下有日志目录
    :param exp_id: 在实验中的特定实验运行编号
    :param base_log_dir: 所有日志应该保存的目录
    :return: 日志目录
    """
    exp_name = create_exp_name(exp_prefix, exp_id=exp_id, seed=seed)
    if base_log_dir is None:
        base_log_dir = './data'
    if include_exp_prefix_sub_dir:
        log_dir = osp.join(base_log_dir, exp_prefix.replace("_", "-"), exp_name)
    else:
        log_dir = osp.join(base_log_dir, exp_name)
    if osp.exists(log_dir):
        print("WARNING: Log directory already exists {}".format(log_dir), flush=True)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

def setup_logger(
        exp_prefix="default",
        variant=None,
        text_log_file="debug.log",
        variant_log_file="variant.json",
        tabular_log_file="progress.csv",
        snapshot_mode="last",
        snapshot_gap=1,
        log_tabular_only=False,
        log_dir=None,
        git_infos=None,
        script_name=None,
        **create_log_dir_kwargs
):
    """
    设置logger的一些合理的默认设置。
    将日志输出保存到based_log_dir/exp_prefix/exp_name。
    exp_name将是自动生成的唯一标识。
    如果指定了log_dir，则使用该目录作为输出目录。
    :param exp_prefix: 此特定实验的子目录
    :param variant: 变量信息
    :param text_log_file: 文本日志文件名称
    :param variant_log_file: 变量日志文件名称
    :param tabular_log_file: 表格日志文件名称
    :param snapshot_mode: 快照模式
    :param log_tabular_only: 仅记录表格日志
    :param snapshot_gap: 快照间隔
    :param log_dir: 日志目录
    :param git_infos: Git信息
    :param script_name: 脚本名称
    :return: 日志目录
    """
    first_time = log_dir is None
    if first_time:
        log_dir = create_log_dir(exp_prefix, **create_log_dir_kwargs)

    if variant is not None:
        logger.log("Variant:")
        logger.log(json.dumps(dict_to_safe_json(variant), indent=2))
        variant_log_path = osp.join(log_dir, variant_log_file)
        logger.log_variant(variant_log_path, variant)

    tabular_log_path = osp.join(log_dir, tabular_log_file)
    text_log_path = osp.join(log_dir, text_log_file)

    logger.add_text_output(text_log_path)
    if first_time:
        logger.add_tabular_output(tabular_log_path)
    else:
        logger._add_output(tabular_log_path, logger._tabular_outputs,
                           logger._tabular_fds, mode='a')
        for tabular_fd in logger._tabular_fds:
            logger._tabular_header_written.add(tabular_fd)
    logger.set_snapshot_dir(log_dir)
    logger.set_snapshot_mode(snapshot_mode)
    logger.set_snapshot_gap(snapshot_gap)
    logger.set_log_tabular_only(log_tabular_only)
    exp_name = log_dir.split("/")[-1]
    logger.push_prefix("[%s] " % exp_name)

    if script_name is not None:
        with open(osp.join(log_dir, "script_name.txt"), "w") as f:
            f.write(script_name)
    return log_dir


def create_stats_ordered_dict(
        name,
        data,
        stat_prefix=None,
        always_show_all_stats=True,
        exclude_max_min=False,
):
    # 如果 stat_prefix 不为空，则添加前缀到实验名称
    if stat_prefix is not None:
        name = "{}{}".format(stat_prefix, name)

    # 如果 data 是一个数值类型，则返回带有名称的有序字典
    if isinstance(data, Number):
        return OrderedDict({name: data})

    # 如果 data 中没有数据，则返回空的有序字典
    if len(data) == 0:
        return OrderedDict()

    # 如果 data 是一个元组，则递归调用 create_stats_ordered_dict 来处理每个元素，并将它们合并到一个有序字典中
    if isinstance(data, tuple):
        ordered_dict = OrderedDict()
        for number, d in enumerate(data):
            sub_dict = create_stats_ordered_dict(
                "{0}_{1}".format(name, number),
                d,
            )
            ordered_dict.update(sub_dict)
        return ordered_dict

    # 如果 data 是一个列表，则判断第一个元素是否为可迭代对象，如果是则将列表转换为 numpy 数组
    if isinstance(data, list):
        try:
            iter(data[0])
        except TypeError:
            pass
        else:
            data = np.concatenate(data)

    # 如果 data 是一个大小为 1 的 numpy 数组，且 always_show_all_stats 为 False，则返回带有名称的有序字典
    if (isinstance(data, np.ndarray) and data.size == 1
            and not always_show_all_stats):
        return OrderedDict({name: float(data)})

    # 计算数据的平均值和标准差，并存储在有序字典中
    stats = OrderedDict([
        (name + ' Mean', np.mean(data)),
        (name + ' Std', np.std(data)),
    ])
    # 如果 exclude_max_min 为 False，则计算数据的最大值和最小值，并存储在有序字典中
    if not exclude_max_min:
        stats[name + ' Max'] = np.max(data)
        stats[name + ' Min'] = np.min(data)
    return stats


# 定义一个类，用于在终端打印表格
class TerminalTablePrinter(object):
    def __init__(self):
        # 表头
        self.headers = None
        # 表格数据
        self.tabulars = []

    # 打印表格
    def print_tabular(self, new_tabular):
        # 如果表头为空，则设置表头为新表格的第一列
        if self.headers is None:
            self.headers = [x[0] for x in new_tabular]
        else:
            # 断言表头和新表格的列数相同
            assert len(self.headers) == len(new_tabular)
        # 将新表格的第二列添加到表格数据中
        self.tabulars.append([x[1] for x in new_tabular])
        # 刷新终端
        self.refresh()

    # 刷新终端
    def refresh(self):
        import os
        # 获取终端的行数和列数
        rows, columns = os.popen('stty size', 'r').read().split()
        # 取最近的行数 -3 行表格数据
        tabulars = self.tabulars[-(int(rows) - 3):]
        # 清空终端
        sys.stdout.write("\x1b[2J\x1b[H")
        # 使用tabulate函数将表格数据和表头打印到终端
        sys.stdout.write(tabulate(tabulars, self.headers))
        sys.stdout.write("\n")


# 定义一个自定义的JSON编码器
class MyEncoder(json.JSONEncoder):
    def default(self, o):
        # 如果对象是一个类，则返回类的模块和名称
        if isinstance(o, type):
            return {'$class': o.__module__ + "." + o.__name__}
        # 如果对象是枚举类型，则返回枚举的模块、类名称和枚举常量名称
        elif isinstance(o, Enum):
            return {
                '$enum': o.__module__ + "." + o.__class__.__name__ + '.' + o.name
            }
        # 如果对象是可调用的函数或方法，则返回函数或方法的模块和名称
        elif callable(o):
            return {
                '$function': o.__module__ + "." + o.__name__
            }
        # 使用默认的JSON编码器对其他对象进行编码
        return json.JSONEncoder.default(self, o)


# 定义一个函数，用于创建多级目录
def mkdir_p(path):
    try:
        # 创建目录，如果目录已存在则不抛出异常
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


class Logger(object):
    def __init__(self):
        self._prefixes = []  # 前缀列表，用于设置输出时的前缀
        self._prefix_str = ''  # 前缀字符串

        self._tabular_prefixes = []  # 表格前缀列表，用于设置输出表格时的前缀
        self._tabular_prefix_str = ''  # 表格前缀字符串

        self._tabular = []  # 表格数据列表

        self._text_outputs = []  # 文本输出文件列表
        self._tabular_outputs = []  # 表格输出文件列表

        self._text_fds = {}  # 文本输出文件对象字典
        self._tabular_fds = {}  # 表格输出文件对象字典
        self._tabular_header_written = set()  # 表格输出文件头信息写入标记集合

        self._snapshot_dir = None  # 快照文件保存目录
        self._snapshot_mode = 'all'  # 快照模式
        self._snapshot_gap = 1  # 快照间隔

        self._log_tabular_only = False  # 只记录表格数据
        self._header_printed = False  # 是否输出过表头
        self.table_printer = TerminalTablePrinter()  # 表格打印对象

    def reset(self):
        self.__init__()

    def _add_output(self, file_name, arr, fds, mode='a'):
        # 添加输出文件
        if file_name not in arr:
            mkdir_p(os.path.dirname(file_name))  # 创建目录
            arr.append(file_name)  # 添加到文件列表
            fds[file_name] = open(file_name, mode)  # 打开文件对象

    def _remove_output(self, file_name, arr, fds):
        # 移除输出文件
        if file_name in arr:
            fds[file_name].close()  # 关闭文件对象
            del fds[file_name]  # 删除文件对象
            arr.remove(file_name)  # 移除文件名

    def push_prefix(self, prefix):
        # 添加前缀
        self._prefixes.append(prefix)
        self._prefix_str = ''.join(self._prefixes)

    def add_text_output(self, file_name):
        # 添加文本输出文件
        self._add_output(file_name, self._text_outputs, self._text_fds, mode='a')

    def remove_text_output(self, file_name):
        # 移除文本输出文件
        self._remove_output(file_name, self._text_outputs, self._text_fds)

    def add_tabular_output(self, file_name, relative_to_snapshot_dir=False):
        # 添加表格输出文件
        if relative_to_snapshot_dir:
            file_name = osp.join(self._snapshot_dir, file_name)
        self._add_output(file_name, self._tabular_outputs, self._tabular_fds, mode='w')

    def remove_tabular_output(self, file_name, relative_to_snapshot_dir=False):
        # 移除表格输出文件
        if relative_to_snapshot_dir:
            file_name = osp.join(self._snapshot_dir, file_name)
        if self._tabular_fds[file_name] in self._tabular_header_written:
            self._tabular_header_written.remove(self._tabular_fds[file_name])
        self._remove_output(file_name, self._tabular_outputs, self._tabular_fds)

    def set_snapshot_dir(self, dir_name):
        # 设置快照文件保存目录
        self._snapshot_dir = dir_name

    def get_snapshot_dir(self, ):
        # 获取快照文件保存目录
        return self._snapshot_dir

    def get_snapshot_mode(self, ):
        # 获取快照模式
        return self._snapshot_mode

    def set_snapshot_mode(self, mode):
        # 设置快照模式
        self._snapshot_mode = mode

    def get_snapshot_gap(self, ):
        # 获取快照间隔
        return self._snapshot_gap

    def set_snapshot_gap(self, gap):
        # 设置快照间隔
        self._snapshot_gap = gap

    def set_log_tabular_only(self, log_tabular_only):
        # 设置只记录表格数据
        self._log_tabular_only = log_tabular_only

    def get_log_tabular_only(self, ):
        # 获取只记录表格数据标志
        return self._log_tabular_only

    def log(self, s, with_prefix=True, with_timestamp=True):
        # 记录日志
        out = s  # 记录的字符串
        if with_prefix:
            out = self._prefix_str + out  # 加上前缀
        if with_timestamp:
            now = datetime.datetime.now(dateutil.tz.tzlocal())  # 获取当前时间
            timestamp = now.strftime('%y-%m-%d.%H:%M')  # :%S
            out = "%s|%s" % (timestamp, out)  # 加上时间戳
        if not self._log_tabular_only:
            # 输出到stdout
            print(out, flush=True)
            for fd in list(self._text_fds.values()):
                fd.write(out + '\n')
                fd.flush()
            sys.stdout.flush()

    def record_tabular(self, key, val):
        # 记录表格数据
        self._tabular.append((self._tabular_prefix_str + str(key), str(val)))

    def record_dict(self, d, prefix=None):
        # 记录字典数据
        if prefix is not None:
            self.push_tabular_prefix(prefix)
        for k, v in d.items():
            self.record_tabular(k, v)
        if prefix is not None:
            self.pop_tabular_prefix()

    def push_tabular_prefix(self, key):
        # 添加表格前缀
        self._tabular_prefixes.append(key)
        self._tabular_prefix_str = ''.join(self._tabular_prefixes)

    def pop_tabular_prefix(self, ):
        # 移除表格前缀
        del self._tabular_prefixes[-1]
        self._tabular_prefix_str = ''.join(self._tabular_prefixes)

    def save_extra_data(self, data, file_name='extra_data.pkl', mode='joblib'):
        """
        Data saved here will always override the last entry
        :param data: Something pickle'able.
        """
        file_name = osp.join(self._snapshot_dir, file_name)
        if mode == 'joblib':
            import joblib
            joblib.dump(data, file_name, compress=3)
        elif mode == 'pickle':
            pickle.dump(data, open(file_name, "wb"))
        else:
            raise ValueError("Invalid mode: {}".format(mode))
        return file_name

    def get_table_dict(self, ):
        # 获取表格数据字典
        return dict(self._tabular)

    def get_table_key_set(self, ):
        # 获取表格数据键集合
        return set(key for key, value in self._tabular)

    @contextmanager
    def prefix(self, key):
        # 上下文管理器，用于添加前缀
        self.push_prefix(key)
        try:
            yield
        finally:
            self.pop_prefix()

    @contextmanager
    def tabular_prefix(self, key):
        # 上下文管理器，用于添加表格前缀
        self.push_tabular_prefix(key)
        yield
        self.pop_tabular_prefix()

    def log_variant(self, log_file, variant_data):
        # 记录变量数据
        mkdir_p(os.path.dirname(log_file))
        with open(log_file, "w") as f:
            json.dump(variant_data, f, indent=2, sort_keys=True, cls=MyEncoder)

    def record_tabular_misc_stat(self, key, values, placement='back'):
        # 记录表格的统计信息，如平均值、方差、中位数、最小值、最大值
        if placement == 'front':
            prefix = ""
            suffix = key
        else:
            prefix = key
            suffix = ""
        if len(values) > 0:
            self.record_tabular(prefix + "Average" + suffix, np.average(values))
            self.record_tabular(prefix + "Std" + suffix, np.std(values))
            self.record_tabular(prefix + "Median" + suffix, np.median(values))
            self.record_tabular(prefix + "Min" + suffix, np.min(values))
            self.record_tabular(prefix + "Max" + suffix, np.max(values))
        else:
            self.record_tabular(prefix + "Average" + suffix, np.nan)
            self.record_tabular(prefix + "Std" + suffix, np.nan)
            self.record_tabular(prefix + "Median" + suffix, np.nan)
            self.record_tabular(prefix + "Min" + suffix, np.nan)
            self.record_tabular(prefix + "Max" + suffix, np.nan)

    def dump_tabular(self, *args, **kwargs):
        # 将表格数据输出到日志文件
        wh = kwargs.pop("write_header", None)  # 获取kwargs中的write_header参数并从kwargs中删除
        if len(self._tabular) > 0:  # 如果_tabular列表长度大于0
            if self._log_tabular_only:  # 如果_log_tabular_only为真
                self.table_printer.print_tabular(self._tabular)  # 调用table_printer对象的print_tabular方法，并传入_tabular列表作为参数
            else:
                for line in tabulate(self._tabular).split('\n'):  # 将_tabular列表使用tabulate函数转化为表格形式，再以'\n'分割为列表的每一行
                    self.log(line, *args, **kwargs)  # 调用log方法，将每一行作为参数传入
            tabular_dict = dict(self._tabular)  # 将_tabular列表转化为字典形式
            # Also write to the csv files
            # This assumes that the keys in each iteration won't change!
            for tabular_fd in list(self._tabular_fds.values()):  # 遍历_tabular_fds.values()得到的列表
                writer = csv.DictWriter(tabular_fd,  # 创建一个csv.DictWriter对象，并传入tabular_fd作为文件对象
                                        fieldnames=list(tabular_dict.keys()))  # 设置fieldnames为_tabular列表的键列表
                if wh or (
                        wh is None and tabular_fd not in self._tabular_header_written):  # 如果wh为真或wh为None并且tabular_fd不在_tabular_header_written集合中
                    writer.writeheader()  # 调用writeheader方法，将fieldnames作为header写入文件
                    self._tabular_header_written.add(tabular_fd)  # 将tabular_fd添加到_tabular_header_written集合中
                writer.writerow(tabular_dict)  # 调用writerow方法，将tabular_dict作为参数传入
                tabular_fd.flush()  # 刷新tabular_fd文件缓冲区
            del self._tabular[:]  # 清空_tabular列表

    def pop_prefix(self, ):
        del self._prefixes[-1]  # 删除_prefixes列表最后一个元素
        self._prefix_str = ''.join(self._prefixes)  # 将_prefixes列表中的元素按顺序拼接为字符串，赋值给_prefix_str

    def save_itr_params(self, itr, params):
        if self._snapshot_dir:  # 如果_snapshot_dir不为None
            if self._snapshot_mode == 'all':  # 如果_snapshot_mode为'all'
                file_name = osp.join(self._snapshot_dir, 'itr_%d.pkl' % itr)  # 将_snapshot_dir和'itr_%d.pkl' % itr拼接为文件路径
                pickle.dump(params, open(file_name, "wb"))  # 使用pickle.dump函数将params写入文件
            elif self._snapshot_mode == 'last':  # 如果_snapshot_mode为'last'
                # override previous params
                file_name = osp.join(self._snapshot_dir, 'params.pkl')  # 将_snapshot_dir和'params.pkl'拼接为文件路径
                pickle.dump(params, open(file_name, "wb"))  # 使用pickle.dump函数将params写入文件
            elif self._snapshot_mode == "gap":  # 如果_snapshot_mode为'gap'
                if itr % self._snapshot_gap == 0:  # 如果itr除以_snapshot_gap的余数为0
                    file_name = osp.join(self._snapshot_dir,
                                         'itr_%d.pkl' % itr)  # 将_snapshot_dir和'itr_%d.pkl' % itr拼接为文件路径
                    pickle.dump(params, open(file_name, "wb"))  # 使用pickle.dump函数将params写入文件
            elif self._snapshot_mode == "gap_and_last":  # 如果_snapshot_mode为'gap_and_last'
                if itr % self._snapshot_gap == 0:  # 如果itr除以_snapshot_gap的余数为0
                    file_name = osp.join(self._snapshot_dir,
                                         'itr_%d.pkl' % itr)  # 将_snapshot_dir和'itr_%d.pkl' % itr拼接为文件路径
                    pickle.dump(params, open(file_name, "wb"))  # 使用pickle.dump函数将params写入文件
                file_name = osp.join(self._snapshot_dir, 'params.pkl')  # 将_snapshot_dir和'params.pkl'拼接为文件路径
                pickle.dump(params, open(file_name, "wb"))  # 使用pickle.dump函数将params写入文件
            elif self._snapshot_mode == 'none':  # 如果_snapshot_mode为'none'
                pass  # 什么也不做
            else:
                raise NotImplementedError  # 抛出NotImplementedError异常


logger = Logger()


def setup_logger(
        exp_prefix="default",                                        # 实验前缀，默认为"default"
        variant=None,                                                # 变量
        text_log_file="debug.log",                                   # 文本日志文件名，默认为"debug.log"
        variant_log_file="variant.json",                             # 变量日志文件名，默认为"variant.json"
        tabular_log_file="progress.csv",                             # 表格日志文件名，默认为"progress.csv"
        snapshot_mode="last",                                        # 快照模式，默认为"last"
        snapshot_gap=1,                                              # 快照间隔，默认为1
        log_tabular_only=False,                                      # 只记录表格日志，默认为False
        log_dir=None,                                                # 日志文件目录，默认为None
        git_infos=None,                                              # git信息，默认为None
        script_name=None,                                            # 脚本名称，默认为None
        my_logger = None,                                            # 自定义日志记录器，默认为None
        **create_log_dir_kwargs                                      # 其他通过create_log_dir函数创建日志目录时传递的参数
):
    """
    设置日志记录器的一些合理的默认设置。
    将日志输出保存到 based_log_dir/exp_prefix/exp_name 目录下。
    exp_name 将自动生成以保证其唯一性。
    如果指定了 log_dir，则使用该目录作为输出目录。
    :param exp_prefix: 这个实验的子目录。
    :param variant: 变量。
    :param text_log_file: 文本日志文件名。
    :param variant_log_file: 变量日志文件名。
    :param tabular_log_file: 表格日志文件名。
    :param snapshot_mode: 快照模式。
    :param log_tabular_only: 是否只记录表格日志。
    :param snapshot_gap: 快照间隔。
    :param log_dir: 日志文件目录。
    :param git_infos: git信息。
    :param script_name: 如果设置了，将该脚本的名称保存到这里。
    :return: 日志文件目录。
    """

    my_logger = logger if my_logger is None else my_logger           # 如果传入的my_logger为空，则使用默认的logger

    first_time = log_dir is None                                     # 判断是否是第一次调用函数
    if first_time:
        log_dir = create_log_dir(exp_prefix, **create_log_dir_kwargs) # 根据exp_prefix和create_log_dir_kwargs参数创建日志目录

    if variant is not None:
        my_logger.log("Variant:")                                    # 在日志中记录"Variant:"
        my_logger.log(json.dumps(dict_to_safe_json(variant), indent=2)) # 在日志中记录variant的JSON字符串表示
        variant_log_path = osp.join(log_dir, variant_log_file)      # 获取variant日志文件的路径
        my_logger.log_variant(variant_log_path, variant)            # 将variant记录到variant日志文件中

    tabular_log_path = osp.join(log_dir, tabular_log_file)          # 获取表格日志文件的路径
    text_log_path = osp.join(log_dir, text_log_file)                # 获取文本日志文件的路径

    my_logger.add_text_output(text_log_path)                        # 添加文本日志输出
    if first_time:
        my_logger.add_tabular_output(tabular_log_path)              # 如果是第一次调用函数，则添加表格日志输出
    else:
        my_logger._add_output(tabular_log_path, my_logger._tabular_outputs,  # 如果不是第一次调用函数，则将表格日志追加到已有的日志文件
                           my_logger._tabular_fds, mode='a')
        for tabular_fd in my_logger._tabular_fds:
            my_logger._tabular_header_written.add(tabular_fd)
    my_logger.set_snapshot_dir(log_dir)                             # 设置快照保存目录为日志目录
    my_logger.set_snapshot_mode(snapshot_mode)                      # 设置快照模式
    my_logger.set_snapshot_gap(snapshot_gap)                        # 设置快照间隔
    my_logger.set_log_tabular_only(log_tabular_only)                # 设置是否只记录表格日志
    exp_name = log_dir.split("/")[-1]                               # 获取日志目录的末尾作为实验名称
    my_logger.push_prefix("[%s] " % exp_name)                       # 在日志记录中添加一个前缀，指定实验名称

    if script_name is not None:
        with open(osp.join(log_dir, "script_name.txt"), "w") as f:   # 如果指定了script_name，则将脚本名称保存到日志目录下的script_name.txt文件中
            f.write(script_name)
    return log_dir                                                    # 返回日志目录
