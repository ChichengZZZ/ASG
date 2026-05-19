import os.path
import re
import shutil
import time
import math
import torch
import numpy as np
import re


def print_banner(s, separator="-", num_star=60):
    print(separator * num_star, flush=True)
    print(s, flush=True)
    print(separator * num_star, flush=True)


class Progress:
    def __init__(self, total, name='Progress', ncol=3, max_length=20, indent=0, line_width=100, speed_update_freq=100):
        # 初始化函数，设置相关参数
        self.total = total                   # 总的步数
        self.name = name                     # 进度条名称
        self.ncol = ncol                     # 进度条分栏数
        self.max_length = max_length         # 每个参数的最大长度
        self.indent = indent                 # 进度条缩进
        self.line_width = line_width         # 进度条的宽度
        self._speed_update_freq = speed_update_freq       # 更新速度的频率

        self._step = 0                       # 当前步数
        self._prev_line = '\033[F'           # 用于清除之前的行
        self._clear_line = ' ' * self.line_width       # 用于清空一行

        self._pbar_size = self.ncol * self.max_length       # 进度条尺寸
        self._complete_pbar = '#' * self._pbar_size         # 完成进度条
        self._incomplete_pbar = ' ' * self._pbar_size        # 未完成进度条

        self.lines = ['']                     # 参数行
        self.fraction = '{} / {}'.format(0, self.total)       # 完成比例

        self.resume()                        # 继续进度条

    def update(self, description, n=1):
        # 更新进度条
        self._step += n
        if self._step % self._speed_update_freq == 0:
            self._time0 = time.time()
            self._step0 = self._step
        self.set_description(description)         # 设置描述信息

    def resume(self):
        # 继续进度条
        self._skip_lines = 1
        print('\n', end='')
        self._time0 = time.time()
        self._step0 = self._step

    def pause(self):
        # 暂停进度条
        self._clear()
        self._skip_lines = 1

    def set_description(self, params=[]):
        # 设置描述信息

        if type(params) == dict:
            params = sorted([(key, val) for key, val in params.items()])

        # 清除前一行
        self._clear()

        # 计算完成比例和完成进度条
        percent, fraction = self._format_percent(self._step, self.total)
        self.fraction = fraction

        # 计算速度
        speed = self._format_speed(self._step)

        # 格式化参数
        num_params = len(params)
        nrow = math.ceil(num_params / self.ncol)
        params_split = self._chunk(params, self.ncol)
        params_string, lines = self._format(params_split)
        self.lines = lines

        # 拼接描述信息
        description = '{} | {}{}'.format(percent, speed, params_string)
        print(description)
        self._skip_lines = nrow + 1

    def append_description(self, descr):
        # 添加描述信息
        self.lines.append(descr)

    def _clear(self):
        # 清除当前行
        position = self._prev_line * self._skip_lines
        empty = '\n'.join([self._clear_line for _ in range(self._skip_lines)])
        print(position, end='')
        print(empty)
        print(position, end='')

    def _format_percent(self, n, total):
        # 格式化完成比例
        if total:
            percent = n / float(total)

            complete_entries = int(percent * self._pbar_size)
            incomplete_entries = self._pbar_size - complete_entries

            pbar = self._complete_pbar[:complete_entries] + self._incomplete_pbar[:incomplete_entries]
            fraction = '{} / {}'.format(n, total)
            string = '{} [{}] {:3d}%'.format(fraction, pbar, int(percent * 100))
        else:
            fraction = '{}'.format(n)
            string = '{} iterations'.format(n)
        return string, fraction

    def _format_speed(self, n):
        # 格式化速度
        num_steps = n - self._step0
        t = time.time() - self._time0
        speed = num_steps / t
        string = '{:.1f} Hz'.format(speed)
        if num_steps > 0:
            self._speed = string
        return string

    def _chunk(self, l, n):
        # 将列表分块
        return [l[i:i + n] for i in range(0, len(l), n)]

    def _format(self, chunks):
        # 格式化参数块
        lines = [self._format_chunk(chunk) for chunk in chunks]
        lines.insert(0, '')
        padding = '\n' + ' ' * self.indent
        string = padding.join(lines)
        return string, lines

    def _format_chunk(self, chunk):
        # 格式化参数块
        line = ' | '.join([self._format_param(param) for param in chunk])
        return line

    def _format_param(self, param):
        # 格式化参数
        k, v = param
        return '{} : {}'.format(k, v)[:self.max_length]

    def stamp(self):
        # 打印进度条
        if self.lines != ['']:
            params = ' | '.join(self.lines)
            string = '[ {} ] {}{} | {}'.format(self.name, self.fraction, params, self._speed)
            self._clear()
            print(string, end='\n')
            self._skip_lines = 1
        else:
            self._clear()
            self._skip_lines = 0

    def close(self):
        # 关闭进度条
        self.pause()


class Silent:
    # 创建一个名为Silent的类
    def __init__(self, *args, **kwargs):
        # 类的构造函数，接受任意数量的位置参数和关键字参数
        pass

    def __getattr__(self, attr):
        # 当对象的属性不存在时，调用此方法
        return lambda *args: None
        # 返回一个匿名函数，该函数接受任意数量的位置参数，并返回None


class EarlyStopping(object):
    # 创建一个名为EarlyStopping的类
    def __init__(self, tolerance=5, min_delta=0):
        # 类的构造函数，可接受tolerance和min_delta两个参数
        self.tolerance = tolerance
        # 将tolerance赋值给对象的tolerance属性
        self.min_delta = min_delta
        # 将min_delta赋值给对象的min_delta属性
        self.counter = 0
        # 初始化对象的counter属性为0
        self.early_stop = False
        # 初始化对象的early_stop属性为False

    def __call__(self, train_loss, validation_loss):
        # 当对象被调用时，执行此方法
        if (validation_loss - train_loss) > self.min_delta:
            # 如果验证损失减去训练损失大于min_delta
            self.counter += 1
            # counter加1
            if self.counter >= self.tolerance:
                # 如果counter大于等于tolerance
                return True
                # 返回True
        else:
            self.counter = 0
            # 否则，将counter重置为0
        return False
        # 返回False


def save_checkpoints(state, save_dir, is_best=False, model_name=''):
    # 定义一个保存检查点的函数，接受state、save_dir、is_best和model_name四个参数
    if not os.path.exists(save_dir):
        # 如果保存目录不存在
        os.makedirs(save_dir)
        # 创建保存目录
    filename = os.path.join(save_dir, model_name+'_ckpt.pth')
    # 拼接文件名
    torch.save(state, filename)
    # 保存state到文件
    new_model_name = re.sub(r'\d+', 'latest', model_name)
    # 使用正则表达式将model_name中的数字替换为'latest'，并赋值给new_model_name
    shutil.copyfile(filename, os.path.join(save_dir, new_model_name+'_ckpt.pth'))
    # 将保存的文件复制到新的文件名

    if is_best:
        # 如果is_best为True
        best_filename = os.path.join(save_dir, model_name+'_best_ckpt.pth')
        # 拼接最佳检查点文件名
        shutil.copyfile(filename, best_filename)
        # 复制最佳检查点文件


def get_current_time():
    # 定义一个获取当前时间的函数
    return time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())
    # 返回格式化后的当前时间字符串