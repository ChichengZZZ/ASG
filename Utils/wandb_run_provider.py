import wandb
import time
from Utils import print_banner
class WanDBRunProvider(object):
    '''
    一个用于为实验提供wandb运行对象的类。
    应该使用静态方法来获取运行对象。
    '''

    _run = None

    def __init__(self, *args, **kwargs):
        # 初始化WanDBRunProvider对象时，调用wandb.init方法创建一个运行对象，并将其赋值给_run类变量
        WanDBRunProvider._run = wandb.init(*args, **kwargs)

    @staticmethod
    def get_run():
        # 静态方法，返回_run类变量，即运行对象
        return WanDBRunProvider._run

    def __del__(self):
        # 在对象销毁时，输出“close wandb.”的信息，并调用运行对象的finish()方法进行清理操作
        print_banner('close wandb.')
        WanDBRunProvider._run.finish()
        # 将_run类变量置为None，释放引用
        WanDBRunProvider._run = None

def create_wandb_runner(project, name_zh, job_type, upload = True):
    # 按照指定的格式生成一个运行名称
    run_name = f"{name_zh}@{time.strftime('%y%m%d%H%M%S', time.localtime(time.time()))}"
    if upload:
        # 如果upload为True，则创建一个带上传模式的WanDBRunProvider对象并返回
        return WanDBRunProvider(project=project, group=name_zh, name=run_name, job_type=job_type, mode="online")
    else:
        # 如果upload为False，则创建一个禁用上传模式的WanDBRunProvider对象并返回
        return WanDBRunProvider(project=project, group=name_zh, name=run_name, job_type=job_type, mode="disabled")
