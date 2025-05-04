import os
import sys
current_dir = sys.path[0].replace("\\", "/")  # 获取当前路径
project_dir = os.sep.join(current_dir.split('/')[:-2]).replace("\\", "/")  # 获取项目路径
sys.path.append(project_dir)  # 将项目路径添加到系统路径中
from NGSIM_env.envs.interaction_env import InterActionEnv  # 导入自定义的交互环境类
from NGSIM_env.envs.interaction_gail_env import InterActionGAILEnv  # 导入自定义的交互GAIL环境类
from NGSIM_env.utils import *  # 导入自定义的工具函数
from gym import wrappers  # 导入gym包中的wrappers模块
import os  # 导入os模块
import pickle  # 导入pickle模块

vehicle_dir = f'{project_dir}/NGSIM_env/data/trajectory_set/intersection/'  # 车辆轨迹文件夹路径
vehicle_names = os.listdir(vehicle_dir)[:]  # 获取车辆轨迹文件夹中的文件名列表

# select_vehicle_names = np.random.choice(vehicle_names, size=200, replace=False)  # 随机选择200个车辆轨迹文件名
all_trajs =[]  # 存储所有轨迹数据
all_ttc_thw = []  # 存储所有的TTC和THW数据
trajs_num = 0  # 轨迹数量
vehicle_names = [v for v in vehicle_names if 'P' not in v.split('.')[0].split('_')[-1]]  # 过滤出不包含'P'的车辆轨迹文件名
save_video = False  # 是否保存视频
vehicle = True  # 是否有车辆运行
IDM = True  # 是否使用IDM模型

for i, vn in enumerate(vehicle_names[:]):  # 遍历车辆轨迹文件名列表
    print("\r", end="")
    print("progress: {} %. ".format((i + 1) / len(vehicle_names[:]) * 100), end="")
    path = os.path.join(vehicle_dir, vn)  # 车辆轨迹文件路径
    vehicle_id = vn.split('.')[0].split('_')[-1]  # 车辆ID
    print(path)

    # vehicle_id = 317
    period = 0  # 仿真周期
    # env = InterActionEnv(path=path, vehicle_id=vehicle_id, IDM=IDM, render=True, save_video=save_video)  # 创建交互环境
    env = InterActionGAILEnv(path=path, vehicle_id=vehicle_id, IDM=False, render=False, gail=False)  # 创建交互GAIL环境

    if vehicle:
        env.reset()  # 重置环境
        env.reset(reset_time=0)  # 重置环境，设置重置时间为0
        terminated = False  # 是否中断
        while not terminated:
            # env.render()  # 渲染环境
            obs, reword, terminated, info = env.step(action=None)  # 执行一步动作

else:
        if save_video:
            env = wrappers.Monitor(env,
                                   directory="{}/examples/pedestrian_traffic_flow/videos/{}/".format(project_dir, vn.split('.')[0]),
                                   force=True, video_callable=lambda e: True)  # 创建视频监视器
            env.configure({"simulation_frequency": 3})  # 配置仿真频率
            env.set_monitor(env)  # 设置监视器
        expert_traj = []
        env.reset()
        if len(env.ped_ids) == 0:
            pass
            # env.reset(reset_time=1)  # video_save_name=f'{vn}_{start}'
            # # action = (7.314843035659861, 7.196439643428118, 5)
            # obs, reword, terminated, info = env.step(action=None)
            # # env.render()
            # expert_traj += info['features']
            # all_ttc_thw.extend(info['TTC_THW'])
            # trajs_num += 1
        else:
            for start in env.ped_ids.keys():
                env.reset(reset_time=start)  # 重置环境，设置重置时间为start
                # action = (7.314843035659861, 7.196439643428118, 5)
                terminated = False  # 是否中断
                while not terminated:
                    env.render()  # 渲染环境
                    obs, reword, terminated, info = env.step(action=None)  # 执行一步动作

                # expert_traj += info['features']
                    all_ttc_thw.extend(info['TTC_THW'])  # 将TTC和THW数据添加到列表中
                trajs_num += 1

        all_trajs += expert_traj

# print('trajs num:', trajs_num)
# pickle.dump(all_trajs, open(f'{project_dir}/NGSIM_env/data/interaction_all_trajs_v4.p', 'wb'))
# pickle.dump(all_ttc_thw, open(f'{project_dir}/NGSIM_env/data/interaction_all_ttc_thw_v4.p', 'wb'))