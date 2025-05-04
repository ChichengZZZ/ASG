import os  # 导入os模块
import sys  # 导入sys模块
current_dir = sys.path[0].replace("\\", "/")  # 获取当前路径
project_dir = os.sep.join(current_dir.split('/')[:-2]).replace("\\", "/")  # 获取项目路径
sys.path.append(project_dir)  # 将项目路径添加到系统路径中
import numpy as np  # 导入numpy模块
from visualization_module.data_process import plot_data  # 导入自定义的数据处理模块
from NGSIM_env.envs.ngsim_env import NGSIMEnv  # 导入自定义的NGSIM环境类
from NGSIM_env.utils import *  # 导入自定义的工具函数
from gym import wrappers  # 导入gym包中的wrappers模块
import os  # 导入os模块
import pickle  # 导入pickle模块
import time  # 导入time模块
import argparse  # 导入argparse模块

if __name__ == '__main__':
    parser = argparse.ArgumentParser()  # 创建参数解析器
    parser.add_argument('--dataset-path', type=str, default="data/ngsim/trajectory_set", help="ngsim dataset path")  # 添加数据集路径参数
    parser.add_argument('--save-path', type=str, default="data/ngsim")  # 添加保存路径参数
    args = parser.parse_args()  # 解析参数

    print("project_dir==",project_dir)
    print("args.save_path==",args.save_path)

    vehicle_dir = f'{project_dir}/{args.dataset_path}'  # 车辆轨迹文件夹路径
    vehicle_names = os.listdir(vehicle_dir)[:]  # 获取车辆轨迹文件夹中的文件名列表
    all_trajs =[]  # 存储所有轨迹数据
    TTC_THW = []  # 存储所有的TTC和THW数据
    crashed_num = 0  # 碰撞次数
    offroad_num = 0  # 离开道路次数
    trajs_num = 0  # 轨迹数量
    total_distance = 0  # 总距离
    skip = 0  # 跳过次数
    for i, vn in enumerate(vehicle_names[:]):  # 遍历车辆轨迹文件名列表
        print("\r", end="")
        print("progress: {} %. ".format((i + 1) / len(vehicle_names[:]) * 100), end="")
        path = os.path.join(vehicle_dir, vn)  # 车辆轨迹文件路径
        vehicle_id = vn.split('.')[0].split('_')[-1]  # 车辆ID
        period = 0  # 仿真周期
        env = NGSIMEnv(scene='us-101', path=path, period=period, vehicle_id=vehicle_id, IDM=False, show=True)  # 创建NGSIM环境

        expert_traj = []
        env.reset(reset_time=0,)  # 重置环境
        if env.spd_mean < 5:
            skip += 1
            continue

        obs, reword, terminated, info = env.step(action=None)  # 执行一步动作
        total_distance += info['distance']  # 累计行驶距离

        if info['crashed']:
            crashed_num += 1  # 碰撞次数加一

        if info['offroad']:
            offroad_num += 1  # 离开道路次数加一

        expert_traj += info["features"]
        TTC_THW.extend(info["TTC_THW"])  # 将TTC和THW数据添加到列表中
        trajs_num += 1

        all_trajs += expert_traj

    print(np.array(all_trajs).shape)  # 打印轨迹数据的形状

    total_distance /= 1000
    pickle.dump(all_trajs, open(f'{project_dir}/{args.save_path}/ngsim_clear_day_all_lc_trajs_v3.p', 'wb'))  # 保存轨迹数据到文件中
    pickle.dump(TTC_THW, open(f'{project_dir}/{args.save_path}/ngsim_clear_day_all_lc_thw_v3.p', 'wb'))  # 保存TTC和THW数据到文件中