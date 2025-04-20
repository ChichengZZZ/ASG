import os
import sys
current_dir = sys.path[0].replace("\\", "/")
project_dir = os.sep.join(current_dir.split('/')[:-2]).replace("\\", "/")
sys.path.append(project_dir)
import wandb
import pickle
import matplotlib
import pandas
import numpy as np
from scipy.stats import sem
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontManager
import subprocess
from Utils.wandb_run_provider import create_wandb_runner

scenario = ['adv_vissim', 'traffic_flow_vissim']
name = ["adv", "tflow"]
for i, s in enumerate(scenario):
    p = os.path.join(project_dir, 'examples', 'videos', s)
    wandb_run = create_wandb_runner('V2.0-评价指标', "聚类分析及视频实例", "视频")

    for j, n in enumerate(os.listdir(p)):
        video_file_path = os.path.join(p, n)
        title = f"video_{j}"
        wandb_run.get_run().log({f"Video/{name[i]}_"+title: wandb.Video(video_file_path)})

    del wandb_run