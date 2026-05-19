import os
import sys
import pickle
import matplotlib
import pandas
import numpy as np
from scipy.stats import sem
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontManager
import subprocess
import wandb

# plt.rc("font", family="Times New Roman")
# project_dir = "/home/sang/wjc/project/AvdAgent_v2/AdvAgentV2"


def plot_data(expert_traj, expert_ttc_thw, scenario, color=['sandybrown', 'lightsteelblue'], WanDBRunProvider=None):
    fig, axs = plt.subplots(2, 2)
    expert_traj = np.array(expert_traj)
    expert_traj = expert_traj[:, -2:]
    expert_ttc_thw = np.array(expert_ttc_thw)
    compare_ttc = expert_ttc_thw[:, 0]
    compare_thw = expert_ttc_thw[:, 1]
    compare_ttc = compare_ttc[compare_ttc < 100]
    compare_thw = compare_thw[compare_thw < 10]
    steer = expert_traj[:, 0]
    acc = expert_traj[:, 1]
    axs[0, 0].hist(steer, bins=50, color=color[0], density=True, alpha=0.5)
    axs[0, 0].set_xlabel('Steering', fontsize=15)
    axs[0, 0].set_ylabel('Frequency', fontsize=15)
    axs[0, 1].hist(acc, bins=50, color=color[0], density=True, alpha=0.5)
    axs[0, 1].set_xlabel('Acceleration', fontsize=15)
    axs[0, 1].set_ylabel('Frequency', fontsize=15)
    axs[1, 0].hist(compare_ttc, bins=50, color=color[0], density=True, alpha=0.5)
    axs[1, 0].set_xlabel('TTC', fontsize=15)
    axs[1, 0].set_ylabel('Frequency', fontsize=15)
    axs[1, 1].hist(compare_thw, bins=50, color=color[0], density=True, alpha=0.5)
    axs[1, 1].set_xlabel('THW', fontsize=15)
    axs[1, 1].set_ylabel('Frequency', fontsize=15)
    plt.tight_layout()
    # handles, labels = axs[0][0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper right')
    if WanDBRunProvider is not None:
        WanDBRunProvider.get_run().log({"Distribution of natural data": wandb.Image(plt)})

    plt.savefig(f'{scenario}_ttc_thw.jpg')