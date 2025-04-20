import os
import sys

import Utils

current_dir = sys.path[0].replace("\\", "/")
project_dir = os.sep.join(current_dir.split('/')[:-2]).replace("\\", "/")
sys.path.append(project_dir)
import pickle
import matplotlib
import pandas
import numpy as np
from scipy.stats import sem
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontManager
import subprocess
from argparse import ArgumentParser
plt.rc("font", family="Times New Roman")
# project_dir = "/home/sang/wjc/project/AvdAgent_v2/AdvAgentV2"


def analysis_statistic_data(scenario, name=['clear', 'rain'], color=['sandybrown', 'lightsteelblue']):
    fig, axs = plt.subplots(2, 2)
    for i in range(2):
        expert_traj_path = f'{project_dir}/NGSIM_env/data/ngsim_{name[i]}_day_all_lc_trajs_v4.p'
        expert_traj = pickle.load(open(expert_traj_path, "rb"))
        expert_traj = np.array(expert_traj)
        expert_traj = expert_traj[:, -2:]
        expert_ttc_thw_path = f'{project_dir}/NGSIM_env/data/ngsim_{name[i]}_day_all_lc_thw_v4.p'
        expert_ttc_thw = pickle.load(open(expert_ttc_thw_path, "rb"))
        expert_ttc_thw = np.array(expert_ttc_thw)
        compare_ttc = expert_ttc_thw[:, 0]
        compare_thw = expert_ttc_thw[:, 1]
        compare_ttc = compare_ttc[compare_ttc < 100]
        compare_thw = compare_thw[compare_thw < 10]
        steer = expert_traj[:, 0]
        acc = expert_traj[:, 1]
        axs[0, 0].hist(steer, bins=50, color=color[i], label=f'{name[i]} day', density=True, alpha=0.5)
        axs[0, 0].set_xlabel('steering', fontsize=15)
        axs[0, 1].hist(acc, bins=50, color=color[i], density=True, alpha=0.5)
        axs[0, 1].set_xlabel('acceleration', fontsize=15)
        axs[1, 0].hist(compare_ttc, bins=50, color=color[i], density=True, alpha=0.5)
        axs[1, 0].set_xlabel('ttc', fontsize=15)
        axs[1, 1].hist(compare_thw, bins=50, color=color[i], density=True, alpha=0.5)
        axs[1, 1].set_xlabel('thw', fontsize=15)
    plt.tight_layout()
    handles, labels = axs[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.savefig(f'./{scenario}_ttc_thw.jpg')


def plot_loss(args):
    if args.scenario == 'merge':
        p = os.path.join(project_dir, 'examples/model_data', args.scenario, args.model, args.exp_name, 'train.csv')
    elif args.model is None:
        p = os.path.join(project_dir, 'examples/model_data', args.scenario, args.agent_name, args.update_mode, args.exp_name, 'train.csv')
    else:
        p = os.path.join(project_dir, 'examples/model_data', args.scenario, args.agent_name, args.model,
                         args.supervise_model, args.exp_name, 'train.csv')
    Utils.print_banner(f'Load Data from {p}')
    assert os.path.exists(p)
    data = pandas.read_csv(p)
    col = data.columns
    for i in range(len(col)):
        data[col[i]].plot(subplots=True, figsize=(6, 6))
        plt.title(col[i])
        plt.show()
    # loss = data[args.index_name].to_numpy()
    # fig, ax = plt.subplots()
    # ax.plot(range(loss.shape[0]), loss, color='blue', lw=1, label=args.agent_name)
    # ax.grid(alpha=.4)
    # ax.set_title(f"Train Loss Curve With Epoch", fontsize=20)
    #
    # fig.gca().spines['top'].set_alpha(0)
    # fig.gca().spines['bottom'].set_alpha(1)
    # fig.gca().spines['right'].set_alpha(0)
    # fig.gca().spines['left'].set_alpha(1)
    # s, e = plt.gca().get_xlim()
    # ax.set_xlim(s, e)
    # fig.tight_layout()
    # handles, labels = ax.get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper right')
    # plt.savefig(f'./{args.agent_name}_{args.exp_name}_loss_curve.png')


def make_args():
    args = ArgumentParser()
    args.add_argument('--agent-name', default='human_driving_policy', type=str, required=False, help=""" It's one of three: human_driving_policy, adversarial_model and natural_adversarial_model.""")
    args.add_argument('--scenario', default='merge', type=str, required=False, help=""" It's one of two: highway and intersection.""")
    args.add_argument('--supervise-model', default='diffusion', type=str, required=False, help="""It's one of two: gail, diffusion""")
    args.add_argument('--exp-name', default='exp_1_ddqn', type=str)
    args.add_argument('--model', default="DQNAgent", type=str)
    args.add_argument('--update-mode', default='ddpg', type=str)
    args.add_argument('--index-name', default='reward_mean', type=str) # 'BC Loss', 'reward_mean', 'natural_reward'
    return args


if __name__ == '__main__':

    args = make_args().parse_args()
    plot_loss(args)



