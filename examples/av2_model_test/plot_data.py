import os
import sys
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


def plot_rewards(mean,
                 std_error,
                 natural_rewards,
                 adv_reward_list,
                 save_path,
                 name,
                 labels=None,
                 colors=None):
    fig, ax = plt.subplots(3, 1)
    ax[0].set_ylabel("rewards", fontsize=15)
    ax[0].set_xlabel('epoch', fontsize=15)
    ax[1].set_ylabel("rewards", fontsize=15)
    ax[1].set_xlabel('epoch', fontsize=15)
    ax[2].set_ylabel("rewards", fontsize=15)
    ax[2].set_xlabel('epoch', fontsize=15)
    for i in range(len(mean)):
        se = std_error[i] * 1.96
        x = range(mean[i].shape[0])
        ax[0].plot(x, mean[i], color=colors[i][0], lw=1, label=labels[i])
        ax[0].fill_between(x, mean[i] - se, mean[i] + se, color=colors[i][1], alpha=0.5)
        ax[1].plot(x, adv_reward_list[i], color=colors[i][0], lw=1)
        ax[2].plot(x, natural_rewards[i], color=colors[i][0], lw=1)

    ax[0].grid(alpha=.4)
    ax[1].grid(alpha=.4)
    ax[2].grid(alpha=.4)

    ax[0].set_title(f"Reward Curve With Epoch", fontsize=20)
    ax[1].set_title(f"Adversarial Reward Curve With Epoch", fontsize=20)
    ax[2].set_title(f"Natural Reward Curve With Epoch", fontsize=20)

    fig.gca().spines['top'].set_alpha(0)
    fig.gca().spines['bottom'].set_alpha(1)
    fig.gca().spines['right'].set_alpha(0)
    fig.gca().spines['left'].set_alpha(1)
    s, e = plt.gca().get_xlim()
    ax[0].set_xlim(s, e)
    ax[1].set_xlim(s, e)
    ax[2].set_xlim(s, e)
    fig.tight_layout()
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.savefig(os.path.join(save_path, f'{name}_reward_curve.png'))
    plt.show()


if __name__ == '__main__':
    # mpl_fonts = set(f.name for f in FontManager().ttflist)
    #
    # print('all font list get from matplotlib.font_manager:')
    # for f in sorted(mpl_fonts):
    #     print('\t' + f)

    scenario = 'highway'
    exp_names = {
        # 'highway': ["exp_2_improve_naturalness", "exp_1_trpo", "exp_1_a2c"],
        'highway': ["exp_1_rain", "exp_1", "exp_1", 'exp_1', 'exp_1'],
        'intersection': ["exp_1", "exp_1", "exp_1_a2c"]
    }

    analysis_statistic_data(scenario)


    # model_file = [f'examples/model_data/{scenario}/NaturalAdversarialAgent/ppo/diffusion/{exp_names[scenario][0]}',
    #               # f'examples/model_data/{scenario}/NaturalAdversarialAgent/trpo/diffusion/{exp_names[scenario][1]}',
    #               # f'examples/model_data/{scenario}/NaturalAdversarialAgent/a2c/diffusion/{exp_names[scenario][2]}',
    #               # f'examples/model_data/{scenario}/NaturalAdversarialAgent/sac/diffusion/{exp_names[scenario][3]}',
    #               # f'examples/model_data/{scenario}/NaturalAdversarialAgent/td3/diffusion/{exp_names[scenario][3]}',
    #               ]
    # reward_name = ['Reward']
    # for n in reward_name:
    #     means_list = []
    #     std_error_list = []
    #     natural_reward_list = []
    #     adv_reward_list = []
    #     for file in model_file:
    #         path = os.path.join(project_dir, file)
    #         data = pandas.read_csv(os.path.join(path, 'train.csv'))
    #         rewards_means = data["reward_mean"].ewm(span=100).mean()
    #         rewards_std_error = data['reward_std_error'].ewm(span=100).mean()
    #         adv_rewards = data["avg_step_adversarial_reward"].ewm(span=100).mean()
    #         natural_rewards_means = data["natural_reward"].ewm(span=100).mean()
    #         means_list.append(rewards_means.to_numpy())
    #         std_error_list.append(rewards_std_error.to_numpy())
    #         adv_reward_list.append(np.array(adv_rewards))
    #         natural_reward_list.append(natural_rewards_means.to_numpy())
    #
    #     plot_rewards(means_list, std_error_list, natural_reward_list, adv_reward_list,'./', scenario,
    #                  labels=['PPO', 'A2C', 'TRPO', 'SAC', 'TD3'],
    #                  colors=[('#FFA500', '#FFEFD5'),
    #                          ("#006400", "#8FBC8F"),
    #                          ("#FF4500", "#FFA07A"),
    #                          ("#A52A2A", "#F08080"),
    #                          ("#20B2AA", "#40E0D0")
    #                          ])


