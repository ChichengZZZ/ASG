import os
import sys
import random
import pandas as pd
import numpy as np
from NGSIM_env.envs import NGSIMGAILEnv
from NGSIM_env.utils import img_2_video
from Utils import *
import wandb
import pickle
from gym import wrappers
from matplotlib import pyplot as plt
from pykalman import KalmanFilter
from torch.distributions import MultivariateNormal, Normal, kl_divergence
print(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from matplotlib.font_manager import FontProperties
# zhfont = FontProperties(fname='/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf', size = 16)
# zhfont = FontProperties(fname="../examples/simsun.ttc", size=14)
# plt.rcParams['font.family'] = ['sans-serif']
# plt.rcParams['font.sans-serif'] = ['SimHei']

# plt.rc("font", family="Times New Roman")
dtype = torch.float64
torch.set_default_dtype(dtype)


def visualize_distribution(data_file, title=None, type_name='ttc', max_value=100, wandb_run=None):

    row_len = len(range(1, 10, 2))
    col_len = len(range(0, 21, 3))
    plt.rcParams['figure.figsize'] = [3 * col_len, 3 * row_len]
    fig1, axs1 = plt.subplots(row_len, col_len)

    for i, D in enumerate(range(1, 10, 2)):
        for j, N in enumerate(range(0, 21, 3)):
            save_name = "dis_{}_time_{}".format(D, N)
            assert os.path.exists(data_file)
            assert os.path.exists(os.path.join(data_file, save_name+'.json'))

            data = load('{}/{}.json'.format(data_file, save_name))
            data = data[0][save_name]
            TTC_THW = data['TTC_THW']
            TTC_THW = np.array(TTC_THW).squeeze(axis=1)
            np_data = TTC_THW[:, 2] if type_name == 'ttc' else TTC_THW[:, 3]

            np_data = np_data[np_data < max_value]
            if j == 0:
                np_data = np_data[np_data > 0.1]
                base_data_ttc = np_data.copy()
                base_data_thw = np_data.copy()
                axs1[i, j].hist(np_data, bins=50, color='sandybrown', density=True, alpha=0.5, histtype='stepfilled', edgecolor='none')
            else:
                axs1[i, j].hist(np_data, bins=50, color='lightskyblue', label='AV', density=True, alpha=0.5, histtype='stepfilled', edgecolor='none')

            if j > 0:
                if type_name == 'ttc':
                    axs1[i, j].hist(base_data_ttc, bins=50, color='sandybrown', label='No Adversarial Behavior', density=True, alpha=0.5, histtype='stepfilled', edgecolor='none')
                else:
                    axs1[i, j].hist(base_data_thw, bins=50, color='sandybrown', label='No Adversarial Behavior', density=True, alpha=0.5, histtype='stepfilled', edgecolor='none')

    cols = ['Time={}(s)'.format(s/10) for s in range(0, 21, 3)]
    rows = ['D={}(m)'.format(w) for w in range(1, 10, 2)]
    for a, col in zip(axs1[0], cols):
        a.set_title(col)
    for a, row in zip(axs1[:, 0], rows):
        a.set_ylabel(row, rotation=90, size='large')
    plt.tight_layout()
    fig1.suptitle('{} Distribution of AV'.format(type_name.upper()), fontsize=25)
    handles, labels = axs1[-1][-1].get_legend_handles_labels()
    fig1.legend(handles, labels, loc='upper right')
    if wandb_run is not None and title is not None:
        wandb_run.get_run().log({"Result/"+title: wandb.Image(plt)})
    plt.savefig('{}/trend_{}_max_value_{}.jpg'.format(data_file, type_name, max_value))


def visualize_collision_rate(data_file, title=None, wandb_run=None):
    row_len = len(range(1, 10, 2))
    col_len = len(range(0, 21, 3))
    fig1, axs1 = plt.subplots()
    for i, D in enumerate(range(1, 10, 2)):
        collision_rates = []
        for j, N in enumerate(range(0, 21, 3)):
            save_name = "dis_{}_time_{}".format(D, N)
            data = load('{}/{}.json'.format(data_file, save_name))
            data = data[0][save_name]
            crashed_num = data['ego_crashed']
            distance = data['distance']
            # print(save_name)
            c_r = crashed_num/distance
            # print('agent crashed num with av:', data['ego_crashed'])
            # print('agent driving distance:', data['distance'])
            # print('collision rate:', c_r)
            collision_rates.append(c_r)

        axs1.plot(np.arange(0, 2.0, 0.3), np.array(collision_rates), label='D={}m'.format(D))
    axs1.set_xticks(np.arange(0, 2.1, 0.3))
    # axs1.xlim((0, 2.1))
    # plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    fig1.legend()
    axs1.set_xlabel('seconds')
    axs1.set_ylabel('collision rate')
    axs1.set_title('Collision Rate Curve', fontsize=15)
    fig1.tight_layout()
    if wandb_run is not None and title is not None:
        wandb_run.get_run().log({"Result/"+title: wandb.Image(plt)})
    fig1.savefig(f'{data_file}/trend_collision_rate.jpg')


def show_ttc_proportion(data_file, wandb_run):

    # Define a function to calculate the proportion of values within a range
    def proportion_within_range(df, column_name, lower, upper, filter_min=0, filter_max=100):
        total_data = df[(df[column_name] > filter_min) & (df[column_name] < filter_max)]
        filtered_data = df[(df[column_name] >= lower) & (df[column_name] <= upper)]
        return round(len(filtered_data) / len(total_data), 3)

    row_len = len(range(1, 10, 2))
    col_len = len(range(0, 21, 3))

    for i, D in enumerate(range(1, 10, 2)):
        result_df = None
        for j, N in enumerate(range(0, 21, 3)):
            save_name = "dis_{}_time_{}".format(D, N)
            assert os.path.exists(data_file)
            assert os.path.exists(os.path.join(data_file, save_name + '.json'))

            data = load('{}/{}.json'.format(data_file, save_name))
            data = data[0][save_name]
            TTC_THW = data['TTC_THW']
            TTC_THW = np.array(TTC_THW).squeeze(axis=1)
            np_data_ttc = TTC_THW[:, 2]
            np_data_thw = TTC_THW[:, 3]
            df = pd.DataFrame.from_dict({
                'ttc': np_data_ttc, 'thw': np_data_thw
            })
            # Apply the function to calculate the proportion of values within the specified ranges for each group
            proportion_ttc = proportion_within_range(df, 'ttc', lower=0, upper=10, filter_min=0, filter_max=100)
            proportion_thw = proportion_within_range(df, 'thw', lower=0, upper=1, filter_min=0, filter_max=10)

            # Create a new dataframe to store the results
            if j == 0:
                result_df = pd.DataFrame.from_dict(
                    {'prop_ttc_[0,10](%)': [proportion_ttc * 100], 'prop_thw_[0,1](%)': [proportion_thw * 100], 'Distance': [D], 'Steps': [N]})
            else:
                result_df = pd.concat((result_df, pd.DataFrame.from_dict({'prop_ttc_[0,10](%)': [proportion_ttc * 100], 'prop_thw_[0,1](%)': [proportion_thw * 100], 'Distance': [D], 'Steps': [N]})))

            # show results
        print("TTC和THW的区间占比情况")
        print(result_df)
            # print("保存区间占比评估表格到{}目录下".format(output_path))
            # result_df.to_csv(os.path.join(
            #     output_path, 'proportion.csv'), index=False)

        wandb_table = wandb.Table(dataframe=result_df)
        wandb_run.get_run().log(
            {f"Table/Percentage of intervals for TTC & THW | Distance {D}": wandb_table}
        )
