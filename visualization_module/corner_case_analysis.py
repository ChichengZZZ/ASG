import hdbscan
import numpy
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import random
import pandas as pd
import numpy as np
from pathlib import Path
import os
import re
from argparse import ArgumentParser
import json
import time
import wandb

"""
# 算法流程
1.获取features:shape:(n, s)
2.如果需要可视化,需要将featues进行降维,shape:(n, 2)
3.如果需要可视化则使用降维的数据进行聚类,否则对原始数据聚类,返回labels。
5.从每个类别中随机挑选N个样例。
6.获取每个样例对应的初始化参数,通过初始化参数进行重现。
"""

MARKERS = ['x', ',', 'o', 'v', '^', '<', '>',
           '8', 's', 'p', '*', '+', 'D', 'd', '_', '|']


# 采用hdbscan算法进行聚类,该算法是dbscan的改进版,对epsilon设置不敏感
def make_cluster(features: numpy.ndarray, dim_reduction=True, n_cluster=5):
    if dim_reduction:
        features = PCA(n_components=2).fit_transform(features)

    # search for best min_cluster_size
    min_cluster_size = 5
    res = []
    data_size = len(features)
    max_min_cluster_size = max(int(data_size/(n_cluster**2)), 21)
    print(f'Total collision data count:{data_size}')
    for min_cluster_size in range(2, max_min_cluster_size):
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        cluster_labels = clusterer.fit_predict(features)
        n = numpy.max(cluster_labels) + 1
        res.append((min_cluster_size, n))

        # If found expected size, break
        if n == n_cluster:
            break

    # get the min_cluster_size which makes the cluster number closest to n_cluster
    min_cluster_size = min(res, key=lambda x: abs(x[1] - n_cluster))[0]
    print("min_cluster_size: {}.".format(min_cluster_size))

    # make cluster
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    cluster_labels = clusterer.fit_predict(features)
    print("class num: {}.".format(numpy.max(cluster_labels) + 1))
    return features, cluster_labels


# 绘制聚类效果
def draw_cluster_result(features, labels, output_path, wandb_run):
    assert features.shape[1] == 2, "features shape must be (n, 2)."
    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.rcParams['figure.figsize'] = [4, 3]
    plt.rcParams['figure.dpi'] = 300

    cluster_set = set(labels)

    for i in cluster_set:
        data = []
        for j, label in enumerate(labels):
            if i == label:
                data.append(features[j])
        data = np.array(data)
        if i == -1:
            # set the color of noise to black
            continue
            ax.scatter(data[:, 0], data[:, 1], marker='.',
                       label='noise', color='black', alpha=0.07, zorder=1)
        else:
            ax.scatter(data[:, 0], data[:, 1], marker=MARKERS[i % len(MARKERS)],
                       label=' cluster-{}'.format(i), alpha=0.4, zorder=3)
        print(f"label {i} cluster size: {len(data)}.")

    plt.legend(prop={'family': 'Times New Roman', 'size': 12})
    plt.rcParams["font.family"] = "Times New Roman"
    plt.title("Search Cluster Result")  # 设置标题
    # fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.xlabel('x', fontdict={'family': 'Times New Roman', 'size': 16})
    plt.ylabel('y', fontdict={'family': 'Times New Roman', 'size': 16})
    # plt.savefig(os.path.join(output_path, "SearchCluster.png"))
    # plt.savefig(os.path.join(output_path, "SearchCluster.pdf"), bbox_inches='tight', dpi=300)
    # print("SearchCluster.png saved in {}.".format(output_path))
    if wandb_run is not None:
        wandb_run.get_run().log({
            'Result/Collision Cluster Result': wandb.Image(plt)
        })
    plt.savefig(os.path.join(output_path, "SearchCluster.png"))
    # plt.savefig(os.path.join(output_path, "SearchCluster.pdf"), bbox_inches='tight', dpi=300)
    print("SearchCluster.png saved in {}.".format(output_path))
    # plt.show()
