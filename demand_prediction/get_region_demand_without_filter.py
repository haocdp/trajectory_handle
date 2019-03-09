"""
训练数据：（car_id, region_id, poi_id, week_day, time_slot）
"""
import os
import sys
if sys.platform == 'linux':
    sys.path.append('/root/trajectory_handle/')

import torch
from torch import nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as Data
import torch.nn.functional as F
import random
import logger
from demand_prediction.region2matrix import get_region_matrix

# torch.manual_seed(1)    # reproducible

linux_path = "/root/taxiData"
windows_path = "F:/TaxiData"
base_path = linux_path


def load_data():


    filepath1 = base_path + "/trajectory_without_filter/2014-11-03/trajectory_2014-11-03_result.npy"
    # filepath1 = base_path + "/trajectory/allday/youke_0_result_npy.npy"
    filepath2 = base_path + "/trajectory_without_filter/2014-11-04/trajectory_2014-11-04_result.npy"
    filepath3 = base_path + "/trajectory_without_filter/2014-11-05/trajectory_2014-11-05_result.npy"
    filepath4 = base_path + "/trajectory_without_filter/2014-11-06/trajectory_2014-11-06_result.npy"
    filepath5 = base_path + "/trajectory_without_filter/2014-11-07/trajectory_2014-11-07_result.npy"
    filepath6 = base_path + "/trajectory_without_filter/2014-11-08/trajectory_2014-11-08_result.npy"
    filepath7 = base_path + "/trajectory_without_filter/2014-11-09/trajectory_2014-11-09_result.npy"

    trajectories1 = list(np.load(filepath1))
    trajectories2 = list(np.load(filepath2))
    trajectories3 = list(np.load(filepath3))
    trajectories4 = list(np.load(filepath4))
    trajectories5 = list(np.load(filepath5))
    trajectories6 = list(np.load(filepath6))
    trajectories7 = list(np.load(filepath7))

    all_trajectories = []
    all_trajectories.extend(trajectories1)
    all_trajectories.extend(trajectories2)
    all_trajectories.extend(trajectories3)
    all_trajectories.extend(trajectories4)
    all_trajectories.extend(trajectories5)
    all_trajectories.extend(trajectories6)
    all_trajectories.extend(trajectories7)

    # 打乱
    # random.shuffle(all_trajectories)

    print("all trajectories num : {}".format(len(all_trajectories)))
    # count = len(all_trajectories) * 0.8

    # region_demand = list(np.load(base_path + "/demand_all/region_demand_without_filter.npy"))
    region_demand = None
    if region_demand is None:
        region_demand = []
        for i in range(7):
            region_demand.append([])
            for j in range(48):
                region_demand[i].append([])
                for k in range(28):
                    region_demand[i][j].append([])
                    for l in range(60):
                        region_demand[i][j][k].append(0)

    all_grid, region_to_ix = get_region_matrix()
    for trajectory, label, weekday, time_slot in all_trajectories:
        time_slot = int(time_slot / 30)
        i, j = region_to_ix[int(trajectory[0][-2])]
        region_demand[weekday][time_slot][i][j] += 1

    np.save(base_path + "/demand_all/region_demand_without_filter", region_demand)

    all_sum = 0
    for i in range(7):
        for j in range(48):
            for k in range(28):
                for l in range(60):
                    all_sum += region_demand[i][j][k][l]
    print(all_sum)


load_data()