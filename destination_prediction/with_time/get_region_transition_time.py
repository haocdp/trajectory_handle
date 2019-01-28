"""
统计所有载客轨迹在区域之间转移的时间，形成区域转移时间矩阵
"""
import sys
if sys.platform == 'linux':
    sys.path.append('/root/trajectory_handle/')

import numpy as np
from datetime import datetime
# torch.manual_seed(1)    # reproducible

linux_path = "/root/taxiData"
windows_path = "K:\毕业论文\TaxiData"
base_path = linux_path


def load_data():
    filepath1 = base_path + "/trajectory_without_filter/2014-10-20/trajectory_2014-10-20result_npy.npy"
    # filepath1 = base_path + "/trajectory/allday/youke_0_result_npy.npy"
    filepath2 = base_path + "/trajectory_without_filter/2014-10-21/trajectory_2014-10-21result_npy.npy"
    filepath3 = base_path + "/trajectory_without_filter/2014-10-22/trajectory_2014-10-22result_npy.npy"
    filepath4 = base_path + "/trajectory_without_filter/2014-10-23/trajectory_2014-10-23result_npy.npy"
    filepath5 = base_path + "/trajectory_without_filter/2014-10-24/trajectory_2014-10-24result_npy.npy"
    filepath6 = base_path + "/trajectory_without_filter/2014-10-25/trajectory_2014-10-25result_npy.npy"
    filepath7 = base_path + "/trajectory_without_filter/2014-10-26/trajectory_2014-10-26result_npy.npy"

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

    # region_transtion_time = list(np.load("region_transition_time.npy"))
    region_transtion_times = None
    if region_transtion_times is None:
        region_transtion_times = [[[] for j in range(0, 918)] for i in range(0, 918)]

    for trajectory, label, weekday, time_slot in all_trajectories:
        region_transtion_times[int(trajectory[0][-2])][int(trajectory[-1][-2])].append(
            (datetime.strptime(trajectory[-1][3], "%Y-%m-%d %H:%M:%S") -
            datetime.strptime(trajectory[0][3], "%Y-%m-%d %H:%M:%S")).seconds)

    region_transtion_time = [[0. for j in range(0, 918)] for i in range(0, 918)]
    for row_ix, row in enumerate(region_transtion_times):
        for col_ix, col in enumerate(row):
            if not len(region_transtion_times[row_ix][col_ix]) == 0:
                region_transtion_time[row_ix][col_ix] = sum(region_transtion_times[row_ix][col_ix]) / \
                                                        len(region_transtion_times[row_ix][col_ix])
    np.save("region_transition_time", region_transtion_time)


load_data()
