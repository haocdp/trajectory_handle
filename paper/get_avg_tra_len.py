"""
统计数据集中轨迹点数据采集的时间间隔
"""

import sys

if sys.platform == 'linux':
    sys.path.append('/root/trajectory_handle/')
from ast import literal_eval
from datetime import datetime
import numpy as np
from trajectory.cal_distance import haversine


linux_path = "/root/taxiData"
windows_path = "H:/TaxiData"
base_path = windows_path + "/trajectory_without_filter/"

file_path1 = "2014-10-20"
file_path2 = "2014-10-21"
file_path3 = "2014-10-22"
file_path4 = "2014-10-23"
file_path5 = "2014-10-24"
file_path6 = "2014-10-25"
file_path7 = "2014-10-26"

# file_paths = [file_path1, file_path2, file_path3, file_path4, file_path5, file_path6, file_path7]
file_paths = [file_path5]
for file_path in file_paths:
    youke = base_path + file_path + "/trajectory_" + file_path + ".npy"

    youke_times = []

    youke_trajectories = np.load(youke).tolist()
    for trajectory in youke_trajectories:
        youke_times.append((datetime.strptime(str(trajectory[-1][3]), "%Y-%m-%d %H:%M:%S") -
             datetime.strptime(str(trajectory[0][3]), "%Y-%m-%d %H:%M:%S")).seconds)

    print("day: {}, youke_avg time: {}".format(file_path, sum(youke_times) / len(youke_times)))