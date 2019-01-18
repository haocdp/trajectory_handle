"""
统计数据集中 空驶时间和载客时间的比值
"""
from ast import literal_eval
from datetime import datetime
import numpy as np

linux_path = "/root/taxiData"
windows_path = "F:/TaxiData"
base_path = windows_path + "/trajectory_without_filter/"

file_path1 = "2014-10-20"
file_path2 = "2014-10-21"
file_path3 = "2014-10-22"
file_path4 = "2014-10-23"
file_path5 = "2014-10-24"
file_path6 = "2014-10-25"
file_path7 = "2014-10-26"

file_paths = [file_path1, file_path2, file_path3, file_path4, file_path5, file_path6, file_path7]
for file_path in file_paths:
    xunke_0 = base_path + file_path + "/xunke_0"
    xunke_1 = base_path + file_path + "/xunke_1"
    xunke_2 = base_path + file_path + "/xunke_2"
    xunke = [xunke_0, xunke_1, xunke_2]

    youke = base_path + file_path + "/trajectory_" + file_path + ".npy"

    xunke_time = 0.
    youke_time = 0.

    for xunke_file in xunke:
        with open(xunke_file) as fr:
            for line in fr.readlines():
                if line == '' or line == '\n':
                    continue
                trajectory = literal_eval(line.strip('\n'))

                start_time = datetime.strptime(str(trajectory[0][3]), "%Y-%m-%d %H:%M:%S")
                end_time = datetime.strptime(str(trajectory[-1][3]), "%Y-%m-%d %H:%M:%S")

                xunke_time += (end_time - start_time).seconds

    youke_trajectories = np.load(youke).tolist()
    for trajectory in youke_trajectories:
        start_time = datetime.strptime(str(trajectory[0][3]), "%Y-%m-%d %H:%M:%S")
        end_time = datetime.strptime(str(trajectory[-1][3]), "%Y-%m-%d %H:%M:%S")

        youke_time += (end_time - start_time).seconds

    print("")


