"""
统计数据集中 空驶时间和载客时间的比值
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
base_path = linux_path + "/trajectory_without_filter/"

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
    xunke_distance = 0.
    youke_time = 0.
    youke_distance = 0.

    for xunke_file in xunke:
        with open(xunke_file) as fr:
            for line in fr.readlines():
                if line == '' or line == '\n':
                    continue
                trajectory = literal_eval(line.strip('\n'))

                start_time = datetime.strptime(str(trajectory[0][3]), "%Y-%m-%d %H:%M:%S")
                end_time = datetime.strptime(str(trajectory[-1][3]), "%Y-%m-%d %H:%M:%S")

                if trajectory[0][1] == trajectory[-1][1] and trajectory[0][2] == trajectory[-1][2]:
                    continue

                xunke_time += (end_time - start_time).seconds / 60

                start_lng = float(trajectory[0][1])
                start_lat = float(trajectory[0][2])
                end_lng = float(trajectory[-1][1])
                end_lat = float(trajectory[-1][2])

                xunke_distance += haversine(start_lng, start_lat, end_lng, end_lat)

    youke_trajectories = np.load(youke).tolist()
    for trajectory in youke_trajectories:
        start_time = datetime.strptime(str(trajectory[0][3]), "%Y-%m-%d %H:%M:%S")
        end_time = datetime.strptime(str(trajectory[-1][3]), "%Y-%m-%d %H:%M:%S")

        youke_time += (end_time - start_time).seconds / 60

        start_lng = float(trajectory[0][1])
        start_lat = float(trajectory[0][2])
        end_lng = float(trajectory[-1][1])
        end_lat = float(trajectory[-1][2])

        youke_distance += haversine(start_lng, start_lat, end_lng, end_lat)

    print("day: {}, xunke distance: {}, youke distance: {}, xunke time: {}, youke time: {}, ".format(file_path, xunke_distance, youke_distance, xunke_time, youke_time))


