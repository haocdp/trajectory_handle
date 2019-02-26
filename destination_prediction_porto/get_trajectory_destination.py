# ！/usr/bin/env python3
import numpy as np
import sys
import math
from ast import literal_eval
from datetime import datetime
import time
import csv
"""
加载轨迹数据，并判断每条轨迹最有一个点所属的类簇
最后得到的是轨迹序列和轨迹最终的目的地
"""

windows_path = "K:\毕业论文\TaxiData_Porto"
linux_path = "/root/taxiData"
base_path = windows_path

file_path = "2014-10-20"
dir_path = "/trajectory/"


# 加载聚类数据
def init():
    cluster_dataset = list(np.load(base_path + "/cluster/cluster_dataset_new.npy"))
    labels = list(np.load(base_path + "/cluster/destination_labels_new.npy"))
    cluster_dict = {}
    cluster_center_dict = {}
    for index, value in enumerate(labels):
        if value == -1:
            continue

        if not cluster_dict.keys().__contains__(value):
            cluster_dict[value] = []
        # if cluster_dict[value] is None:
        #     cluster_dict[value] = []
        cluster_dict[value].append(list(cluster_dataset[index]))

    for key in cluster_dict.keys():
        lng = np.mean([x[0] for x in cluster_dict[key]])
        lat = np.mean([x[1] for x in cluster_dict[key]])
        cluster_center_dict[key] = [lng, lat]
    return cluster_dict, cluster_center_dict


# 两点之间的距离
def dis(point, p):
    return math.sqrt(np.square(point[0] - p[0]) + np.square(point[1] - p[1]))


# 根据时间得到每周的第几天和每天时间的什么时刻（1440个timeslot）
def get_weekday_time(t):
    t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))

    date, t = str(t).strip().split(" ")
    week_day = datetime.strptime(date, "%Y-%m-%d").weekday()
    hour, minute, second = t.split(":")
    timeslot = int(hour) * 60 + int(minute)
    return week_day, timeslot


# 过滤轨迹，如果轨迹存在连续相同区域，则进行过滤
def filter(tra):
    first_index = tra[0]
    new_tra = [tra[0]]
    for t in tra:
        if t[-1] == first_index[-1]:
            continue
        new_tra.append(t)
        first_index = t
    return new_tra


def get_cluster_num(cluster_center_dict, point):
    min_distance = sys.float_info.max
    cluster_class = -1
    for key in cluster_center_dict.keys():
        cluster_points = cluster_center_dict[key]
        distance = dis(point, cluster_points)
        if distance < min_distance:
            min_distance = distance
            cluster_class = key
    return cluster_class


def classify_point(filepath, result_filepath):
    cluster_dict, cluster_center_dict = init()

    trajectories = []
    file = csv.reader(open("K:/毕业论文/TaxiData_Porto/train.csv", 'r'))

    all_count = len(file)

    count = 0
    flag = False
    for line in file:
        if not flag:
            flag = True
            continue
        if line == 'True':
            continue
        trajectory = literal_eval(line[-1])
        if len(trajectory) == 0:
            continue

        taxi_id = line[4]
        t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(line[5]))
        status = 1
        region = 0
        poi = 0
        trajectory = line[-1]

        new_trajectory = []
        if len(trajectory) <= 10:
            count += 1
            if count % 1000 == 0:
                print("has finish: {} %".format(float(count) / all_count * 100))
            continue
        week_day, timeslot = get_weekday_time(t)
        for point in trajectory:
            new_point = []
            new_point.append(taxi_id)
            new_point.append(point[0])
            new_point.append(point[-1])
            new_point.append(t)
            new_point.append(status)
            new_point.append(region)
            new_point.append(poi)
            new_trajectory.append(new_point)
            # point.append(poi_dict[int(point[-1])] if int(point[-1]) in poi_dict.keys() else -1)
        cluster_class = get_cluster_num(cluster_center_dict, trajectory[-1])
        trajectory_destination = (new_trajectory, cluster_class, week_day, timeslot)
        trajectories.append(trajectory_destination)
        # end_time = time.time()
        # print("cost time: {}".format(end_time - start_time))
        count += 1
        if count % 1000 == 0:
            print("has finish: {} %".format(float(count) / all_count * 100))
    np.save(base_path + "/trajectory_result", trajectories)


def run():
    filepath = base_path + "/train.csv"
    result_filepath = base_path + dir_path + "/trajectory_result"
    classify_point(filepath, result_filepath)


def main(argv=None):
    if argv is None:
        argv = sys.argv
    filepath = base_path + "/train.csv"
    result_filepath = base_path + dir_path + "/trajectory_result"
    classify_point(filepath, result_filepath)


if __name__ == '__main__':
    sys.exit(main())