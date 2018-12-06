# ！/usr/bin/env python3
import numpy as np
import sys
import math
from ast import literal_eval
from datetime import datetime
"""
加载轨迹数据，并判断每条轨迹最有一个点所属的类簇
最后得到的是轨迹序列和轨迹最终的目的地
"""


# 加载聚类数据
def init():
    cluster_dataset = list(np.load("F:\FCD data/cluster/cluster_dataset.npy"))
    labels = list(np.load("F:\FCD data/cluster/destination_labels.npy"))
    cluster_dict = {}
    for index, value in enumerate(labels):
        if value == -1:
            continue

        if not cluster_dict.keys().__contains__(value):
            cluster_dict[value] = []
        # if cluster_dict[value] is None:
        #     cluster_dict[value] = []
        cluster_dict[value].append(list(cluster_dataset[index]))
    return cluster_dict


# 加载区域POI数据
def init_district_poi():
    poi_dict = np.load("F:\FCD data/shenzhen_map_poi/taz_region_poi/poi_dict.npy").item()
    return poi_dict


# 两点之间的距离
def dis(point, p):
    return math.sqrt(np.square(point[0] - p[0]) + np.square(point[1] - p[1]))


# 根据时间得到每周的第几天和每天时间的什么时刻（1440个timeslot）
def get_weekday_time(time):
    date, t = str(time).strip().split(" ")
    week_day = datetime.strptime(date, "%Y-%m-%d").weekday()
    hour, minute, second = t.split(":")
    timeslot = int(hour) * 60 + int(minute)
    return week_day, timeslot


def get_cluster_num(cluster_dict, point):
    min_distance = sys.float_info.max
    cluster_class = -1
    for key in cluster_dict.keys():
        cluster_points = cluster_dict[key]
        distance = 0
        for p in cluster_points:
            distance = distance + dis(point, p)
        if distance < min_distance:
            min_distance = distance
            cluster_class = key
    return cluster_class


def classify_point(filepath, result_filepath):
    cluster_dict = init()
    poi_dict = init_district_poi()

    trajectories = []
    result = open(result_filepath, 'w')
    with open(filepath, 'r') as fr:
        for line in fr.readlines():
            if line == '' or line == '\n':
                continue
            trajectory = literal_eval(line.strip('\n'))
            if len(trajectory) <= 10:
                continue
            week_day, timeslot = get_weekday_time(trajectory[0][3])
            for point in trajectory:
                point.append(poi_dict[int(point[-1])] if int(point[-1]) in poi_dict.keys() else -1)
            cluster_class = get_cluster_num(cluster_dict, list(map(float, trajectory[-1][1:3])))
            trajectory_destination = (trajectory, cluster_class, week_day, timeslot)
            trajectories.append(trajectory_destination)
            result.write(str(trajectory) + ";" + str(cluster_class) + '\n')
    result.close()
    np.save("F:\FCD data/trajectory/allday/youke_0_result_npy", trajectories)


def run():
    filepath = "F:\FCD data/trajectory/allday/youke_0"
    result_filepath = "F:\FCD data/trajectory/allday/youke_0_result"
    classify_point(filepath, result_filepath)


def main(argv=None):
    if argv is None:
        argv = sys.argv
    filepath = "F:\FCD data/trajectory/allday/youke_0"
    result_filepath = "F:\FCD data/trajectory/allday/youke_0_result"
    classify_point(filepath, result_filepath)


if __name__ == '__main__':
    sys.exit(main())