# ！/usr/bin/env python3
import numpy as np
import sys
import math
from ast import literal_eval

# 加载聚类数据
def init():
    cluster_dataset = list(np.load("F:\FCD data\cluster\cluster_dataset.npy"))
    labels = list(np.load("F:\FCD data\cluster\destination_labels.npy"))
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


# 两点之间的距离
def dis(point, p):
    return math.sqrt(np.square(point[0] - p[0]) + np.square(point[1] - p[1]))


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

    result = open(result_filepath, 'w')
    with open(filepath) as fr:
        for line in fr.readlines():
            if line == '' or line == '\n':
                continue
            trajectory = literal_eval(line.strip('\n'))
            if len(trajectory) <= 10:
                continue
            clusters = []
            regions = []
            for point in trajectory:
                point = list(map(float, point[0:2]))
                cluster_class = get_cluster_num(cluster_dict, point)
                clusters.append(cluster_class)
                regions.append(trajectory[4])
            result.write(str(clusters) + ";" + str(regions) + '\n')
    result.close()


def main(argv=None):
    if argv is None:
        argv = sys.argv
    filepath = "F:\FCD data\\trajectory\workday_trajectory\\youke_1"
    result_filepath = "F:\FCD data\\trajectory\workday_trajectory\\youke_1_result"
    classify_point(filepath, result_filepath)


if __name__ == '__main__':
    sys.exit(main())