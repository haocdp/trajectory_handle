# ！/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import json
import time
from sklearn.cluster import DBSCAN
from sklearn import metrics
from collections import Counter
from ast import literal_eval


file_path = "H:/TaxiData"

def load_dataset(filename):
    origin_dataSet = []
    destination_dataSet = []

    trajectories = np.load(filename).tolist()
    for trajectory, label, time_slot, weekday in trajectories:
        if len(trajectory) <= 10:
            continue
        origin_dataSet.append(list(map(float, trajectory[0][1:3])))
        destination_dataSet.append(list(map(float, trajectory[-1][1:3])))

    filename2 = file_path + "/trajectory/2014-10-21/trajectory_2014-10-21_result_npy.npy"
    filename3 = file_path + "/trajectory/2014-10-22/trajectory_2014-10-22_result_npy.npy"
    filename4 = file_path + "/trajectory/2014-10-23/trajectory_2014-10-23_result_npy.npy"
    filename5 = file_path + "/trajectory/2014-10-24/trajectory_2014-10-24_result_npy.npy"
    filename6 = file_path + "/trajectory/2014-10-25/trajectory_2014-10-25_result_npy.npy"
    filename7 = file_path + "/trajectory/2014-10-26/trajectory_2014-10-26_result_npy.npy"

    flag = True
    trajectories = np.load(filename2).tolist()
    for trajectory, label, time_slot, weekday in trajectories:
        if len(trajectory) <= 10 and flag:
            flag = not flag
            continue
        flag = not flag
        origin_dataSet.append(list(map(float, trajectory[0][1:3])))
        destination_dataSet.append(list(map(float, trajectory[-1][1:3])))

    trajectories = np.load(filename3).tolist()
    for trajectory, label, time_slot, weekday in trajectories:
        if len(trajectory) <= 10 and flag:
            flag = not flag
            continue
        flag = not flag
        origin_dataSet.append(list(map(float, trajectory[0][1:3])))
        destination_dataSet.append(list(map(float, trajectory[-1][1:3])))

    trajectories = np.load(filename4).tolist()
    for trajectory, label, time_slot, weekday in trajectories:
        if len(trajectory) <= 10 and flag:
            flag = not flag
            continue
        flag = not flag
        origin_dataSet.append(list(map(float, trajectory[0][1:3])))
        destination_dataSet.append(list(map(float, trajectory[-1][1:3])))

    trajectories = np.load(filename5).tolist()
    for trajectory, label, time_slot, weekday in trajectories:
        if len(trajectory) <= 10 and flag:
            flag = not flag
            continue
        flag = not flag
        origin_dataSet.append(list(map(float, trajectory[0][1:3])))
        destination_dataSet.append(list(map(float, trajectory[-1][1:3])))

    trajectories = np.load(filename6).tolist()
    for trajectory, label, time_slot, weekday in trajectories:
        if len(trajectory) <= 10 and flag:
            flag = not flag
            continue
        flag = not flag
        origin_dataSet.append(list(map(float, trajectory[0][1:3])))
        destination_dataSet.append(list(map(float, trajectory[-1][1:3])))

    trajectories = np.load(filename7).tolist()
    for trajectory, label, time_slot, weekday in trajectories:
        if len(trajectory) <= 10 and flag:
            flag = not flag
            continue
        flag = not flag
        origin_dataSet.append(list(map(float, trajectory[0][1:3])))
        destination_dataSet.append(list(map(float, trajectory[-1][1:3])))

    return origin_dataSet, destination_dataSet


def main():
    # dataSet = load_dataset('F:\FCD data\\trajectory\workday_point\\cluster_test.txt')
    # dataSet = np.mat(dataSet).transpose()
    # print(dataSet)
    # clusters, clusterNum = dbscan(dataSet, 2, 15)
    # print("cluster Numbers = ", clusterNum)
    # print(clusters)
    # plotFeature(dataSet, clusters, clusterNum)

    origin_dataset, destination_dataSet = load_dataset(file_path + '/trajectory/2014-10-20/trajectory_2014-10-20_result_npy.npy')
    origin_dataset_array = np.array(origin_dataset)
    destination_dataset_array = np.array(destination_dataSet)
    origin_db = DBSCAN(eps=0.0002, min_samples=40, metric='haversine').fit(origin_dataset_array)
    destination_db = DBSCAN(eps=0.0002, min_samples=40, metric='haversine').fit(destination_dataset_array)

    origin_labels = origin_db.labels_
    destination_labels = destination_db.labels_

    '''
        origin points cluster
    '''
    origin_new_dataset = []
    origin_new_labels = []
    origin_list_labels = list(origin_labels)
    origin_dict_lables = Counter(origin_list_labels)
    for key, value in enumerate(origin_list_labels):
        if value == -1 or origin_dict_lables[value] < 20:
            continue
        origin_new_dataset.append(origin_dataset[key])
        origin_new_labels.append(value)
    origin_new_dataset = np.array(origin_new_dataset)
    origin_dict_lables = Counter(origin_new_labels)
    print(origin_dict_lables)

    '''
        destination points cluster
    '''
    destination_new_dataset = []
    destination_new_labels = []
    destination_list_labels = list(destination_labels)
    destination_dict_lables = Counter(destination_list_labels)
    for key, value in enumerate(destination_list_labels):
        if value == -1 or destination_dict_lables[value] < 10:
            continue
        destination_new_dataset.append(destination_dataSet[key])
        destination_new_labels.append(value)
    destination_new_dataset = np.array(destination_new_dataset)
    destination_dict_lables = Counter(destination_new_labels)
    print(destination_dict_lables)
    np.save("D:\haoc\data\TaxiData\cluster\cluster_dataset_new", destination_new_dataset)
    np.save("D:\haoc\data\TaxiData\cluster\destination_labels_new", destination_new_labels)


    # single_class_dataset = []
    # single_class_labels = []
    # for key, value in enumerate(list_labels):
    #     if value == -1:
    #         continue
    #     single_class_dataset.append(dataset[key])
    #     single_class_labels.append(value)
    # single_class_dataset = np.array(single_class_dataset)


    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(destination_labels)) - (1 if -1 in destination_labels else 0)
    origin_n_clusters_ = len(set(origin_labels)) - (1 if -1 in origin_labels else 0)

    print('Estimated number of destination clusters: %d' % n_clusters_)
    print('Estimated number of origin clusters: %d' % origin_n_clusters_)

    plt.figure()
    plt.scatter(origin_new_dataset[:, 0], origin_new_dataset[:, 1], c=origin_new_labels, s=10, cmap='seismic')
    plt.title('origin cluster')
    plt.figure()
    plt.scatter(destination_new_dataset[:, 0], destination_new_dataset[:, 1], c=destination_new_labels, s=10, cmap='seismic')
    plt.title('destination cluster')
    plt.show()
    plt.show()


if __name__ == '__main__':
    start = time.clock()
    main()
    end = time.clock()
    print('finish all in %s' % str(end - start))