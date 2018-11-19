# ！/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import json
import time
from sklearn.cluster import DBSCAN
from sklearn import metrics
from collections import Counter
from ast import literal_eval


def load_dataset(filename):
    origin_dataSet = []
    destination_dataSet = []
    with open(filename) as fr:
        for line in fr.readlines():
            if line == '' or line == '\n':
                continue
            trajectory = literal_eval(line.strip('\n'))
            if len(trajectory) <= 10:
                continue
            origin_dataSet.append(list(map(float, trajectory[0][:2])))
            destination_dataSet.append(list(map(float, trajectory[len(trajectory) - 1][:2])))
    return origin_dataSet, destination_dataSet


def main():
    # dataSet = load_dataset('F:\FCD data\\trajectory\workday_point\\cluster_test.txt')
    # dataSet = np.mat(dataSet).transpose()
    # print(dataSet)
    # clusters, clusterNum = dbscan(dataSet, 2, 15)
    # print("cluster Numbers = ", clusterNum)
    # print(clusters)
    # plotFeature(dataSet, clusters, clusterNum)

    origin_dataset, destination_dataSet = load_dataset('F:\FCD data\\trajectory\workday_trajectory\\youke_0')
    origin_dataset_array = np.array(origin_dataset)
    destination_dataset_array = np.array(destination_dataSet)
    origin_db = DBSCAN(eps=0.0002, min_samples=20, metric='haversine').fit(origin_dataset_array)
    destination_db = DBSCAN(eps=0.0002, min_samples=20, metric='haversine').fit(destination_dataset_array)

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
    np.save("F:\FCD data\cluster\cluster_dataset", destination_new_dataset)
    np.save("F:\FCD data\cluster\destination_labels", destination_new_labels)


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