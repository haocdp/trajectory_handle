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
            if len(trajectory) <= 40:
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
    origin_db = DBSCAN(eps=0.001, min_samples=50, metric='haversine').fit(origin_dataset_array)
    destination_db = DBSCAN(eps=0.001, min_samples=20, metric='haversine').fit(destination_dataset_array)

    origin_labels = origin_db.labels_
    destination_labels = destination_db.labels_

    # new_dataset = []
    # new_labels = []
    # list_labels = list(origin_labels)
    # dict_lables = Counter(list_labels)
    # for key, value in enumerate(list_labels):
    #     if value == -1 or dict_lables[value] < 20:
    #         continue
    #     new_dataset.append(origin_dataset[key])
    #     new_labels.append(value)
    # new_dataset = np.array(new_dataset)
    # dict_lables = Counter(new_labels)
    # print(dict_lables)

    new_dataset = []
    new_labels = []
    list_labels = list(destination_labels)
    dict_lables = Counter(list_labels)
    for key, value in enumerate(list_labels):
        if value == -1 or dict_lables[value] < 20:
            continue
        new_dataset.append(destination_dataSet[key])
        new_labels.append(value)
    new_dataset = np.array(new_dataset)
    dict_lables = Counter(new_labels)
    print(dict_lables)



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

    print('Estimated number of clusters: %d' % n_clusters_)

    plt.scatter(new_dataset[:, 0], new_dataset[:, 1], c=new_labels, s=10, cmap='seismic')
    plt.show()


if __name__ == '__main__':
    start = time.clock()
    main()
    end = time.clock()
    print('finish all in %s' % str(end - start))