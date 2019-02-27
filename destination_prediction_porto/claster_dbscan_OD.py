# ！/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import json
import time
from sklearn.cluster import DBSCAN
from sklearn import metrics
from collections import Counter
from ast import literal_eval
import csv

linux_path = "/root/TaxiData_Porto"
window_path = "K:/毕业论文/TaxiData_Porto"
file_path = linux_path

def load_dataset(filename):
    destination_dataSet = []

    file = csv.reader(open(file_path + "/train.csv", 'r'))

    flag = False
    count = 0
    for line in file:
        if not flag:
            flag = not flag
            continue
        if line[-2] == 'True':
            continue
        trajectory = literal_eval(line[-1])
        if len(trajectory) >= 10 and count % 5 == 0:
            destination_dataSet.append(trajectory[-1])
        count += 1

    return destination_dataSet


def main():
    destination_dataSet = load_dataset(file_path + '/train.csv')
    destination_dataset_array = np.array(destination_dataSet)
    destination_db = DBSCAN(eps=0.0002, min_samples=40, metric='haversine').fit(destination_dataset_array)

    destination_labels = destination_db.labels_

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
    np.save(file_path + "/cluster/cluster_dataset", destination_new_dataset)
    np.save(file_path + "/cluster/destination_labels", destination_new_labels)

    n_clusters_ = len(set(destination_labels)) - (1 if -1 in destination_labels else 0)

    print('Estimated number of destination clusters: %d' % n_clusters_)
    # import pylab
    # # plt.figure()
    # # plt.scatter(destination_new_dataset[:, 0], destination_new_dataset[:, 1], c=destination_new_labels, s=10, cmap='seismic')
    # # plt.title('destination cluster')
    # # plt.show()
    # pylab.scatter(destination_new_dataset[:, 0], destination_new_dataset[:, 1], c=destination_new_labels, s=10, cmap='seismic')
    # pylab.savefig('cluster', format='pdf')


if __name__ == '__main__':
    start = time.clock()
    main()
    end = time.clock()
    print('finish all in %s' % str(end - start))