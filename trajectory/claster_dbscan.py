# ！/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import math
import time
from sklearn.cluster import DBSCAN
from sklearn import metrics
from collections import Counter

UNCLASSIFIED = False
NOISE = 0


def loadDataSet(fileName, splitChar='\t'):
    """
    输入：文件名
    输出：数据集
    描述：从文件读入数据集
    """
    dataSet = []
    with open(fileName) as fr:
        for line in fr.readlines():
            curline = line.strip().split(splitChar)
            fltline = list(map(float, curline))
            dataSet.append(fltline)
    return dataSet


def dist(a, b):
    """
    输入：向量A, 向量B
    输出：两个向量的欧式距离
    """
    return math.sqrt(np.power(a - b, 2).sum())


def eps_neighbor(a, b, eps):
    """
    输入：向量A, 向量B
    输出：是否在eps范围内
    """
    return dist(a, b) < eps


def region_query(data, pointId, eps):
    """
    输入：数据集, 查询点id, 半径大小
    输出：在eps范围内的点的id
    """
    nPoints = data.shape[1]
    seeds = []
    for i in range(nPoints):
        if eps_neighbor(data[:, pointId], data[:, i], eps):
            seeds.append(i)
    return seeds


def expand_cluster(data, clusterResult, pointId, clusterId, eps, minPts):
    """
    输入：数据集, 分类结果, 待分类点id, 簇id, 半径大小, 最小点个数
    输出：能否成功分类
    """
    seeds = region_query(data, pointId, eps)
    if len(seeds) < minPts: # 不满足minPts条件的为噪声点
        clusterResult[pointId] = NOISE
        return False
    else:
        clusterResult[pointId] = clusterId # 划分到该簇
        for seedId in seeds:
            clusterResult[seedId] = clusterId

        while len(seeds) > 0: # 持续扩张
            currentPoint = seeds[0]
            queryResults = region_query(data, currentPoint, eps)
            if len(queryResults) >= minPts:
                for i in range(len(queryResults)):
                    resultPoint = queryResults[i]
                    if clusterResult[resultPoint] == UNCLASSIFIED:
                        seeds.append(resultPoint)
                        clusterResult[resultPoint] = clusterId
                    elif clusterResult[resultPoint] == NOISE:
                        clusterResult[resultPoint] = clusterId
            seeds = seeds[1:]
        return True


def dbscan(data, eps, minPts):
    """
    输入：数据集, 半径大小, 最小点个数
    输出：分类簇id
    """
    clusterId = 1
    nPoints = data.shape[1]
    clusterResult = [UNCLASSIFIED] * nPoints
    for pointId in range(nPoints):
        point = data[:, pointId]
        if clusterResult[pointId] == UNCLASSIFIED:
            if expand_cluster(data, clusterResult, pointId, clusterId, eps, minPts):
                clusterId = clusterId + 1
    return clusterResult, clusterId - 1


def plotFeature(data, clusters, clusterNum):
    nPoints = data.shape[1]
    matClusters = np.mat(clusters).transpose()
    fig = plt.figure()
    scatterColors = ['black', 'blue', 'green', 'yellow', 'red', 'purple', 'orange', 'brown']
    ax = fig.add_subplot(111)
    for i in range(clusterNum + 1):
        colorSytle = scatterColors[i % len(scatterColors)]
        subCluster = data[:, np.nonzero(matClusters[:, 0].A == i)]
        ax.scatter(subCluster[0, :].flatten().A[0], subCluster[1, :].flatten().A[0], c=colorSytle, s=50)


def load_dataset(filename):
    dataSet = []
    with open(filename) as fr:
        for line in fr.readlines():
            curline = line.strip().split(";")[:2]
            fltline = list(map(float, curline))
            dataSet.append(fltline)
    return dataSet


def main():
    # dataSet = load_dataset('F:\FCD data\\trajectory\workday_point\\cluster_test.txt')
    # dataSet = np.mat(dataSet).transpose()
    # print(dataSet)
    # clusters, clusterNum = dbscan(dataSet, 2, 15)
    # print("cluster Numbers = ", clusterNum)
    # print(clusters)
    # plotFeature(dataSet, clusters, clusterNum)

    dataset = load_dataset('F:\FCD data\\trajectory\workday_point\\youke_2011-7-4')
    dataset_array = np.array(dataset)
    db = DBSCAN(eps=0.0005, min_samples=5, metric='haversine').fit(dataset_array)

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    new_dataset = []
    new_labels = []
    list_labels = list(labels)
    dict_lables = Counter(list_labels)
    for key, value in enumerate(list_labels):
        if value == -1 or dict_lables[value] < 20:
            continue
        new_dataset.append(dataset[key])
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
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    print('Estimated number of clusters: %d' % n_clusters_)

    plt.scatter(new_dataset[:, 0], new_dataset[:, 1], c=new_labels, s=10, cmap='seismic')
    plt.show()


if __name__ == '__main__':
    start = time.clock()
    main()
    end = time.clock()
    print('finish all in %s' % str(end - start))