import numpy as np

linux_path = "/root/taxiData"
windows_path = "F:/FCD data"
base_path = linux_path

def get_cluster_center():
    cluster_dataset = list(np.load(base_path + "/cluster/cluster_dataset.npy"))
    labels = list(np.load(base_path + "/cluster/destination_labels.npy"))
    cluster_dict = {}
    for index, value in enumerate(labels):
        if value == -1:
            continue

        if not cluster_dict.keys().__contains__(value):
            cluster_dict[value] = []
        # if cluster_dict[value] is None:
        #     cluster_dict[value] = []
        cluster_dict[value].append(list(cluster_dataset[index]))

    cluster_center_dict = {}
    for key in cluster_dict.keys():
        points = cluster_dict.get(key)
        sum_lat = 0.
        sum_lon = 0.
        for point in points:
            sum_lat += point[1]
            sum_lon += point[0]

        cluster_center_dict[key] = [sum_lon / len(points), sum_lat / len(points)]
    return cluster_center_dict
