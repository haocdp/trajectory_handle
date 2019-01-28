"""
遍历所有轨迹数据得到 region_to_ix, car_to_ix, poi_to_ix
"""
import random
import numpy as np

linux_path = "/root/taxiData"
windows_path = "D:\haoc\data\TaxiData"
base_path = linux_path

filepath1 = base_path + "/trajectory_without_filter/2014-10-20/trajectory_2014-10-20_result_new_cluster.npy"
filepath2 = base_path + "/trajectory_without_filter/2014-10-21/trajectory_2014-10-21_result_new_cluster.npy"
filepath3 = base_path + "/trajectory_without_filter/2014-10-22/trajectory_2014-10-22_result_new_cluster.npy"
filepath4 = base_path + "/trajectory_without_filter/2014-10-23/trajectory_2014-10-23_result_new_cluster.npy"
filepath5 = base_path + "/trajectory_without_filter/2014-10-24/trajectory_2014-10-24_result_new_cluster.npy"
filepath6 = base_path + "/trajectory_without_filter/2014-10-25/trajectory_2014-10-25_result_new_cluster.npy"
filepath7 = base_path + "/trajectory_without_filter/2014-10-26/trajectory_2014-10-26_result_new_cluster.npy"

trajectories1 = list(np.load(filepath1))
trajectories2 = list(np.load(filepath2))
trajectories3 = list(np.load(filepath3))
trajectories4 = list(np.load(filepath4))
trajectories5 = list(np.load(filepath5))
trajectories6 = list(np.load(filepath6))
trajectories7 = list(np.load(filepath7))

all_trajectories = []
all_trajectories.extend(trajectories1)
all_trajectories.extend(trajectories2)
all_trajectories.extend(trajectories3)
all_trajectories.extend(trajectories4)
all_trajectories.extend(trajectories5)
all_trajectories.extend(trajectories6)
all_trajectories.extend(trajectories7)

# 打乱
random.shuffle(all_trajectories)

print("all trajectories num : {}".format(len(all_trajectories)))
count = len(all_trajectories) * 0.8

train_data = []
train_labels = []
test_data = []
test_labels = []
test_dest = []

car_to_ix = {}
poi_to_ix = {}
region_to_ix = {}
for trajectory, label, weekday, time_slot in all_trajectories:
    for t in trajectory:
        if t[0] not in car_to_ix:
            car_to_ix[t[0]] = len(car_to_ix)
        if t[-1] not in poi_to_ix:
            poi_to_ix[t[-1]] = len(poi_to_ix)
        if t[-2] not in region_to_ix:
            region_to_ix[t[-2]] = len(region_to_ix)

np.save("trajectory_without_filter_car_to_ix", car_to_ix)
np.save("trajectory_without_filter_poi_to_ix", poi_to_ix)
np.save("trajectory_without_filter_region_to_ix", region_to_ix)