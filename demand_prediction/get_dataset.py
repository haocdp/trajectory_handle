import os
import sys
if sys.platform == 'linux':
    sys.path.append('/root/trajectory_handle/')
import numpy as np

linux_path = "/root/taxiData"
windows_path = "F:/TaxiData"
base_path = linux_path

seq_length = 6

# 训练集和测试集的格式：(weekday, time_slot, region, [d(l - 6), d(l - 5) d(l - 4), d(l - 3), d(l - 2), d(l - 1)], conv(7 x 7), real_demand)
train_set = []
test_set = []

region_demand = list(np.load(base_path + "/demand/region_demand.npy"))
region_matrix = list(np.load(base_path + "/demand/region_matrix.npy"))
region_to_ix = np.load(base_path + "/demand/region_to_ix.npy").item()

for weekday, region_timeslot_demand in enumerate(region_demand):

    for i in range(48 - seq_length + 1 + 1):
        seq_set = []
        for j in range(i, seq_length + 1):
            seq_set.append(region_timeslot_demand[j])

        for k in range(28):
            for l in range(60):
                if region_matrix[28][60] == -1:
                    continue


