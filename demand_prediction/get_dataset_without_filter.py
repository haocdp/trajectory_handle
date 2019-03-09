import os
import sys
if sys.platform == 'linux':
    sys.path.append('/root/trajectory_handle/')
import numpy as np

linux_path = "/root/taxiData"
windows_path = "K:\毕业论文\TaxiData"
base_path = windows_path

seq_length = 6
conv_length = 7

'''
训练集和测试集的格式：
    [
        weekday, 
        time_slot, 
        region, 
        [
            (d(l - 6), conv(7 x 7)), 
            (d(l - 5), conv(7 x 7)), 
            (d(l - 4), conv(7 x 7)), 
            (d(l - 3), conv(7 x 7)),
            (d(l - 2), conv(7 x 7)),
            (d(l - 1), conv(7 x 7))
        ], 
        real_demand
    ]
'''
net_dataset = []

region_demand = np.load(base_path + "/demand_all/region_demand_without_filter.npy").tolist()
region_matrix = np.load(base_path + "/demand/region_matrix.npy").tolist()
region_to_ix = np.load(base_path + "/demand/region_to_ix.npy").item()

region_demand = list(map(list, region_demand))

for weekday, region_timeslot_demand in enumerate(region_demand):

    for i in range(48 - seq_length - 1 + 1):
        seq_set = []
        for j in range(i, i + seq_length + 1):
            seq_set.append(region_timeslot_demand[j])

        time_slot = i + seq_length
        for k in range(28):
            for l in range(60):
                if region_matrix[k][l] == -1:
                    continue
                seq_demand = []
                for demand in seq_set[:-1]:
                    conv_demand = []
                    for m in range(conv_length):
                        conv_demand.append([])
                        for q in range(conv_length):
                            conv_demand[m].append(0)
                    for n in range(-3, 4):
                        for p in range(-3, 4):
                            if k + n < 0 or k + n > 27 or l + p < 0 or l + p > 59:
                                conv_demand[n + 3][p + 3] = 0
                            else:
                                conv_demand[n + 3][p + 3] = demand[k + n][l + p]
                    seq_demand.append((demand[k][l], conv_demand))
                net_dataset.append(np.array((weekday, time_slot, region_matrix[k][l], seq_demand, seq_set[-1][k][l]),object))
np.save(base_path + "/demand_all/net_data_without_filter_1", net_dataset)


