import os
import sys
if sys.platform == 'linux':
    sys.path.append('/root/trajectory_handle/')
import numpy as np

linux_path = "/root/taxiData"
windows_path = "F:/TaxiData"
base_path = linux_path

seq_length = 6

# 训练集和测试集的格式：(weekday, time_slot, [d(l - 6), d(l - 5) d(l - 4), d(l - 3), d(l - 2), d(l - 1)], conv(7 x 7), real_demand)
train_set = []
test_set = []

region_demand = list(np.load(base_path + "/demand/region_demand.npy"))
print()
