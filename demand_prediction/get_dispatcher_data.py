"""" 从数据集中找到星期三晚上7点的需求数据 """
import numpy as np


weekday = 2
time_slot = 18
region = 238
file_path = "K:\毕业论文\TaxiData\demand\\net_data_11_05.npy"

dispatcher_data = []
demand_data = np.load(file_path).tolist()
for data in demand_data:
    if data[0] == weekday and data[2] == region:
        dispatcher_data.append(np.array(data, object))

np.save("K:\毕业论文\TaxiData\demand\dispatcher_data_11_05_region_238", dispatcher_data)



