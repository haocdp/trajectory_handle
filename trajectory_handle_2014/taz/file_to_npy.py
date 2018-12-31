import numpy as np
from ast import literal_eval

# file_path = "F:/TaxiData/trajectory_taz/2014-10-26/trajectory_2014-10-26_result"
linux_file_path = "/root/taxiData/trajectory/2014-10-20/trajectory_2014-10-20_result"
# save_path = "F:/TaxiData/trajectory_taz/2014-10-26/trajectory_2014-10-26result_npy"
linux_save_path = "/root/taxiData/trajectory/2014-10-20/trajectory_2014-10-20_result_npy"

file = open(linux_file_path, 'r')
lines = file.readlines()

trajectories = []
for line in lines:
    trajectory, cluster, weekday, time_slot = line.strip().split(";")
    trajectory = literal_eval(trajectory)
    trajectories.append((trajectory, int(cluster), int(weekday), int(time_slot)))
np.save(linux_save_path, trajectories)