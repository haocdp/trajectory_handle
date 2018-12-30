import numpy as np
from ast import literal_eval

file_path = "F:/TaxiData/trajectory_taz/2014-10-26/trajectory_2014-10-26_result"
save_path = "F:/TaxiData/trajectory_taz/2014-10-26/trajectory_2014-10-26result_npy"

file = open(file_path, 'r')
lines = file.readlines()

trajectories = []
for line in lines:
    trajectory, cluster, weekday, time_slot = line.strip().split(";")
    trajectory = literal_eval(trajectory)
    trajectories.append((trajectory, int(cluster), int(weekday), int(time_slot)))
np.save(save_path, trajectories)