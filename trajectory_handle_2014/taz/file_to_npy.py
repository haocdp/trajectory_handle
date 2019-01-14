import numpy as np
from ast import literal_eval

linux_file_path1 = "D:\haoc\data\TaxiData/trajectory_wrong/2014-10-20/trajectory_2014-10-20_result"
linux_file_path2 = "D:\haoc\data\TaxiData/trajectory_wrong/2014-10-21/trajectory_2014-10-21_result"
linux_file_path3 = "D:\haoc\data\TaxiData/trajectory_wrong/2014-10-22/trajectory_2014-10-22_result"
linux_file_path4 = "D:\haoc\data\TaxiData/trajectory_wrong/2014-10-23/trajectory_2014-10-23_result"
linux_file_path5 = "D:\haoc\data\TaxiData/trajectory_wrong/2014-10-24/trajectory_2014-10-24_result"
linux_file_path6 = "D:\haoc\data\TaxiData/trajectory_wrong/2014-10-25/trajectory_2014-10-25_result"
linux_file_path7 = "D:\haoc\data\TaxiData/trajectory_wrong/2014-10-26/trajectory_2014-10-26_result"

linux_save_path1 = "E:/haoc/trajectory_wrong/2014-10-20/trajectory_2014-10-20_result_npy"
linux_save_path2 = "E:/haoc/trajectory_wrong/2014-10-21/trajectory_2014-10-21_result_npy"
linux_save_path3 = "E:/haoc/trajectory_wrong/2014-10-22/trajectory_2014-10-22_result_npy"
linux_save_path4 = "E:/haoc/trajectory_wrong/2014-10-23/trajectory_2014-10-23_result_npy"
linux_save_path5 = "E:/haoc/trajectory_wrong/2014-10-24/trajectory_2014-10-24_result_npy"
linux_save_path6 = "E:/haoc/trajectory_wrong/2014-10-25/trajectory_2014-10-25_result_npy"
linux_save_path7 = "E:/haoc/trajectory_wrong/2014-10-26/trajectory_2014-10-26_result_npy"

file = open(linux_file_path1, 'r')
lines = file.readlines()

trajectories = []
for line in lines:
    trajectory, cluster, weekday, time_slot = line.strip().split(";")
    trajectory = literal_eval(trajectory)
    trajectories.append((trajectory, int(cluster), int(weekday), int(time_slot)))
np.save(linux_save_path1, trajectories)

file = open(linux_file_path2, 'r')
lines = file.readlines()

trajectories = []
for line in lines:
    trajectory, cluster, weekday, time_slot = line.strip().split(";")
    trajectory = literal_eval(trajectory)
    trajectories.append((trajectory, int(cluster), int(weekday), int(time_slot)))
np.save(linux_save_path2, trajectories)

file = open(linux_file_path3, 'r')
lines = file.readlines()

trajectories = []
for line in lines:
    trajectory, cluster, weekday, time_slot = line.strip().split(";")
    trajectory = literal_eval(trajectory)
    trajectories.append((trajectory, int(cluster), int(weekday), int(time_slot)))
np.save(linux_save_path3, trajectories)


file = open(linux_file_path4, 'r')
lines = file.readlines()

trajectories = []
for line in lines:
    trajectory, cluster, weekday, time_slot = line.strip().split(";")
    trajectory = literal_eval(trajectory)
    trajectories.append((trajectory, int(cluster), int(weekday), int(time_slot)))
np.save(linux_save_path4, trajectories)

file = open(linux_file_path5, 'r')
lines = file.readlines()

trajectories = []
for line in lines:
    trajectory, cluster, weekday, time_slot = line.strip().split(";")
    trajectory = literal_eval(trajectory)
    trajectories.append((trajectory, int(cluster), int(weekday), int(time_slot)))
np.save(linux_save_path5, trajectories)

file = open(linux_file_path6, 'r')
lines = file.readlines()

trajectories = []
for line in lines:
    trajectory, cluster, weekday, time_slot = line.strip().split(";")
    trajectory = literal_eval(trajectory)
    trajectories.append((trajectory, int(cluster), int(weekday), int(time_slot)))
np.save(linux_save_path6, trajectories)

file = open(linux_file_path7, 'r')
lines = file.readlines()

trajectories = []
for line in lines:
    trajectory, cluster, weekday, time_slot = line.strip().split(";")
    trajectory = literal_eval(trajectory)
    trajectories.append((trajectory, int(cluster), int(weekday), int(time_slot)))
np.save(linux_save_path7, trajectories)
