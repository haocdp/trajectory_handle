import numpy as np
from destination_prediction.evaluation import get_cluster_center

origin_cords = []
real_dest_cords = []
youke_trajectories_data_9am = np.load("youke_trajectories_data_9am.npy").tolist()
for trajectory, label, weekday, time_slot in youke_trajectories_data_9am:
    origin_cords.append([float(trajectory[0][1]), float(trajectory[0][2])])
    real_dest_cords.append([float(trajectory[-1][1]), float(trajectory[-1][2])])

dest_cords = []
dest_label = np.load("dest_label.npy").tolist()
cluster_center = get_cluster_center.get_cluster_center()
for label in dest_label:
    dest_cords.append(cluster_center[label])

file = open("origin_to_dest.txt", 'w')
for ix, origin_cord in enumerate(origin_cords):
    file.write(str(origin_cord[0]) + "," + str(origin_cord[1]) + ","
               + str(dest_cords[ix][0]) + "," + str(dest_cords[ix][1])
               + "," + str(real_dest_cords[ix][0]) + "," + str(real_dest_cords[ix][1]))
    file.write('\n')
file.close()
