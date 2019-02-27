# ！/usr/bin/env python3
import csv
import numpy as np

windows_path = "K:/毕业论文/TaxiData_Porto"
linux_path = "/root/TaxiData_Porto"
base_path = linux_path


trajectories = []
file = csv.reader(open(base_path + "/train.csv", 'r'))

all_count = 1710671

min_lng = 180.
max_lng = -180.
min_lat = 90.
max_lat = -90.
count = 0
flag = False
for line in file:
    count += 1
    if not flag:
        flag = True
        continue
    if line[-2] == 'True':
        continue

    trajectory = line[-1]
    for point in trajectory:
        lng = float(point[0])
        lat = float(point[1])
        if lng < min_lng:
            min_lng = lng
        if lng > max_lng:
            max_lng = lng
        if lat < min_lat:
            min_lat = lat
        if lat > max_lat:
            max_lat = lat

    if count % 1000 == 0:
        print("has finish: {} %".format(float(count) / all_count * 100))

np.save("city_range", (min_lng, max_lng, min_lat, max_lat))