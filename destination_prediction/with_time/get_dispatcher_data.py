""" 获取星期三 晚上6点30分的空载出租车分布
以及所有6:00 ~ 6:30时间的载客出租车的轨迹数据 """
from datetime import datetime
from ast import literal_eval
import numpy as np

linux_path = "/root/taxiData"
windows_path = "H:/TaxiData"
base_path = windows_path + "/trajectory_without_filter/"

kong_start_time = datetime.strptime("2014-10-22 18:30:00", "%Y-%m-%d %H:%M:%S")
kong_end_time = datetime.strptime("2014-10-22 18:40:00", "%Y-%m-%d %H:%M:%S")

you_start_time = datetime.strptime("2014-10-22 18:00:00", "%Y-%m-%d %H:%M:%S")
you_end_time = datetime.strptime("2014-10-22 18:30:00", "%Y-%m-%d %H:%M:%S")

file_path = "2014-10-22"
# xunke_0 = base_path + file_path + "/xunke_0"
# xunke_1 = base_path + file_path + "/xunke_1"
# xunke_2 = base_path + file_path + "/xunke_2"
# xunke = [xunke_0, xunke_1, xunke_2]
#
# region_kong_taxi_distribution = {}
# # 记录是否已经存在这辆车，防止重复记录
# has_this_car = {}
# for xunke_file in xunke:
#     with open(xunke_file) as fr:
#         for line in fr.readlines():
#             if line == '' or line == '\n':
#                 continue
#             trajectory = literal_eval(line.strip('\n'))
#
#             start_time = datetime.strptime(str(trajectory[0][3]), "%Y-%m-%d %H:%M:%S")
#             end_time = datetime.strptime(str(trajectory[-1][3]), "%Y-%m-%d %H:%M:%S")
#             if start_time < kong_start_time < end_time and trajectory[0][0] not in has_this_car:
#
#                 max_time = 3600
#                 index = 0
#                 for i, tra in enumerate(trajectory):
#                     if kong_start_time > datetime.strptime(str(tra[3]), "%Y-%m-%d %H:%M:%S"):
#                         if abs((kong_start_time - datetime.strptime(str(tra[3]), "%Y-%m-%d %H:%M:%S")).seconds) < max_time:
#                             max_time = (kong_start_time - datetime.strptime(str(tra[3]), "%Y-%m-%d %H:%M:%S")).seconds
#                             index = i
#                     else:
#                         if abs((datetime.strptime(str(tra[3]), "%Y-%m-%d %H:%M:%S") - kong_start_time).seconds) < max_time:
#                             max_time = (datetime.strptime(str(tra[3]), "%Y-%m-%d %H:%M:%S") - kong_start_time).seconds
#                             index = i
#                 if int(trajectory[index][-1]) not in region_kong_taxi_distribution:
#                     region_kong_taxi_distribution[int(trajectory[index][-1])] = 1
#                 else:
#                     region_kong_taxi_distribution[int(trajectory[index][-1])] = region_kong_taxi_distribution[int(trajectory[index][-1])] + 1
#                 # region_kong_taxi_distribution[int(trajectory[index][-1])] = 1 \
#                 #     if int(trajectory[index][-1]) not in region_kong_taxi_distribution \
#                 #     else region_kong_taxi_distribution[int(trajectory[index][-1])] + 1
#                 has_this_car[trajectory[index][0]] = 1
#
# np.save("region_taxi_distribution", region_kong_taxi_distribution)

dispatcher_trajectory = []
youke = base_path + file_path + "/trajectory_2014-10-22result_npy.npy"
youke_trajectories = np.load(youke).tolist()
for trajectory, label, weekday, time_slot in youke_trajectories:
    if you_start_time < datetime.strptime(trajectory[0][3], "%Y-%m-%d %H:%M:%S") < you_end_time < datetime.strptime(trajectory[-1][3], "%Y-%m-%d %H:%M:%S"):
        dispatcher_trajectory.append((trajectory, label, weekday, time_slot))

np.save("youke_trajectories_data", dispatcher_trajectory)
