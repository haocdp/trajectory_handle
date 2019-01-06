# ！/usr/bin/env python3
"""
本文件主要用于提取车辆的轨迹段
分为两类：寻客 和 载客
过滤规则：
    1、如果文件的第一行的轨迹点状态便是载客，则过滤点这一系列点
    2、如果文件的最后一行的轨迹点状态是载客，则过滤掉这些点

2018/12/29:
    增加过滤轨迹功能，将轨迹中连续相同的轨迹点过滤点，并保持轨迹在10点加上最后一个轨迹点的长度
"""
import sys
import os
import numpy as np

windows_path = "F:/TaxiData"
linux_path = "/root/taxiData"
base_path = linux_path

file_dir = "2014-10-20"


# 过滤轨迹，如果轨迹存在连续相同区域，则进行过滤
def filter(tra):
    first_index = tra[0]
    new_tra = [tra[0]]
    for t in tra:
        if t[-1] == first_index[-1]:
            continue
        new_tra.append(t)
        first_index = t
    return new_tra


def get_weekday_trajectory(path):
    save_path = base_path + "/trajectory_without_filter/" + file_dir

    file_youke_0 = open(save_path + "/youke_0", "w")
    file_youke_1 = open(save_path + "/youke_1", "w")
    file_youke_2 = open(save_path + "/youke_2", "w")
    file_youke_3 = open(save_path + "/youke_3", "w")
    file_youke_4 = open(save_path + "/youke_4", "w")
    file_youke_5 = open(save_path + "/youke_5", "w")
    file_youke_6 = open(save_path + "/youke_6", "w")
    file_youke_7 = open(save_path + "/youke_7", "w")
    file_youke_8 = open(save_path + "/youke_8", "w")
    file_youke_9 = open(save_path + "/youke_9", "w")

    file_xunke_0 = open(save_path + "/xunke_0", "w")
    file_xunke_1 = open(save_path + "/xunke_1", "w")
    file_xunke_2 = open(save_path + "/xunke_2", "w")
    file_xunke_3 = open(save_path + "/xunke_3", "w")
    file_xunke_4 = open(save_path + "/xunke_4", "w")
    file_xunke_5 = open(save_path + "/xunke_5", "w")
    file_xunke_6 = open(save_path + "/xunke_6", "w")
    file_xunke_7 = open(save_path + "/xunke_7", "w")
    file_xunke_8 = open(save_path + "/xunke_8", "w")
    file_xunke_9 = open(save_path + "/xunke_9", "w")

    # dictionaries = os.listdir(path)
    file_xunke = file_xunke_0
    file_youke = file_youke_0
    count = 0

    trajectories = []
    # for dictionary in dictionaries:
    # if not os.path.isfile(path + "/" + dictionary):
    #     continue
    # car_dict = np.load(path + "/" + dictionary).item()
    car_dict = np.load(path).item()
    # filename = dictionary.split(".")[0]
    filename = file_dir

    for car_name in car_dict.keys():
        points = car_dict.get(car_name)
        if len(points) == 0:
            continue

        trajectory = []
        j = 0
        while j != len(points) and points[j].split(';')[4] == '1':
            j += 1
        pre_point_status = '0'
        item = []
        for i in range(j, len(points)):
            line = points[i].strip('\n')
            if line == '':
                continue
            # pre_point_status = line.split(";")[4]

            item = line.split(";")
            if not len(item) == 6:
                continue
            mod = count % 3
            if pre_point_status == item[4]:
                trajectory.append(item[0:6])
            else:
                if item[4] == '1':
                    if mod == 0:
                        file_xunke = file_xunke_0
                    elif mod == 1:
                        file_xunke = file_xunke_1
                    elif mod == 2:
                        file_xunke = file_xunke_2
                    elif mod == 3:
                        file_xunke = file_xunke_3
                    elif mod == 4:
                        file_xunke = file_xunke_4
                    elif mod == 5:
                        file_xunke = file_xunke_5
                    elif mod == 6:
                        file_xunke = file_xunke_6
                    elif mod == 7:
                        file_xunke = file_xunke_7
                    elif mod == 8:
                        file_xunke = file_xunke_8
                    elif mod == 9:
                        file_xunke = file_xunke_9
                    file_xunke.write(str(trajectory))
                    file_xunke.write("\n")
                else:
                    if mod == 0:
                        file_youke = file_youke_0
                    elif mod == 1:
                        file_youke = file_youke_1
                    elif mod == 2:
                        file_youke = file_youke_2
                    elif mod == 3:
                        file_youke = file_youke_3
                    elif mod == 4:
                        file_youke = file_youke_4
                    elif mod == 5:
                        file_youke = file_youke_5
                    elif mod == 6:
                        file_youke = file_youke_6
                    elif mod == 7:
                        file_youke = file_youke_7
                    elif mod == 8:
                        file_youke = file_youke_8
                    elif mod == 9:
                        file_youke = file_youke_9
                    # trajectory = filter(trajectory)
                    if len(trajectory) < 11:
                        continue
                    if len(trajectory) < 21:
                        new_tra = []
                        new_tra.extend(trajectory[:10])
                        new_tra.append(trajectory[-1])
                        trajectories.append(new_tra)
                        file_youke.write(str(new_tra))
                        file_youke.write("\n")
                    else:
                        new_tra = []
                        new_tra.extend(trajectory[:20])
                        new_tra.append(trajectory[-1])
                        trajectories.append(new_tra)
                        file_youke.write(str(new_tra))
                        file_youke.write("\n")
                count = count + 1
                pre_point_status = item[4]
                trajectory = []
                trajectory.append(item[0:6])

        if len(item) == 0:
            continue
        if item[4] == '0':
            file_xunke.write(str(trajectory))
            file_xunke.write("\n")

    file_youke_0.close()
    file_youke_1.close()
    file_youke_2.close()
    file_youke_3.close()
    file_youke_4.close()
    file_youke_5.close()
    file_youke_6.close()
    file_youke_7.close()
    file_youke_8.close()
    file_youke_9.close()

    file_xunke_0.close()
    file_xunke_1.close()
    file_xunke_2.close()
    file_xunke_3.close()
    file_xunke_4.close()
    file_xunke_5.close()
    file_xunke_6.close()
    file_xunke_7.close()
    file_xunke_8.close()
    file_xunke_9.close()
    np.save(save_path + "/trajectory_" + file_dir, trajectories)


def run():
    npy_path = base_path + "/divide_by_taxi/car_trajectory_" + file_dir + ".npy"
    # weekend_path = "F:\FCD data\\trajectory\weekend"
    get_weekday_trajectory(npy_path)


def main(argv=None):
    if argv is None:
        argv = sys.argv
    npy_path = base_path + "/divide_by_taxi/car_trajectory_" + file_dir + ".npy"
    # weekend_path = "F:\FCD data\\trajectory\weekend"
    get_weekday_trajectory(npy_path)


if __name__ == "__main__":
    sys.exit(main())

