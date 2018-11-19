# ！/usr/bin/env python3
"""
本文件主要用于提取车辆的轨迹段
分为两类：寻客 和 载客
过滤规则：
    1、如果文件的第一行的轨迹点状态便是载客，则过滤点这一系列点
    2、如果文件的最后一行的轨迹点状态是载客，则过滤掉这些点

先不用过滤规则
"""
import sys
import os


def get_weekday_trajectory(path):
    save_path = "F:\FCD data\\trajectory\workday_trajectory"

    file_youke = open(save_path + "\youke_2011-7-5", "w")
    file_xunke = open(save_path + "\\xunke_2011-7-5", "w")

    dictionaries = os.listdir(path)
    count = 0
    files = os.listdir(path)
    for filename in files:
        file = open(path + "/" + filename)
        trajectory = []
        # pre_point_status = 0
        line = file.readline().strip('\n')
        if line == '':
            continue
        pre_point_status = line.split(";")[4]
        # while line.split(";")[4] == 1:
        #     line = file.readline()

        item = []
        while line:
            item = line.split(";")
            mod = count % 10
            if pre_point_status == item[4]:
                trajectory.append(item[1:4])
            else:
                if item[4] == '1':
                    file_xunke.write(str(trajectory))
                    file_xunke.write("\n")
                else:
                    file_youke.write(str(trajectory))
                    file_youke.write("\n")
                count = count + 1
                pre_point_status = item[4]
                trajectory.clear()
                trajectory.append(item[1:4])

            line = file.readline().strip('\n')

        if item[4] == '0':
            file_xunke.write(str(trajectory))
            file_xunke.write("\n")
        else:
            file_youke.write(str(trajectory))
            file_youke.write("\n")
        file.close()
    file_youke.flush()
    file_xunke.flush()
    file_youke.close()
    file_xunke.close()


def main(argv=None):
    if argv is None:
        argv = sys.argv
    weekday_path = "F:\FCD data\\trajectory\workday\\2011-7-5"
    # weekend_path = "F:\FCD data\\trajectory\weekend"
    get_weekday_trajectory(weekday_path)


if __name__ == "__main__":
    sys.exit(main())

