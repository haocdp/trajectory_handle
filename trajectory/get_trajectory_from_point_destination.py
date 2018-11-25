# ！/usr/bin/env python3
"""
本文件主要用于提取车辆的轨迹段
分为两类：寻客 和 载客
过滤规则：
    1、如果文件的第一行的轨迹点状态便是载客，则过滤点这一系列点
    2、如果文件的最后一行的轨迹点状态是载客，则过滤掉这些点
"""
import sys
import os


def get_weekday_trajectory(path):
    save_path = "F:\FCD data\\trajectory\workday_trajectory_destination"

    file_youke_0 = open(save_path + "\youke_0", "w")
    file_youke_1 = open(save_path + "\youke_1", "w")
    file_youke_2 = open(save_path + "\youke_2", "w")
    file_youke_3 = open(save_path + "\youke_3", "w")
    file_youke_4 = open(save_path + "\youke_4", "w")
    file_youke_5 = open(save_path + "\youke_5", "w")
    file_youke_6 = open(save_path + "\youke_6", "w")
    file_youke_7 = open(save_path + "\youke_7", "w")
    file_youke_8 = open(save_path + "\youke_8", "w")
    file_youke_9 = open(save_path + "\youke_9", "w")

    file_xunke_0 = open(save_path + "\\xunke_0", "w")
    file_xunke_1 = open(save_path + "\\xunke_1", "w")
    file_xunke_2 = open(save_path + "\\xunke_2", "w")
    file_xunke_3 = open(save_path + "\\xunke_3", "w")
    file_xunke_4 = open(save_path + "\\xunke_4", "w")
    file_xunke_5 = open(save_path + "\\xunke_5", "w")
    file_xunke_6 = open(save_path + "\\xunke_6", "w")
    file_xunke_7 = open(save_path + "\\xunke_7", "w")
    file_xunke_8 = open(save_path + "\\xunke_8", "w")
    file_xunke_9 = open(save_path + "\\xunke_9", "w")

    dictionaries = os.listdir(path)
    file_xunke = file_xunke_0
    file_youke = file_youke_0
    count = 0
    for dictionary in dictionaries:
        if os.path.isfile(dictionary):
            continue
        files = os.listdir(path + "/" + dictionary)
        for filename in files:
            file = open(path + "/" + dictionary + "/" + filename)
            trajectory = []
            pre_point_status = 0
            line = file.readline().strip('\n')
            if line == '':
                continue
            # pre_point_status = line.split(";")[4]
            while line.split(";")[4] == 1:
                line = file.readline()

            item = []
            while line:
                item = line.split(";")
                mod = count % 3
                if pre_point_status == item[4]:
                    trajectory.append(item[1:6])
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
                        file_youke.write(str(trajectory))
                        file_youke.write("\n")
                    count = count + 1
                    pre_point_status = item[4]
                    trajectory.clear()
                    trajectory.append(item[1:6])

                line = file.readline().strip('\n')

            if item[4] == '0':
                file_xunke.write(str(trajectory))
                file_xunke.write("\n")
            # else:
            #     file_youke.write(str(trajectory))
            #     file_youke.write("\n")
            file.close()
        file_youke_0.flush()
        file_youke_1.flush()
        file_youke_2.flush()
        file_youke_3.flush()
        file_youke_4.flush()
        file_youke_5.flush()
        file_youke_6.flush()
        file_youke_7.flush()
        file_youke_8.flush()
        file_youke_9.flush()

        file_xunke_0.flush()
        file_xunke_1.flush()
        file_xunke_2.flush()
        file_xunke_3.flush()
        file_xunke_4.flush()
        file_xunke_5.flush()
        file_xunke_6.flush()
        file_xunke_7.flush()
        file_xunke_8.flush()
        file_xunke_9.flush()
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


def main(argv=None):
    if argv is None:
        argv = sys.argv
    weekday_path = "F:\FCD data\\trajectory_week_day\\weekday"
    # weekend_path = "F:\FCD data\\trajectory\weekend"
    get_weekday_trajectory(weekday_path)


if __name__ == "__main__":
    sys.exit(main())

