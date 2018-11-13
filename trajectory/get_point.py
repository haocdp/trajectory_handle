# ！/usr/bin/env python3
"""
本文件主要用于将轨迹中的点提取出来用于聚类
分为两类：工作日 和 周末
"""
import sys
import os


# 读取工作日文件夹中的所有文件，并根据寻客\载客状态把数据分别存储到文件中
def get_weekday_point(path):
    save_path = "F:\FCD data\\trajectory\workday_point"

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
        files = os.listdir(path + "/" + dictionary)
        for filename in files:
            file = open(path + "/" + dictionary + "/" + filename)
            line = file.readline().strip('\n')
            while line:
                item = line.split(";")
                mod = count % 10
                if item[4] == '0':
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
                    file_xunke.write(";".join(item[1:4]))
                    file_xunke.write('\n')
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
                    file_youke.write(";".join(item[1:4]))
                    file_youke.write('\n')
                line = file.readline().strip('\n')
                count = count + 1
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


def get_weekend_point():
    pass


def main(argv=None):
    if argv is None:
        argv = sys.argv
    weekday_path = "F:\FCD data\\trajectory\workday"
    # weekend_path = "F:\FCD data\\trajectory\weekend"
    get_weekday_point(weekday_path)


if __name__ == "__main__":
    sys.exit(main())