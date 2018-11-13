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

    file_youke = open(save_path + "\youke_2011-7-4", "w")
    file_xunke = open(save_path + "\\xunke_2011-7-4", "w")

    count = 0
    files = os.listdir(path)
    for filename in files:
        if count > 3000:
            break
        file = open(path + "/" + filename)
        line = file.readline().strip('\n')
        while line:
            item = line.split(";")
            if item[4] == '0':
                file_xunke.write(";".join(item[1:4]))
                file_xunke.write('\n')
            else:
                file_youke.write(";".join(item[1:4]))
                file_youke.write('\n')
            line = file.readline().strip('\n')

        count = count + 1
        file.close()
    file_xunke.close()
    file_youke.close()


def get_weekend_point():
    pass


def main(argv=None):
    if argv is None:
        argv = sys.argv
    weekday_path = "F:\FCD data\\trajectory\workday\\2011-7-4"
    # weekend_path = "F:\FCD data\\trajectory\weekend"
    get_weekday_point(weekday_path)


if __name__ == "__main__":
    sys.exit(main())