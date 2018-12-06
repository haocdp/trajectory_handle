# ！/usr/bin/env python3
import sys
from concurrent.futures import ThreadPoolExecutor
import os
import datetime
from trajectory.cal_distance import haversine
import redis
from shenzhen_map.save_region import is_point_in_polygon
import json
import time


# 从redis中读取城市区域划分
def get_region():
    r = redis.Redis(host='127.0.0.1', port=6379, charset='utf-8')
    taz_dict = {}
    for i in range(1, 1068):
        taz_id = "taz_" + str(i)
        polygon = str(r.get(taz_id), 'utf-8')
        taz_dict[taz_id] = list(json.loads(polygon))
    return taz_dict


taz_dict = get_region()


def divide_trajectory_by_car(file_path, file_name):
    file = open(file_path, 'r')
    line = file.readline()

    plateNumber_dict = {}
    start_time = time.time()
    while line:
        new_line = simplify_line(line)
        if not new_line == '':
            plateNumber = transform_line(line)
            if plateNumber in plateNumber_dict.keys():
                plateNumber_dict[plateNumber].append(new_line)
            else:
                plateNumber_dict[plateNumber] = []
                plateNumber_dict[plateNumber].append(new_line)

        # print(line)
        line = file.readline()

    end_time = time.time()
    print(end_time - start_time)

    filter_trajectory_point(plateNumber_dict)
    # 线程池
    pool = ThreadPoolExecutor(4)

    dictionary = file_name.split(".")[0][0:-5]
    dic_path = "F:/FCD data/trajectory_week_day/" + dictionary
    if not os.path.exists(dic_path):
        os.mkdir(dic_path)
    for key in plateNumber_dict.keys():
        pool.submit(write_file, key, plateNumber_dict, dic_path)

    # print(plateNumber_dict)
    file.close()


# 判断坐标点所在的位置
def get_point_region(point):
    # taz_dict = get_region()
    for key in taz_dict.keys():
        polygon = taz_dict.get(key)
        if is_point_in_polygon(point, polygon) is True:
            return str(key).split("_")[1]
    return -1


# 对车辆轨迹进行错误轨迹点过滤
def filter_trajectory_point(dict):
    pre_time = ''
    pre_longitude = 0
    pre_latitude = 0
    new_points = []
    for key in dict.keys():
        points = dict.get(key)
        for point in points:
            items = point.split(';')
            # if float(items[1]) > 114.627314 or float(items[1]) < 113.756360 \
                    # or float(items[2]) < 22.448471 or float(items[2]) > 22.856739:
                # continue
            if pre_time == '' or pre_longitude == 0 or pre_latitude == 0:
                pre_time = items[3]
                pre_longitude = float(items[1])
                pre_latitude = float(items[2])
                new_points.append(point)
            else:
                diff_time = get_time_diff(pre_time, items[3])
                diff_distance = get_distance_diff(pre_longitude, pre_latitude, float(items[1]), float(items[2]))
                if diff_time == 0 or diff_distance / diff_time > 30:
                    continue
                new_points.append(point)
                pre_time = items[3]
                pre_longitude = float(items[1])
                pre_latitude = float(items[2])

        dict[key] = new_points
        new_points = []


def get_distance_diff(pre_longitude, pre_latitude, longitude, latitude):
    return haversine(pre_longitude, pre_latitude, longitude, latitude)


# 时间差值（以秒为单位）
def get_time_diff(time_a, time_b):
    t_a = datetime.datetime.strptime(time_a, "%Y-%m-%d %H:%M:%S")
    t_b = datetime.datetime.strptime(time_b, "%Y-%m-%d %H:%M:%S")
    return (t_b - t_a).seconds


# 写入文件
def write_file(key, plateNumber_dict, path):
    if key == '\n' or key is '\n':
        return

    file = open(path + "/" + key + ".txt", 'w')
    for line in plateNumber_dict.get(key):
        file.write(line)
    file.close()


# 将文件行转换格式返回车牌号
def transform_line(line):
    plate = line.split(";")[0]
    if plate == '\n' or plate is None or not plate.__contains__(":"):
        return plate
    else:
        if len(plate.split(":")) > 0:
            return plate.split(":")[1]
        else:
            return plate


# 简化记录信息
def simplify_line(line):
    if line == '\n' or line is None:
        return ''

    records = line.split(";")
    if not len(records) == 11:
        return ''
    plate_no = records[0].split(':')[1] if len(records[0].split(':')) > 0 else ''
    longitude = records[1].split(':')[1] if len(records[0].split(':')) > 0 else ''
    latitude = records[2].split(':')[1] if len(records[0].split(':')) > 0 else ''
    t = records[3][3:] if len(records[0].split(':')) > 0 else ''
    status = records[8].split(':')[1] if len(records[0].split(':')) > 0 else ''

    # 判断区域
    region = get_point_region([float(longitude) / 1000000, float(latitude) / 1000000])
    if region == -1:
        return ''

    if plate_no == '' or longitude == '' or latitude == '' or time == '' or status == '':
        return ''
    new_line = plate_no + ";" + str(float(longitude) / 1000000) + ";" + str(float(latitude) / 1000000) + ";" + \
               t + ";" + status + ";" + region + "\n"
    # new_line = plate_no + ";" + str(float(longitude) / 1000000) + ";" + str(float(latitude) / 1000000) + ";" + \
    #            t + ";" + status + "\n"
    return new_line


# 批量处理文件夹中的所有数据
def batch_handle(path):
    # 线程池
    pool = ThreadPoolExecutor(4)

    files = os.listdir(path)
    for file in files:
        try:
            pool.submit(divide_trajectory_by_car(path + "/" + file, file))
            # divide_trajectory_by_car(path + "/" + file, file)
        except Exception as e:
            print("Exception occurred in " + file)
            print(e)


def main(argv=None):
    if argv is None:
        argv = sys.argv

    # batch_handle("F:/FCD data/FCD-7月")
    divide_trajectory_by_car("F:/FCD data/FCD-7月/2011-7-16log_1.txt", "2011-7-16log_1.txt")
    # divide_trajectory_by_car("F:/FCD data/新建文本文档.txt", "新建文本文档.txt")


if __name__ == "__main__":
    sys.exit(main())
