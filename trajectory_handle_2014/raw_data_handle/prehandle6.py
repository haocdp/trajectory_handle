# ！/usr/bin/env python3
import sys
sys.path.append("/root/trajectory_handle/")

from concurrent.futures import ThreadPoolExecutor
import os
import datetime
from trajectory.cal_distance import haversine
import redis
from shenzhen_map.save_region import is_point_in_polygon
import json
import time
import numpy as np

linux_path = "/root"
windows_path = "F:"
base_path = windows_path


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
    global plate_number_dict
    files = os.listdir(file_path)
    plate_number_dict = {}

    all_count = len(files)
    count = 0

    for file in files:

        file = open(file_path + '/' + file, 'r')
        lines = file.readlines()
        for line in lines:
            # start_time = time.time()
            new_line, plate_number = simplify_line(line)
            if not new_line == '':
                if plate_number in plate_number_dict.keys():
                    plate_number_dict[plate_number].append(new_line)
                else:
                    plate_number_dict[plate_number] = []
                    plate_number_dict[plate_number].append(new_line)

            # print(line)
            # end_time = time.time()
            # print(end_time - start_time)
        count += 1
        print(float(count) / all_count)

    filter_trajectory_point(plate_number_dict)
    # 线程池
    pool = ThreadPoolExecutor(4)

    dic_path = base_path + "/taxiData/divide_by_taxi/" + file_name
    if not os.path.exists(dic_path):
        os.mkdir(dic_path)
    for key in plate_number_dict.keys():
        pool.submit(write_file, key, plate_number_dict, dic_path)

    np.save(base_path + "/taxiData/divide_by_taxi/car_trajectory" + "_" + file_name, plate_number_dict)
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
        return '', ''

    records = line.split(",")
    if not len(records) == 10 or not records[0] == '20141025':
        return '', ''
    plate_no = records[3]
    date = records[0][0:4] + '-' + records[0][4:6] + '-' + records[0][6:8]
    time = records[1].zfill(6)
    time = time[0:2] + ':' + time[2:4] + ':' + time[4:6]
    longitude = records[4]
    latitude = records[5]
    status = records[8]

    # 判断区域
    region = get_point_region([float(longitude), float(latitude)])
    if region == -1:
        return '', ''

    if plate_no == '' or longitude == '' or latitude == '' or time == '' or status == '':
        return '', ''
    new_line = plate_no + ";" + longitude + ";" + latitude + ";" + \
               date + " " + time + ";" + status + ";" + region + "\n"
    return new_line, plate_no


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

    divide_trajectory_by_car(base_path + "/taxiData/rawData/25", "2014-10-25")


if __name__ == "__main__":
    sys.exit(main())
