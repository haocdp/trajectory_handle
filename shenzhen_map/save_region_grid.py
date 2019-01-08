# ！/usr/bin/env python3
import sys
import json
import redis

# 读取深圳市网格划分gson文件，然后储存到redis中，并存储本地文件中

linux_path = "/root/taxiData"
windows_path = "F:/TaxiData"
base_path = windows_path

r = redis.Redis(host='127.0.0.1', port=6379, charset='utf-8')
path = base_path + "/shenzhen_map/taz.txt"
taz_file = open(path, "w")


# 存储到redis中
def save_to_redis(taz_id, polygon):
    r.set(taz_id, polygon)


# 存储到本地文件中
def save_to_file(taz_id, polygon):
    line = taz_id + " " + polygon + "\n"
    taz_file.write(line)


# 处理文件
def handle_file(file_path):
    file = open(file_path, 'r')
    taz_map = file.read()
    taz_map_json = json.loads(taz_map)
    count = 0
    for feature in taz_map_json['features']:
        taz_id = 'taz_grid_' + str(count)
        polygon = str(feature['properties']['xmin']) + ";" + str(feature['properties']['xmax']) + ";" + str(feature['properties']['ymin']) + ";" + str(feature['properties']['ymax'])
        save_to_file(taz_id, polygon)
        save_to_redis(taz_id, polygon)
        count += 1
    taz_file.close()


# 判断点是否在区域中
def is_point_in_polygon(point, rangelist):
    # 判断是否在外包矩形内，如果不在，直接返回false
    xmin = float(rangelist[0])
    xmax = float(rangelist[1])
    ymin = float(rangelist[2])
    ymax = float(rangelist[3])

    if xmax > point[0] > xmin and ymax > point[1] > ymin:
        return True
    return False


def main(argv=None):
    if argv is None:
        argv = sys.argv

    file_path = base_path + "/shenzhen_map/TAZ_Grid_JSON.geojson"
    handle_file(file_path)
    # print(is_point_in_polygon([22.704432, 114.006278], [[22.707596, 113.884718],[22.682573, 113.891597],[22.667053, 113.895037],
    #                                       [22.665795, 113.937948], [22.683216, 113.973996],[22.708242, 113.961295],
    #                                       [22.729149, 113.943437], [22.655967, 113.899160],[22.707596, 113.884718]]))


if __name__ == "__main__":
    sys.exit(main())