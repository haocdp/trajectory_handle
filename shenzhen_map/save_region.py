# ！/usr/bin/env python3
import sys
import json
import redis

# 读取深圳市交通小区划分gson文件，然后储存到redis中，并存储本地文件中

linux_path = "/root/data"
windows_path = "F:\FCD data"

r = redis.Redis(host='127.0.0.1', port=6379, charset='utf-8')
path = linux_path + "\shenzhen_map/taz.txt"
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
    for feature in taz_map_json['features']:
        taz_id = 'taz_' + feature['properties']['TAZ_ID']
        polygon = str(feature['geometry']['coordinates'][0])
        save_to_file(taz_id, polygon)
        save_to_redis(taz_id, polygon)
    taz_file.close()


# 判断点是否在区域中
def is_point_in_polygon(point, rangelist):
    # 判断是否在外包矩形内，如果不在，直接返回false
    lnglist = []
    latlist = []
    for i in range(len(rangelist)-1):
        lnglist.append(rangelist[i][0])
        latlist.append(rangelist[i][1])
    # print(lnglist, latlist)
    maxlng = max(lnglist)
    minlng = min(lnglist)
    maxlat = max(latlist)
    minlat = min(latlist)
    # print(maxlng, minlng, maxlat, minlat)
    if (point[0] > maxlng or point[0] < minlng or
        point[1] > maxlat or point[1] < minlat):
        return False
    count = 0
    point1 = rangelist[0]
    for i in range(1, len(rangelist)):
        point2 = rangelist[i]
        # 点与多边形顶点重合
        if (point[0] == point1[0] and point[1] == point1[1]) or (point[0] == point2[0] and point[1] == point2[1]):
            # print("在顶点上")
            return False
        # 判断线段两端点是否在射线两侧 不在肯定不相交 射线（-∞，lat）（lng,lat）
        if (point1[1] < point[1] <= point2[1]) or (point1[1] >= point[1] > point2[1]):
            # 求线段与射线交点 再和lat比较
            point12lng = point2[0] - (point2[1] - point[1]) * (point2[0] - point1[0])/(point2[1] - point1[1])
            # print(point12lng)
            # 点在多边形边上
            if point12lng == point[0]:
                # print("点在多边形边上")
                return False
            if point12lng < point[0]:
                count +=1
        point1 = point2
    # print(count)
    if count%2 == 0:
        return False
    else:
        return True




def main(argv=None):
    if argv is None:
        argv = sys.argv

    file_path = linux_path + "\shenzhen_map/taz.geojson"
    handle_file(file_path)
    # print(is_point_in_polygon([22.704432, 114.006278], [[22.707596, 113.884718],[22.682573, 113.891597],[22.667053, 113.895037],
    #                                       [22.665795, 113.937948], [22.683216, 113.973996],[22.708242, 113.961295],
    #                                       [22.729149, 113.943437], [22.655967, 113.899160],[22.707596, 113.884718]]))


if __name__ == "__main__":
    sys.exit(main())