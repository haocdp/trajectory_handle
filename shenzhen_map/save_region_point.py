# ！/usr/bin/env python3
import sys
import json
from decimal import *
import numpy as np
import math

'''
使用点来表示地图，每个区域里有等距离的点
在对轨迹点进行区域判断时，是需要找对应的点，取出对应的区域即可
'''

linux_path = "/root/data"
windows_path = "F:/TaxiData"
base_path = windows_path


# 从geojson文件中获取grid_code 到 region id的映射并返回最小经纬度
def grid_code_to_region_id(file_path):
    file = open(file_path, 'r')
    taz_map = file.read()
    taz_map_json = json.loads(taz_map)
    grid_to_region = {}
    min_lng = Decimal('180.00000')
    min_lat = Decimal('90.00000')
    for feature in taz_map_json['features']:
        GRID_CODE = feature['properties']['GRID_CODE']
        if GRID_CODE not in grid_to_region:
            grid_to_region[GRID_CODE] = len(grid_to_region)

        lng = feature['geometry']['coordinates'][0]
        lat = feature['geometry']['coordinates'][1]
        if lng < min_lng:
            min_lng = lng
        if lat < min_lat:
            min_lat = lat
    file.close()
    return grid_to_region, min_lng, min_lat, taz_map_json


# 通过上面函数得到的映射dict，建立坐标点到区域的映射
def point_to_region_id(file_path):
    grid_to_region, min_lng, min_lat, taz_map_json = grid_code_to_region_id(file_path)

    point_to_region_id = {}
    for feature in taz_map_json['features']:
        GRID_CODE = feature['properties']['GRID_CODE']
        lng = feature['geometry']['coordinates'][0]
        lat = feature['geometry']['coordinates'][1]
        lng_ix = round(float(Decimal(str(lng)) - Decimal(str(min_lng))) / float(Decimal('0.00034')))
        lat_ix = round(float(Decimal(str(lat)) - Decimal(str(min_lat))) / float(Decimal('0.00034')))

        if lng_ix not in point_to_region_id:
            point_to_region_id[lng_ix] = {}
        if lat_ix not in point_to_region_id[lng_ix]:
            point_to_region_id[lng_ix][lat_ix] = grid_to_region[GRID_CODE]

    np.save(base_path + "/shenzhen_map/TAZ_Point_to_Region", point_to_region_id)
    np.save(base_path + "/shenzhen_map/TAZ_min_lng_lat", (min_lng, min_lat))


def main(argv=None):
    if argv is None:
        argv = sys.argv

    file_path = base_path + "/shenzhen_map/TAZ_Grid_Point.geojson"
    point_to_region_id(file_path)


if __name__ == "__main__":
    sys.exit(main())