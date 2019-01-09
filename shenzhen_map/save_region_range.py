import sys
import json
from decimal import *
import numpy as np
import math

'''
存储城市区域的最大最小经纬度
'''

linux_path = "/root/data"
windows_path = "F:/TaxiData"
base_path = windows_path

file_path = base_path + "/shenzhen_map/TAZ_Grid_Point.geojson"

file = open(file_path, 'r')
taz_map = file.read()
taz_map_json = json.loads(taz_map)
grid_to_region = {}
min_lng = Decimal('180.00000')
max_lng = Decimal('0.00000')
min_lat = Decimal('90.00000')
max_lat = Decimal('0.00000')
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
    if lng > max_lng:
        max_lng = lng
    if lat > max_lat:
        max_lat = lat
file.close()

np.save(base_path + "/demand/region_range", (min_lng, max_lng, min_lat, max_lat))