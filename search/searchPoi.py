# ！/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from urllib import request
import json
import math
import redis
from concurrent.futures import ThreadPoolExecutor

url = "https://restapi.amap.com/v3/place/polygon?"
key = "dd414d3d690331b29f1b25aeebd7c4fd"
types = "010000|020000|030000|040000|050000|060000|070000|080000|090000|100000|110000|120000|130000|140000|150000" \
        "|160000|170000|180000|190000|200000"
output = "json"
extensions = "base"
page = 1
offset = 20

r = redis.Redis(host='127.0.0.1', port=6379, charset='utf-8')


# 调用高德地图API获取区域内的POI
def searchPOIByDistrict(polygon_list):
    length = len(polygon_list)
    if length <= 0:
        return ""
    polygon_str = [str(point[0]) + "," + str(point[1]) for point in polygon_list]
    polygon = "|".join(polygon_str)
    integrityUrl = url + "key=" + key + "&types=" + types + "&output=" + output + "&polygon=" + polygon \
                   + "&offset=" + str(offset) + "&extensions=" + extensions

    # 保存结果
    pois = []
    response = request.urlopen(integrityUrl + "&page=" + str(page))
    data = response.read().decode()
    json_data = json.loads(data)
    pois.extend(json_data['pois'])

    pageNo = math.ceil(float(json_data['count']) / offset)

    if pageNo > 1:
        for p in range(2, pageNo + 1):
            json_data = json.loads(request.urlopen(integrityUrl + "&page=" + str(p)).read().decode())
            pois.extend(json_data['pois'])

    return pois


# 调用searchPOIByDistrict获取区域所有POI并存入redis中
def saveToRedis(id, polygon):
    pois = searchPOIByDistrict(polygon)
    if pois == "":
        return
    # print(pois)
    # print(json.dumps(pois, ensure_ascii=False))
    r.set('taz_poi_' + str(id), json.dumps(pois, ensure_ascii=False))
    saveToFile(id, pois)
    # p = r.get("districtId")
    # a = p.decode('utf-8')


def saveToFile(id, pois):
    file = open("F:\FCD data\shenzhen_map_poi/taz_poi_" + str(id), 'w')
    file.write(str(pois))
    file.close()


# 根据区域ID获取最能表示区域语义的POI标号
def getPOIofDistrict(districtId='districtId'):
    r = redis.Redis(host='127.0.0.1', port=6379, charset='utf-8')
    pois = r.get(districtId)

    if pois is None:
        print("redis中没有该区域数据")
    pois = json.loads(pois)
    types = {}

    for poi in pois:
        type = poi['typecode'][0:2]
        types[type] = types[type] + 1 if type in types.keys() else 1
    return sorted(types.items(), key=lambda x: x[1], reverse=True)[0][0]



def saveToRedis_1(id):
    polygon = list(json.loads(str(r.get("taz_" + str(id)), 'utf-8')))
    polygon = [point for key, point in enumerate(polygon) if key % 2 == 0]
    saveToRedis(id, polygon)
    print(id)



def main(argv=None):
    if argv is None:
        argv = sys.argv

    # saveToRedis()
    # print(getPOIofDistrict())
    pool = ThreadPoolExecutor(8)
    for i in range(64, 1068):
        pool.submit(saveToRedis_1(i))


if __name__ == "__main__":
    sys.exit(main())
