# ！/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from urllib import request
import json
import math
import redis

url = "https://restapi.amap.com/v3/place/polygon?"
key = "dd414d3d690331b29f1b25aeebd7c4fd"
types = "010000|020000|030000|040000|050000|060000|070000|080000|090000|100000|110000|120000|130000|140000|150000" \
        "|160000|170000|180000|190000|200000"
output = "json"
extensions = "base"
page = 1
offset = 20


# 调用高德地图API获取区域内的POI
def searchPOIByDistrict(
        longitude1, latitude1,
        longitude2, latitude2,
        longitude3, latitude3,
        longitude4, latitude4
):
    polygon = str(longitude1) + "," + str(latitude1) + "|" + str(longitude2) + "," + str(latitude2) + "|" \
              + str(longitude3) + "," + str(latitude3) + "|" + str(longitude4) + "," + str(latitude4)
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
def saveToRedis():
    r = redis.Redis(host='127.0.0.1', port=6379, charset='utf-8')

    pois = \
        searchPOIByDistrict(114.052926, 22.539593, 114.061039, 22.540564, 114.060833, 22.534785, 114.053213, 22.534462)
    # print(pois)
    # print(json.dumps(pois, ensure_ascii=False))
    r.set('districtId', json.dumps(pois, ensure_ascii=False))
    # p = r.get("districtId")
    # a = p.decode('utf-8')
    # print(a)


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


def main(argv=None):
    if argv is None:
        argv = sys.argv

    # saveToRedis()
    # print(getPOIofDistrict())

    r = redis.Redis(host='127.0.0.1', port=6379, charset='utf-8')
    s = "粤"
    print(s)


if __name__ == "__main__":
    sys.exit(main())
