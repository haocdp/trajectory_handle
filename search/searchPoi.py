#！/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from urllib import request
import json
import math
import redis

url = "https://restapi.amap.com/v3/place/polygon?"
key = "dd414d3d690331b29f1b25aeebd7c4fd"
types = "010000|020000|030000|040000|050000|060000|070000|080000|090000|100000|110000|120000|130000|140000|150000|160000|170000|180000|190000|200000"
output = "json"
page = 1
offset = 20

def searchPOIByDistrict(
        longitude1, latitude1,
        longitude2, latitude2,
        longitude3, latitude3,
        longitude4, latitude4
):
    polygon = str(longitude1) + "," + str(latitude1) + "|" + str(longitude2) + "," + str(latitude2) + "|" + str(longitude3) \
              + "," + str(latitude3) + "|" + str(longitude4) + "," + str(latitude4)
    integrityUrl = url + "key=" + key + "&types=" + types + "&output=" + output + "&polygon=" + polygon + "&offset=" + str(offset)

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

def saveToRedis() :
    r = redis.Redis(host='127.0.0.1',port=6379)
    r.set('foo','foo')


def main(argv=None):
    if argv is None:
        argv = sys.argv

    # searchPOIByDistrict(114.052926, 22.539593, 114.061039, 22.540564, 114.060833, 22.534785, 114.053213, 22.534462)
    saveToRedis()

if __name__ == "__main__":
    sys.exit(main())
