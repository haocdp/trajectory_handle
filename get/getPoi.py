# ！/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从区域中所有的POI中抽取最能代表区域语义信息的POI
"""
import sys
import os
from ast import literal_eval
from collections import Counter
import numpy as np


def get_representative_poi(input_path, output_path):
    files = os.listdir(input_path)
    poi_dict = {}
    output_file = open(output_path + "\\poi_dict.txt", 'w')
    for file in files:
        if os.path.isdir(input_path + "\\" + file):
            continue

        district_id = int(str(file).split("_")[-1])
        f = open(input_path + "/" + file)
        pois = literal_eval(str(f.readline()))
        ps = []
        for poi in pois:
            ps.append(int(poi['typecode'][0:2]))

        if len(ps) == 0:
            continue
        most_poi = int(Counter(ps).most_common(1)[0][0])
        poi_dict[district_id] = most_poi
        print(str(district_id) + " --> " + str(most_poi))
        output_file.write(str(district_id) + ":" + str(most_poi) + "\n")
        f.close()
    output_file.close()
    np.save("F:\FCD data\shenzhen_map_poi\\taz_region_poi\poi_dict", poi_dict)


def run():
    get_representative_poi("F:\FCD data\shenzhen_map_poi", "F:\FCD data\shenzhen_map_poi\\taz_region_poi")


def main(argv=None):
    if argv is None:
        argv = sys.argv
    get_representative_poi("F:\FCD data\shenzhen_map_poi", "F:\FCD data\shenzhen_map_poi\\taz_region_poi")


if __name__ == "__main__":
    sys.exit(main())
