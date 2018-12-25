import sys
if sys.platform == 'linux':
    sys.path.append('/root/trajectory_handle/')

import redis
from decimal import *
import numpy as np
from shenzhen_map.save_region_grid import main
# 从redis中读取城市网格，并将对应城市区域对应到28 x 60的矩阵中

linux_path = "/root/taxiData"
windows_path = "F:/TaxiData"
base_path = windows_path


def get_region_matrix():
    main()

    r = redis.Redis(host='127.0.0.1', port=6379, charset='utf-8')
    taz_grids = {}
    for i in range(918):
        taz_grid = str(r.get('taz_grid_' + str(i)), 'utf-8')
        xmin, xmax, ymin, ymax = list(map(Decimal, taz_grid.split(";")))
        taz_grids[i] = (xmin, xmax, ymin, ymax)

    # print(taz_grids)

    all_grid = []
    region_to_ix = {}
    for i in range(28):
        all_grid.append([])
        for j in range(60):
            all_grid[i].append(-1)

    xmin = Decimal('180.00000')
    ymin = Decimal('90.00000')
    for key in taz_grids.keys():
        region_xmin = taz_grids.get(key)[0]
        region_ymin = taz_grids.get(key)[2]
        if xmin > region_xmin:
            xmin = region_xmin
        if ymin > region_ymin:
            ymin = region_ymin

    for key in taz_grids.keys():
        region_xmin = taz_grids.get(key)[0]
        region_ymin = taz_grids.get(key)[2]
        i = int(float(region_xmin - xmin) / 0.015)
        j = 27 - int(float(region_ymin - ymin) / 0.015)
        # print("i = {}, j = {}".format(i, j))
        all_grid[j][i] = key
        region_to_ix[key] = (j, i)

    np.save(base_path + "/demand/region_matrix", all_grid)
    np.save(base_path + "/demand/region_to_ix", region_to_ix)
    # print(all_grid)
    return all_grid, region_to_ix
