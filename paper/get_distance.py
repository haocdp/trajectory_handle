# from trajectory.cal_distance import haversine

# print(haversine(113.72915, 22.44785, 113.72915, 22.86469))


import numpy as np
demand = np.load('K:\毕业论文\TaxiData\demand/region_demand_without_filter.npy').tolist()
for every_demand in demand:
    demands = []
    for every_time_demand in every_demand:
        demands.append(every_time_demand)

    diff_demand = [[0 for col in demands[34][0]] for row in demands[34]]
    for i, de in enumerate(demands[34]):
        print(de)
        for j, d in enumerate(de):
            diff_demand[i][j] = abs(demands[34][i][j] -
                                    sum(list(map(sum, [[0 if k < 0 or k > 27 or l < 0 or l > 59 else demands[34][k][l] for l in range(j - 1, j + 2)]for k in range(i - 1, i + 2)]))) / 9)
    print("")
    for d in diff_demand:
        print(d)
    print("")