"""
根据空载出租车区域分布region_taxi_distribution、区域需求分布region_prediction_distribution、
载客出租车目的地分布youke_destination_distribution 对区域的需求供给进行整合：
需求分布 - 目的地分布 = 出租车的真实需求分布
空载分布 - 真实需求分布 = 区域供给分布
区域供给分布表示当前区域是需要出租车 还是供给出租车；正数表示该区域供给出租车，负数表示该区域需要出租车
最后形成区域供给分布模型
"""
import numpy as np

path = "K:/毕业论文/TaxiData/dispatcher"

# 区域需求分布
region_demand_distribution = np.load("../demand_prediction/region_prediction_distribution_without_filter_7pm.npy").item()
region_demand_nums = 0.
for key in region_demand_distribution.keys():
    region_demand_nums += region_demand_distribution[key]
print("region_demand_nums = {}".format(region_demand_nums))

# 区域空载出租车分布
region_taxi_distribution = np.load("../destination_prediction/with_time/region_taxi_distribution_7pm.npy").item()
region_taxi_nums = 0.
for key in region_taxi_distribution.keys():
    if region_taxi_distribution[key] < 10:
        continue
    region_taxi_nums += region_taxi_distribution[key]
print("region_taxi_nums = {}".format(region_taxi_nums))

# 载客出租车目的地分布
youke_destination_distribution = np.load("../destination_prediction/with_time/youke_destination_distribution_7pm.npy").item()
youke_destination_nums = 0.
for key in youke_destination_distribution.keys():
    youke_destination_nums += youke_destination_distribution[key]
print("youke_destination_nums = {}".format(youke_destination_nums))


demand_supply = {}
demand_supply_without_destination_prediction = {}
for region in range(0, 918):
    if region in region_demand_distribution:
        region_demand_num = region_demand_distribution[region]
    else:
        region_demand_num = 0

    if region in region_taxi_distribution:
        region_taxi_num = region_taxi_distribution[region]
    else:
        region_taxi_num = 0

    if region in youke_destination_distribution:
        youke_destination_num = youke_destination_distribution[region]
    else:
        youke_destination_num = 0

    demand_supply[region] = region_taxi_num - (region_demand_num - youke_destination_num)
    demand_supply_without_destination_prediction[region] = region_taxi_num - region_demand_num
np.save(path + "/demand_supply_7pm", demand_supply)
np.save(path + "/demand_supply_without_destination_prediction_7pm", demand_supply_without_destination_prediction)

"""
将区域需求供给数据进行压缩，过滤掉为0的区域
region_to_new_ix 区域ID映射到新dict的下标
new_ix_to_region 新dict的下标映射到区域ID
demand_supply_new_ix 新下标的区域需求供给
"""
region_to_new_ix = {}
new_ix_to_region = {}

demand_supply_new_ix = {}
i = 0
for key in demand_supply.keys():
    if not demand_supply[key] == 0:
        region_to_new_ix[key] = i
        new_ix_to_region[i] = key
        demand_supply_new_ix[i] = demand_supply[key]
        i += 1

np.save(path + "/region_to_new_ix_7pm", region_to_new_ix)
np.save(path + "/new_ix_to_region_7pm", new_ix_to_region)
np.save(path + "/demand_supply_new_ix_7pm", demand_supply_new_ix)

"""
不考虑目的地预测情况下的数据
"""
region_to_new_ix_without_destination_prediction = {}
new_ix_to_region_without_destination_prediction = {}
demand_supply_new_ix_without_destination_prediction = {}

i = 0
for key in demand_supply_without_destination_prediction.keys():
    if not demand_supply_without_destination_prediction[key] == 0:
        region_to_new_ix_without_destination_prediction[key] = i
        new_ix_to_region_without_destination_prediction[i] = key
        demand_supply_new_ix_without_destination_prediction[i] = demand_supply_without_destination_prediction[key]
        i += 1

np.save(path + "/region_to_new_ix_without_destination_prediction_7pm", region_to_new_ix_without_destination_prediction)
np.save(path + "/new_ix_to_region_without_destination_prediction_7pm", new_ix_to_region_without_destination_prediction)
np.save(path + "/demand_supply_new_ix_without_destination_prediction_7pm", demand_supply_new_ix_without_destination_prediction)


