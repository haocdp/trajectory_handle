import numpy as np

file_path1 = "D:\haoc\data\TaxiData\\trajectory_taz_without_filter/2014-10-20/trajectory_2014-10-20_result_npy.npy"
file_path2 = "D:\haoc\data\TaxiData\\trajectory_taz_without_filter/2014-10-21/trajectory_2014-10-21_result_npy.npy"
file_path3 = "D:\haoc\data\TaxiData\\trajectory_taz_without_filter/2014-10-22/trajectory_2014-10-22_result_npy.npy"
file_path4 = "D:\haoc\data\TaxiData\\trajectory_taz_without_filter/2014-10-23/trajectory_2014-10-23_result_npy.npy"
file_path5 = "D:\haoc\data\TaxiData\\trajectory_taz_without_filter/2014-10-24/trajectory_2014-10-24_result_npy.npy"
file_path6 = "D:\haoc\data\TaxiData\\trajectory_taz_without_filter/2014-10-25/trajectory_2014-10-25_result_npy.npy"
file_path7 = "D:\haoc\data\TaxiData\\trajectory_taz_without_filter/2014-10-26/trajectory_2014-10-26_result_npy.npy"

save_path1 = "D:\haoc\data\TaxiData\\trajectory_taz_without_filter/2014-10-20/trajectory_2014-10-20_result_npy_filter_by_time"
save_path2 = "D:\haoc\data\TaxiData\\trajectory_taz_without_filter/2014-10-21/trajectory_2014-10-21_result_npy_filter_by_time"
save_path3 = "D:\haoc\data\TaxiData\\trajectory_taz_without_filter/2014-10-22/trajectory_2014-10-22_result_npy_filter_by_time"
save_path4 = "D:\haoc\data\TaxiData\\trajectory_taz_without_filter/2014-10-23/trajectory_2014-10-23_result_npy_filter_by_time"
save_path5 = "D:\haoc\data\TaxiData\\trajectory_taz_without_filter/2014-10-24/trajectory_2014-10-24_result_npy_filter_by_time"
save_path6 = "D:\haoc\data\TaxiData\\trajectory_taz_without_filter/2014-10-25/trajectory_2014-10-25_result_npy_filter_by_time"
save_path7 = "D:\haoc\data\TaxiData\\trajectory_taz_without_filter/2014-10-26/trajectory_2014-10-26_result_npy_filter_by_time"

''' 1 '''
list1 = list(np.load(file_path1))
newtra1 = []
for trajectory, label, weekday, time_slot in list1:
    if time_slot > 1139:
        newtra1.append((trajectory, label, weekday, time_slot))
np.save(save_path1, newtra1)


''' 2 '''
list2 = list(np.load(file_path2))
newtra2 = []
for trajectory, label, weekday, time_slot in list2:
    if time_slot > 1139:
        newtra2.append((trajectory, label, weekday, time_slot))
np.save(save_path2, newtra2)


''' 3 '''
list3 = list(np.load(file_path3))
newtra3 = []
for trajectory, label, weekday, time_slot in list3:
    if time_slot > 1139:
        newtra3.append((trajectory, label, weekday, time_slot))
np.save(save_path3, newtra3)


''' 4 '''
list4 = list(np.load(file_path4))
newtra4 = []
for trajectory, label, weekday, time_slot in list4:
    if time_slot > 1139:
        newtra4.append((trajectory, label, weekday, time_slot))
np.save(save_path4, newtra4)


''' 5 '''
list5 = list(np.load(file_path5))
newtra5 = []
for trajectory, label, weekday, time_slot in list5:
    if time_slot > 1139:
        newtra5.append((trajectory, label, weekday, time_slot))
np.save(save_path5, newtra5)


''' 6 '''
list6 = list(np.load(file_path6))
newtra6 = []
for trajectory, label, weekday, time_slot in list6:
    if time_slot > 1139:
        newtra6.append((trajectory, label, weekday, time_slot))
np.save(save_path6, newtra6)


''' 7 '''
list7 = list(np.load(file_path7))
newtra7 = []
for trajectory, label, weekday, time_slot in list7:
    if time_slot > 1139:
        newtra7.append((trajectory, label, weekday, time_slot))
np.save(save_path7, newtra7)
