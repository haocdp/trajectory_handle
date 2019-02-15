linux_path = "/root/taxiData"
windows_path = "H:\TaxiData"
base_path = windows_path

def get_region_center():
    file_path = base_path + "/shenzhen_map/taz.txt"
    region_center_dict = {}

    file = open(file_path, 'r')
    lines = file.readlines()

    for line in lines:
        region, cord = line.split(" ")
        region = int(region.split("_")[-1])
        cord = cord.split(";")
        center_lng = (float(cord[0]) + float(cord[1])) / 2
        center_lat = (float(cord[2]) + float(cord[3])) / 2
        cord = [center_lng, center_lat]
        region_center_dict[region] = cord
    return region_center_dict