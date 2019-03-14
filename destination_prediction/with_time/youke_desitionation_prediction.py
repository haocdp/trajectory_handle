"""
加载pkl文件，对youke_destination_prediction进行目的地预测，根据预测结果得到分布情况
"""

import sys

if sys.platform == 'linux':
    sys.path.append('/root/trajectory_handle/')
import os
import torch
from torch import nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as Data
import torch.nn.functional as F
import random
from destination_prediction.with_time import GeoConv
import logger
from destination_prediction.evaluation import get_cluster_center
from datetime import datetime
from datetime import timedelta
import time

# torch.manual_seed(1)    # reproducible
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # gpu
gpu_avaliable = torch.cuda.is_available()

# Hyper Parameters
EPOCH = 10  # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 64
TIME_STEP = 8  # rnn time step / image height
INPUT_SIZE = 59  # rnn input size / image width
HIDDEN_SIZE = 256
LR = 0.0001  # learning rate
LAYER_NUM = 2

linux_path = "/root/taxiData"
windows_path = "K:\毕业论文\TaxiData"
base_path = windows_path

labels = list(np.load(base_path + "/cluster/destination_labels_new.npy"))
# label个数
label_size = len(set(labels))
elogger = logger.Logger("youke_destination_prediction")

min_lng, max_lng, min_lat, max_lat = list(np.load(base_path + "/demand/region_range.npy"))
dis_lng = max_lng - min_lng
dis_lat = max_lat - min_lat


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(  # if use nn.RNN(), it hardly learns
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,  # rnn hidden unit
            num_layers=LAYER_NUM,  # number of rnn layer
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.out = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 512),
            nn.Linear(512, 1024),
            nn.Linear(1024, label_size)
        )
        self.car_embeds = nn.Embedding(11926, 16)
        self.poi_embeds = nn.Embedding(18, 4)
        self.region_embeds = nn.Embedding(758, 8)
        self.week_embeds = nn.Embedding(7, 3)
        self.time_embeds = nn.Embedding(1440, 8)

        self.region_poi_linear = nn.Linear(12, 32)
        self.coord_linear = nn.Linear(2, 8)
        self.conv = GeoConv.Net()

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        new_vector = None
        embedding_vector = None
        for vector in x:
            for item in vector:
                if new_vector is None:
                    if gpu_avaliable:
                        new_vector = torch.cat((self.region_embeds(torch.cuda.LongTensor([item[1].item()]))[0],
                                                self.poi_embeds(torch.cuda.LongTensor([item[2].item()]))[0]))
                        new_vector = torch.cat((new_vector, torch.cuda.FloatTensor([item[-3]])))
                        new_vector = torch.cat((new_vector, torch.cuda.FloatTensor([item[-2]])))
                    else:
                        new_vector = torch.cat((self.region_embeds(torch.LongTensor([item[1].item()]))[0],
                                                self.poi_embeds(torch.LongTensor([item[2].item()]))[0]))
                        new_vector = torch.cat((new_vector, torch.FloatTensor([item[-3]])))
                        new_vector = torch.cat((new_vector, torch.FloatTensor([item[-2]])))
                else:
                    if gpu_avaliable:
                        new_vector = torch.cat(
                            (new_vector, self.region_embeds(torch.cuda.LongTensor([item[1].item()]))[0]))
                        new_vector = torch.cat(
                            (new_vector, self.poi_embeds(torch.cuda.LongTensor([item[2].item()]))[0]))
                        new_vector = torch.cat((new_vector, torch.cuda.FloatTensor([item[-3]])))
                        new_vector = torch.cat((new_vector, torch.cuda.FloatTensor([item[-2]])))
                    else:
                        new_vector = torch.cat((new_vector, self.region_embeds(torch.LongTensor([item[1].item()]))[0]))
                        new_vector = torch.cat((new_vector, self.poi_embeds(torch.LongTensor([item[2].item()]))[0]))
                        new_vector = torch.cat((new_vector, torch.FloatTensor([item[-3]])))
                        new_vector = torch.cat((new_vector, torch.FloatTensor([item[-2]])))

            if embedding_vector is None:
                if gpu_avaliable:
                    embedding_vector = torch.cat((self.car_embeds(torch.cuda.LongTensor([vector[0][0].item()]))[0],
                                                  self.week_embeds(torch.cuda.LongTensor([vector[0][3].item()]))[0]))
                    embedding_vector = torch.cat(
                        (embedding_vector, self.time_embeds(torch.cuda.LongTensor([vector[0][4].item()]))[0]))
                else:
                    embedding_vector = torch.cat((self.car_embeds(torch.LongTensor([vector[0][0].item()]))[0],
                                                  self.week_embeds(torch.LongTensor([vector[0][3].item()]))[0]))
                    embedding_vector = torch.cat(
                        (embedding_vector, self.time_embeds(torch.LongTensor([vector[0][4].item()]))[0]))
            else:
                if gpu_avaliable:
                    embedding_vector = torch.cat(
                        (embedding_vector, self.car_embeds(torch.cuda.LongTensor([vector[0][0].item()]))[0]))
                    embedding_vector = torch.cat(
                        (embedding_vector, self.week_embeds(torch.cuda.LongTensor([vector[0][3].item()]))[0]))
                    embedding_vector = torch.cat(
                        (embedding_vector, self.time_embeds(torch.cuda.LongTensor([vector[0][4].item()]))[0]))
                else:
                    embedding_vector = torch.cat(
                        (embedding_vector, self.car_embeds(torch.LongTensor([vector[0][0].item()]))[0]))
                    embedding_vector = torch.cat(
                        (embedding_vector, self.week_embeds(torch.LongTensor([vector[0][3].item()]))[0]))
                    embedding_vector = torch.cat(
                        (embedding_vector, self.time_embeds(torch.LongTensor([vector[0][4].item()]))[0]))

        new_vector = new_vector.view(-1, 10, 14)
        new_vector = torch.cat(
            (self.region_poi_linear(new_vector[:, :, :-2]), self.coord_linear(new_vector[:, :, -2:])), 2)
        conv_vector = self.conv(new_vector)

        embedding_vector = F.tanh(embedding_vector)
        embedding_vector = embedding_vector.view(-1, 27)
        embedding_vector = torch.unsqueeze(embedding_vector, dim=1)
        expand_embedding_vector = embedding_vector.expand(conv_vector.size()[:2] + (embedding_vector.size()[-1],))
        x = torch.cat((conv_vector, expand_embedding_vector), dim=2)

        if gpu_avaliable:
            x = x.cuda()
        r_out, (h_n, h_c) = self.rnn(x, None)  # None represents zero initial hidden state

        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        del x, r_out, h_c, h_n
        # out = F.softmax(out, 1)
        return out


model = RNN()
model.load_state_dict(torch.load('cnn_lstm_prediction_new_cluster.pkl', map_location=lambda storage, loc: storage))

youke_trajectories = np.load("youke_trajectories_data_9am.npy").tolist()
car_to_ix = np.load("trajectory_without_filter_car_to_ix.npy").item()
poi_to_ix = np.load("trajectory_without_filter_poi_to_ix.npy").item()
region_to_ix = np.load("trajectory_without_filter_region_to_ix.npy").item()
region_transition_time = np.load("region_transition_time.npy").tolist()
region_to_matrix = np.load(base_path + "/demand/region_to_ix.npy").item()

pred_destination = []


def transfer(tra, weekday, time_slot):
    new_tra = []
    for t in tra:
        new_t = []
        new_t.append(car_to_ix[t[0]])
        new_t.append(region_to_ix[t[5]])
        new_t.append(poi_to_ix[t[6]])
        new_t.append(weekday)
        new_t.append(time_slot)
        new_t.append((float(t[1]) - min_lng) / dis_lng)
        new_t.append((float(t[2]) - min_lat) / dis_lat)
        new_t.append(int(t[5]))
        new_tra.append(new_t)
    return new_tra


# 每个聚类结果的
cluter_center_dict = get_cluster_center.get_cluster_center()


def get_region_range_dict():
    file_path = base_path + "/shenzhen_map/taz.txt"
    region_center_dict = {}

    file = open(file_path, 'r')
    lines = file.readlines()

    for line in lines:
        region, cord = line.split(" ")
        region = int(region.split("_")[-1])
        cord = cord.split(";")
        region_center_dict[region] = [float(cord[0]), float(cord[1]), float(cord[2]), float(cord[3])]
    return region_center_dict


# 获取每个区域的范围
region_range_dict = get_region_range_dict()


# 将聚类标签转换为区域ID
def label_to_region(label):
    label_point = cluter_center_dict[label]
    for key in region_range_dict.keys():
        cord = region_range_dict[key]
        if cord[0] < label_point[0] < cord[1] and cord[2] < label_point[1] < cord[3]:
            return key
    return -1


youke_test_data = []
youke_test_labels = []
youke_test_first_region = []
youke_test_first_time = []
for trajectory, label, weekday, time_slot in youke_trajectories:
    new_tra = transfer(trajectory, weekday, time_slot)
    # new_tra = filter(new_tra)
    youke_test_data.append(new_tra[:10])
    youke_test_labels.append(label)
    youke_test_first_region.append(new_tra[0][-1])
    youke_test_first_time.append(time.mktime(time.strptime(trajectory[0][3], "%Y-%m-%d %H:%M:%S")))

test_data = torch.FloatTensor(youke_test_data)
test_labels = torch.LongTensor(youke_test_labels)
test_first_region = torch.LongTensor(youke_test_first_region)
test_first_time = torch.FloatTensor(youke_test_first_time)

test_dataset = Data.TensorDataset(test_data, test_labels, test_first_region, test_first_time)
test_loader = Data.DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

"""
判断载客轨迹的到达目的地时间是否落在6：30~7：00之间，如果在则加入数组，否则剔除
"""
all_pred_y = []
all_pred_time = []
dest_label = []
for t_step, (t_x, t_y, t_f_r, t_f_t) in enumerate(test_loader):
    if gpu_avaliable:
        t_x = t_x.cuda()
        t_y = t_y.cuda()
        t_f_r = t_f_r.cuda()
        t_f_t = t_f_t.cuda()

    t_x = t_x.view(-1, 10, 8)
    test_output = model(t_x)  # (samples, time_step, input_size)
    if gpu_avaliable:
        pred_y = torch.max(test_output, 1)[1].cuda().data
    else:
        pred_y = torch.max(test_output, 1)[1].data.numpy()

    for ix, pred_label in enumerate(pred_y):
        dest_label.append(pred_label)
        pred_region = label_to_region(pred_label)
        if not pred_region == -1 and not pred_region == t_f_r[ix]:
            pred_time = region_transition_time[t_f_r[ix]][pred_region]
            if pred_time == 0:
                start_region_ix = region_to_matrix[t_f_r[ix].item()]
                end_region_ix = region_to_matrix[pred_region]
                grid_distance = abs(end_region_ix[0] - start_region_ix[0]) \
                                + abs(end_region_ix[1] - start_region_ix[1])
                arrive_time = datetime.strptime(
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_f_t[ix].item())),
                    "%Y-%m-%d %H:%M:%S")\
                              + timedelta(minutes=grid_distance * 5)
                if datetime.strptime("2014-10-22 08:30:00", "%Y-%m-%d %H:%M:%S") < arrive_time < \
                        datetime.strptime("2014-10-22 09:00:00", "%Y-%m-%d %H:%M:%S"):
                    all_pred_y.append(pred_region)
                    all_pred_time.append(grid_distance * 5)
            else:
                arrive_time = datetime.strptime(
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_f_t[ix].item())),
                    "%Y-%m-%d %H:%M:%S")\
                              + timedelta(seconds=pred_time)
                if datetime.strptime("2014-10-22 08:30:00", "%Y-%m-%d %H:%M:%S") < arrive_time < \
                        datetime.strptime("2014-10-22 09:00:00", "%Y-%m-%d %H:%M:%S"):
                    all_pred_y.append(pred_region)
                    all_pred_time.append(int(pred_time / 60))

youke_destination_distribution = {}
for ix, region in enumerate(all_pred_y):
    if not region in youke_destination_distribution:
        youke_destination_distribution[region] = 1
    else:
        youke_destination_distribution[region] = youke_destination_distribution[region] + 1
np.save("youke_destination_distribution_9am", youke_destination_distribution)

np.save("dest_label", dest_label)
