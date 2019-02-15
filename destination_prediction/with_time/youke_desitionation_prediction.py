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
from destination_prediction.evaluation.Evaluate import Evaluate

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
windows_path = "D:\haoc\data\TaxiData"
base_path = linux_path

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
                        new_vector = torch.cat((new_vector, torch.cuda.FloatTensor([item[-2]])))
                        new_vector = torch.cat((new_vector, torch.cuda.FloatTensor([item[-1]])))
                    else:
                        new_vector = torch.cat((self.region_embeds(torch.LongTensor([item[1].item()]))[0],
                                                self.poi_embeds(torch.LongTensor([item[2].item()]))[0]))
                        new_vector = torch.cat((new_vector, torch.FloatTensor([item[-2]])))
                        new_vector = torch.cat((new_vector, torch.FloatTensor([item[-1]])))
                else:
                    if gpu_avaliable:
                        new_vector = torch.cat((new_vector, self.region_embeds(torch.cuda.LongTensor([item[1].item()]))[0]))
                        new_vector = torch.cat((new_vector, self.poi_embeds(torch.cuda.LongTensor([item[2].item()]))[0]))
                        new_vector = torch.cat((new_vector, torch.cuda.FloatTensor([item[-2]])))
                        new_vector = torch.cat((new_vector, torch.cuda.FloatTensor([item[-1]])))
                    else:
                        new_vector = torch.cat((new_vector, self.region_embeds(torch.LongTensor([item[1].item()]))[0]))
                        new_vector = torch.cat((new_vector, self.poi_embeds(torch.LongTensor([item[2].item()]))[0]))
                        new_vector = torch.cat((new_vector, torch.FloatTensor([item[-2]])))
                        new_vector = torch.cat((new_vector, torch.FloatTensor([item[-1]])))


            if embedding_vector is None:
                if gpu_avaliable:
                    embedding_vector = torch.cat((self.car_embeds(torch.cuda.LongTensor([vector[0][0].item()]))[0],
                                                  self.week_embeds(torch.cuda.LongTensor([vector[0][3].item()]))[0]))
                    embedding_vector = torch.cat((embedding_vector, self.time_embeds(torch.cuda.LongTensor([vector[0][4].item()]))[0]))
                else:
                    embedding_vector = torch.cat((self.car_embeds(torch.LongTensor([vector[0][0].item()]))[0],
                                                  self.week_embeds(torch.LongTensor([vector[0][3].item()]))[0]))
                    embedding_vector = torch.cat((embedding_vector, self.time_embeds(torch.LongTensor([vector[0][4].item()]))[0]))
            else:
                if gpu_avaliable:
                    embedding_vector = torch.cat(
                        (embedding_vector, self.car_embeds(torch.cuda.LongTensor([vector[0][0].item()]))[0]))
                    embedding_vector = torch.cat(
                        (embedding_vector, self.week_embeds(torch.cuda.LongTensor([vector[0][3].item()]))[0]))
                    embedding_vector = torch.cat(
                        (embedding_vector, self.time_embeds(torch.cuda.LongTensor([vector[0][4].item()]))[0]))
                else:
                    embedding_vector = torch.cat((embedding_vector, self.car_embeds(torch.LongTensor([vector[0][0].item()]))[0]))
                    embedding_vector = torch.cat((embedding_vector, self.week_embeds(torch.LongTensor([vector[0][3].item()]))[0]))
                    embedding_vector = torch.cat((embedding_vector, self.time_embeds(torch.LongTensor([vector[0][4].item()]))[0]))

        new_vector = new_vector.view(-1, 10, 14)
        new_vector = torch.cat((self.region_poi_linear(new_vector[:, :, :-2]), self.coord_linear(new_vector[:, :, -2:])), 2)
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
model.load_state_dict(torch.load('cnn_lstm_prediction_new_cluster.pkl'))

youke_trajectories = np.load("youke_trajectories_data.npy").tolist()

