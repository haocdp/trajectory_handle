"""
训练数据：（car_id, region_id, poi_id, week_day, time_slot）
精细化轨迹，使用坐标转换表示，替代区域表示，并使用卷积来提取局部特征

使用新的目的地聚类结果
"""
"""
训练数据：（car_id, region_id, poi_id, week_day, time_slot）
embedding:
    (car_id -> 16)
    (week_id -> 3)
    (time_id -> 8)
    (poi_id -> 4)
    (region_id -> 8)

"""
import sys
if sys.platform == 'linux':
    sys.path.append('/root/trajectory_handle/')
import os
import torch
from torch import nn
import numpy as np
import torch.utils.data as Data
import torch.nn.functional as F
import random
from destination_prediction.with_time import GeoConv
import logger
from destination_prediction_porto.evaluation.Evaluate import Evaluate

# torch.manual_seed(1)    # reproducible
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # gpu
gpu_avaliable = torch.cuda.is_available()

# Hyper Parameters
EPOCH = 10  # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 64
TIME_STEP = 8  # rnn time step / image height
INPUT_SIZE = 59  # rnn input size / image width
HIDDEN_SIZE = 256
LR = 0.0001  # learning rate
LAYER_NUM = 2

linux_path = "/root/TaxiData_Porto"
windows_path = "K:/毕业论文/TaxiData_Porto"
base_path = linux_path

labels = list(np.load(base_path + "/cluster/destination_labels.npy"))
# label个数
label_size = len(set(labels))
elogger = logger.Logger("cnn_lstm_prediction_porto")

max_lng = -8.55
min_lng = -8.70
max_lat = 41.25
min_lat = 41.
dis_lng = max_lng - min_lng
dis_lat = max_lat - min_lat


def load_data():
    filepath = base_path + "/trajectory_result.npy"
    test_filepath = base_path + '/test_trajectory_result.npy'

    trajectories = list(np.load(filepath))
    test_trajectories = list(np.load(test_filepath))

    # 打乱
    random.shuffle(trajectories)

    print("all trajectories num : {}".format(len(trajectories)))

    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    test_dest = []

    car_to_ix = {}
    poi_to_ix = {}
    region_to_ix = {}
    for trajectory, label, weekday, time_slot in trajectories:
        for t in trajectory:
            if t[0] not in car_to_ix:
                car_to_ix[t[0]] = len(car_to_ix)
            if t[-1] not in poi_to_ix:
                poi_to_ix[t[-1]] = len(poi_to_ix)
            if t[-2] not in region_to_ix:
                region_to_ix[t[-2]] = len(region_to_ix)

    for trajectory, label, weekday, time_slot in test_trajectories:
        for t in trajectory:
            if t[0] not in car_to_ix:
                car_to_ix[t[0]] = len(car_to_ix)
            if t[-1] not in poi_to_ix:
                poi_to_ix[t[-1]] = len(poi_to_ix)
            if t[-2] not in region_to_ix:
                region_to_ix[t[-2]] = len(region_to_ix)
    np.save("trajectory_without_filter_car_to_ix", car_to_ix)
    np.save("trajectory_without_filter_poi_to_ix", poi_to_ix)
    np.save("trajectory_without_filter_region_to_ix", region_to_ix)

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
            new_tra.append(new_t)
        return new_tra

    for trajectory, label, weekday, time_slot in trajectories:
        new_tra = transfer(trajectory, weekday, time_slot)
        if len(new_tra) < 10:
            n_t = []
            n_t.extend(new_tra)
            for i in range(10 - len(new_tra)):
                n_t.append(new_tra[-1])
            train_data.append(n_t)
            train_labels.append(label)
        else:
            for i in range(10, len(new_tra) + 1):
                n_t = []
                n_t.extend(new_tra[:5])
                n_t.extend(new_tra[i - 5:i])
                train_data.append(n_t)
                train_labels.append(label)

    for trajectory, label, weekday, time_slot in test_trajectories:
        new_tra = transfer(trajectory, weekday, time_slot)
        if len(new_tra) < 10:
            n_t = []
            n_t.extend(new_tra)
            for i in range(10 - len(new_tra)):
                n_t.append(new_tra[-1])
            test_data.append(n_t)
            test_labels.append(label)
            test_dest.append(list(map(float, trajectory[-1][1:3])))
        else:
            for i in range(10, len(new_tra) + 1):
                n_t = []
                n_t.extend(new_tra[:5])
                n_t.extend(new_tra[i - 5:i])
                test_data.append(n_t)
                test_labels.append(label)
                test_dest.append(list(map(float, trajectory[-1][1:3])))

    return train_data, train_labels, test_data, test_labels, test_dest, car_to_ix, poi_to_ix, region_to_ix


# trajectory dataset
train_data, train_labels, test_data, test_labels, test_dest, car_to_ix, poi_to_ix, region_to_ix = load_data()

train_data = torch.FloatTensor(train_data)
train_labels = torch.LongTensor(train_labels)
test_data = torch.FloatTensor(test_data)
test_labels = torch.LongTensor(test_labels)
test_dest = torch.FloatTensor(test_dest)

torch_dataset = Data.TensorDataset(train_data, train_labels)
loader = Data.DataLoader(
    dataset=torch_dataset,  # torch TensorDataset format
    batch_size=BATCH_SIZE,  # mini batch size
    shuffle=True  # 要不要打乱数据 (打乱比较好)
)

test_dataset = Data.TensorDataset(test_data, test_labels, test_dest)
test_loader = Data.DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


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
        self.car_embeds = nn.Embedding(len(car_to_ix), 16)
        self.poi_embeds = nn.Embedding(len(poi_to_ix), 4)
        self.region_embeds = nn.Embedding(len(region_to_ix), 8)
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


rnn = RNN()
if gpu_avaliable:
    rnn.cuda()
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted

# training and testing
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(loader):  # gives batch data
        if gpu_avaliable:
            b_x = b_x.cuda()
            b_y = b_y.cuda()

        b_x = b_x.view(-1, 10, 7)  # reshape x to (batch, time_step, input_size)

        output = rnn(b_x)  # rnn output
        loss = loss_func(output, b_y)  # cross entropy loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients
        del b_x, b_y

        if step % 100000 == 0:
            all_pred_y = []
            all_test_y = []
            all_test_d = []
            for t_step, (t_x, t_y, t_d) in enumerate(test_loader):
                if gpu_avaliable:
                    t_x = t_x.cuda()
                    t_y = t_y.cuda()
                    t_d = t_d.cuda()

                t_x = t_x.view(-1, 10, 7)
                test_output = rnn(t_x)  # (samples, time_step, input_size)
                if gpu_avaliable:
                    pred_y = torch.max(test_output, 1)[1].cuda().data
                else:
                    pred_y = torch.max(test_output, 1)[1].data.numpy()
                    # t_y = t_y.data.numpy()
                all_pred_y.extend(pred_y)
                all_test_y.extend(list(t_y.data.cpu().numpy()))
                all_test_d.extend(list(t_d.data.cpu().numpy()))
            print_out = 'Epoch: ' + str(epoch) + '| train loss: %.4f' % loss.data.cpu().numpy() + \
                        '| test accuracy: %.4f' % Evaluate.accuracy(all_pred_y, all_test_y) + \
                        '| test MAE: %.4f' % Evaluate.MAE(all_pred_y, all_test_d) + \
                        '| test RMSE: %.4f' % Evaluate.RMSE(all_pred_y, all_test_d)
            print(print_out)
            elogger.log(str(print_out))

torch.save(rnn.state_dict(), 'cnn_lstm_prediction.pkl')