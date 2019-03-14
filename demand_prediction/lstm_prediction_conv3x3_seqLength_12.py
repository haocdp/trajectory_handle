import os
import sys

if sys.platform == 'linux':
    sys.path.append('/root/trajectory_handle/')

import torch
from torch import nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np
import torch.utils.data as Data
import torch.nn.functional as F
import random
import logger
from demand_prediction.evaluation.Evaluate import Evaluate
from demand_prediction.Demand_Conv_3x3 import Net

# torch.manual_seed(1)    # reproducible
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # gpu
gpu_avaliable = torch.cuda.is_available()

# Hyper Parameters
EPOCH = 100  # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 128
TIME_STEP = 12  # rnn time step / image height
INPUT_SIZE = 19  # rnn input size / image width
HIDDEN_SIZE = 256
LR = 0.0001  # learning rate
LAYER_NUM = 2
WEEKDAY_NUM = 7
TIME_SLOT = 48
REGION_NUM = 918
SEQ_LENGTH = 12

linux_path = "/root/taxiData"
windows_path = "K:\毕业论文\TaxiData"
mac_path = "/Volumes/MyZone/毕业论文/TaxiData"
base_path = linux_path

elogger = logger.Logger("demand_lstm_prediction_conv3x3_seqLength_4")


def load_data():
    net_dataset = np.load(base_path + "/demand/net_data_without_filter_7x7_seq_length_14.npy").tolist()

    # 打乱
    random.shuffle(net_dataset)

    single_region_dataset = []
    for data in net_dataset:
        if data[-1] > 10:
            single_region_dataset.append(data)
    net_dataset = single_region_dataset

    print("all data sample num : {}".format(len(net_dataset)))
    elogger.log("all trajectories num : {}".format(len(net_dataset)))
    count = len(net_dataset) * 0.8

    def flatten(o):
        new_seq = []

        count = 0
        for d, conv in o[3]:
            count += 1
            if count <= 14 - SEQ_LENGTH:
                continue

            new_o = []
            new_o.append(o[0])
            new_o.append(o[1])
            new_o.append(o[2])
            new_o.append(d)

            for row, i in enumerate(conv):
                if not 2 <= row <= 4:
                    continue
                for col, j in enumerate(i):
                    if not 2 <= col <= 4:
                        continue
                    new_o.append(j)
            new_seq.append(new_o)
        return new_seq

    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    train_count = 0
    for obj in net_dataset:
        if train_count < count:
            train_data.append(flatten(obj))
            train_labels.append(obj[-1])
        else:
            test_data.append(flatten(obj))
            test_labels.append(obj[-1])
        train_count += 1
    return train_data, train_labels, test_data, test_labels


# trajectory dataset
train_data, train_labels, test_data, test_labels = load_data()

train_data = torch.LongTensor(train_data)
train_labels = torch.FloatTensor(train_labels)
test_data = torch.LongTensor(test_data)
test_labels = torch.FloatTensor(test_labels)

torch_dataset = Data.TensorDataset(train_data, train_labels)
loader = Data.DataLoader(
    dataset=torch_dataset,  # torch TensorDataset format
    batch_size=BATCH_SIZE,  # mini batch size
    shuffle=True  # 要不要打乱数据 (打乱比较好)
)

test_dataset = Data.TensorDataset(test_data, test_labels)
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

        # self.out = nn.Linear(HIDDEN_SIZE, label_size)
        self.region_embeds = nn.Embedding(REGION_NUM, 8)
        self.week_embeds = nn.Embedding(WEEKDAY_NUM, 3)
        self.time_embeds = nn.Embedding(TIME_SLOT, 4)
        if gpu_avaliable:
            self.conv = Net().cuda()
        else:
            self.conv = Net()

        # self.fc = nn.Linear(HIDDEN_SIZE, 1)
        self.fc = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 128),
            nn.Linear(128, 64),
            nn.Linear(64, 32),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        before_conv = []
        for batches in x:
            for seqs in batches:
                before_conv.extend(seqs[-9:].tolist())
        if gpu_avaliable:
            before_conv = torch.cuda.FloatTensor(before_conv)
        else:
            before_conv = torch.FloatTensor(before_conv)
        before_conv = before_conv.view(-1, SEQ_LENGTH, 3, 3)

        convs = None
        for item in torch.split(before_conv, 1, 1):
            if convs is None:
                convs = torch.unsqueeze(self.conv(item), 1)
            else:
                convs = torch.cat((convs, torch.unsqueeze(self.conv(item), 1)), 1)
        new_x = torch.cat((self.week_embeds(x[:, :, 0]), self.time_embeds(x[:, :, 1])), 2)
        new_x = torch.cat((new_x, self.region_embeds(x[:, :, 2])), 2)
        x = torch.cat((new_x, convs), 2)

        if gpu_avaliable:
            x = x.cuda()
        r_out, (h_n, h_c) = self.rnn(x, None)  # None represents zero initial hidden state

        # choose r_out at the last time step
        out = torch.squeeze(self.fc(r_out[:, -1, :]), 1)
        del x, r_out, h_c, h_n
        # out = F.softmax(out, 1)
        return out


rnn = RNN()
if gpu_avaliable:
    rnn.cuda()
print(rnn)
elogger.log(str(rnn))

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  # optimize all cnn parameters
loss_func = nn.MSELoss()

# training and testing
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(loader):  # gives batch data
        if gpu_avaliable:
            b_x = b_x.cuda()
            b_y = b_y.cuda()

        b_x = b_x.view(-1, TIME_STEP, 13)  # reshape x to (batch, time_step, input_size)

        output = rnn(b_x)  # rnn output
        loss = loss_func(output, b_y)  # cross entropy loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients
        del b_x, b_y

        if step % 1000 == 0:
            all_pred_y = []
            all_test_y = []
            for t_step, (t_x, t_y) in enumerate(test_loader):
                if gpu_avaliable:
                    t_x = t_x.cuda()
                    t_y = t_y.cuda()

                t_x = t_x.view(-1, TIME_STEP, 13)
                test_output = rnn(t_x)  # (samples, time_step, input_size)
                if gpu_avaliable:
                    pred_y = test_output.cuda().data
                else:
                    pred_y = test_output.data.numpy()
                all_pred_y.extend(pred_y)
                all_test_y.extend(list(t_y.data.cpu().numpy()))
            print_out = 'Epoch: ' + str(epoch) + '| train loss: %.4f' % loss.data.cpu().numpy() + \
                        '| test MAPE: %.4f' % Evaluate.MAPE(all_pred_y, all_test_y) + \
                        '| test RMSE: %.4f' % Evaluate.RMSE(all_pred_y, all_test_y)
            print(print_out)
            elogger.log(str(print_out))

# torch.save(rnn.state_dict(), 'params.pkl')

# print 10 predictions from test data
# test_output = rnn(test_data[:10].view(-1, 10, 5))
# if gpu_avaliable:
#     pred_y = torch.max(test_output, 1)[1].cuda().data
# else:
#     pred_y = torch.max(test_output, 1)[1].data.numpy()
# print(pred_y, 'prediction number')
# print(test_labels[:10], 'real number')
