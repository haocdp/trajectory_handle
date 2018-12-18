"""
训练数据：（car_id, region_id, poi_id, week_day, time_slot）
embedding:
    (car_id -> 16)
    (week_id -> 3)
    (time_id -> 8)
    (poi_id -> 4)
    (region_id -> 8)

"""
import os
import sys
if sys.platform == 'linux':
    sys.path.append('/root/trajectory_handle/')


import torch
from torch import nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as Data
import torch.nn.functional as F
import random
import logger

# torch.manual_seed(1)    # reproducible
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # gpu
gpu_avaliable = torch.cuda.is_available()

EPOCH = 10  # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 64
TIME_STEP = 10  # rnn time step / image height
INPUT_SIZE = 390  # rnn input size / image width
HIDDEN_SIZE = 128
LR = 0.01  # learning rate
LAYER_NUM = 2

linux_path = "/root/taxiData"
windows_path = "F:/FCD data"
base_path = linux_path

labels = list(np.load(base_path + "/cluster/destination_labels.npy"))
# label个数
label_size = len(set(labels))
elogger = logger.Logger("mlp")


def load_data():
    # filepath1 = base_path + "/trajectory/allday/youke_0_result_npy.npy"
    filepath1 = base_path + "/trajectory/2014-10-20/trajectory_2014-10-20result_npy.npy"
    filepath2 = base_path + "/trajectory/2014-10-21/trajectory_2014-10-21result_npy.npy"
    filepath3 = base_path + "/trajectory/2014-10-22/trajectory_2014-10-22result_npy.npy"
    filepath4 = base_path + "/trajectory/2014-10-23/trajectory_2014-10-23result_npy.npy"
    filepath5 = base_path + "/trajectory/2014-10-24/trajectory_2014-10-24result_npy.npy"
    filepath6 = base_path + "/trajectory/2014-10-25/trajectory_2014-10-25result_npy.npy"
    filepath7 = base_path + "/trajectory/2014-10-26/trajectory_2014-10-26result_npy.npy"

    trajectories1 = list(np.load(filepath1))
    trajectories2 = list(np.load(filepath2))
    trajectories3 = list(np.load(filepath3))
    trajectories4 = list(np.load(filepath4))
    trajectories5 = list(np.load(filepath5))
    trajectories6 = list(np.load(filepath6))
    trajectories7 = list(np.load(filepath7))

    all_trajectories = []
    all_trajectories.extend(trajectories1)
    all_trajectories.extend(trajectories2)
    all_trajectories.extend(trajectories3)
    all_trajectories.extend(trajectories4)
    all_trajectories.extend(trajectories5)
    all_trajectories.extend(trajectories6)
    all_trajectories.extend(trajectories7)

    # 打乱
    random.shuffle(all_trajectories)

    print("all trajectories num : {}".format(len(all_trajectories)))
    elogger.log("all trajectories num : {}".format(len(all_trajectories)))
    count = len(all_trajectories) * 0.8

    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    car_to_ix = {}
    poi_to_ix = {}
    region_to_ix = {}
    for trajectory, label, weekday, time_slot in all_trajectories:
        for t in trajectory:
            if t[0] not in car_to_ix:
                car_to_ix[t[0]] = len(car_to_ix)
            if t[-1] not in poi_to_ix:
                poi_to_ix[t[-1]] = len(poi_to_ix)
            if t[-2] not in region_to_ix:
                region_to_ix[t[-2]] = len(region_to_ix)

    def transfer(tra, weekday, time_slot):
        new_tra = []
        for t in tra:
            new_t = []
            new_t.append(car_to_ix[t[0]])
            new_t.append(region_to_ix[t[5]])
            new_t.append(poi_to_ix[t[6]])
            new_t.append(weekday)
            new_t.append(time_slot)
            new_tra.append(new_t)
        return new_tra

    # 过滤轨迹，如果轨迹存在连续相同区域，则进行过滤
    def filter(tra):
        first_index = tra[0]
        new_tra = [tra[0]]
        for t in tra:
            if t[1] == first_index[1]:
                continue
            new_tra.append(t)
            first_index = t
        return new_tra

    c = 0
    for trajectory, label, weekday, time_slot in all_trajectories:
        new_tra = transfer(trajectory, weekday, time_slot)
        new_tra = filter(new_tra)
        if len(new_tra) < 10:
            c += 1
            continue
        if c < count:
            train_data.append(new_tra[:10])
            train_labels.append(label)
        else:
            test_data.append(new_tra[:10])
            test_labels.append(label)
        c += 1
    return train_data, train_labels, test_data, test_labels, car_to_ix, poi_to_ix, region_to_ix


# trajectory dataset
train_data, train_labels, test_data, test_labels, car_to_ix, poi_to_ix, region_to_ix = load_data()
train_data = torch.FloatTensor(train_data)
train_labels = torch.LongTensor(train_labels)
test_data = torch.FloatTensor(test_data)
test_labels = torch.LongTensor(test_labels)

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


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.car_embeds = nn.Embedding(len(car_to_ix), 16)
        self.poi_embeds = nn.Embedding(len(poi_to_ix), 4)
        self.region_embeds = nn.Embedding(len(region_to_ix), 8)
        self.week_embeds = nn.Embedding(7, 3)
        self.time_embeds = nn.Embedding(1440, 8)

        self.layers = nn.Sequential(
            nn.Linear(INPUT_SIZE, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, label_size)
        )

    def forward(self, x):
        # convert tensor (128, 1, 28, 28) --> (128, 1*28*28)
        new_vector = None
        for vector in x:
            for item in vector:
                # print(item)
                if new_vector is None:
                    if gpu_avaliable:
                        new_vector = torch.cat((self.car_embeds(torch.cuda.LongTensor([item[0].item()]))[0],
                                                self.region_embeds(torch.cuda.LongTensor([item[1].item()]))[0]))
                        new_vector = torch.cat((new_vector, self.poi_embeds(torch.cuda.LongTensor([item[2].item()]))[0]))
                        new_vector = torch.cat((new_vector, self.week_embeds(torch.cuda.LongTensor([item[3].item()]))[0]))
                        new_vector = torch.cat((new_vector, self.time_embeds(torch.cuda.LongTensor([item[4].item()]))[0]))
                    else:
                        new_vector = torch.cat((self.car_embeds(torch.LongTensor([item[0].item()]))[0],
                                                self.region_embeds(torch.LongTensor([item[1].item()]))[0]))
                        new_vector = torch.cat((new_vector, self.poi_embeds(torch.LongTensor([item[2].item()]))[0]))
                        new_vector = torch.cat((new_vector, self.week_embeds(torch.LongTensor([item[3].item()]))[0]))
                        new_vector = torch.cat((new_vector, self.time_embeds(torch.LongTensor([item[4].item()]))[0]))
                else:
                    if gpu_avaliable:
                        new_vector = torch.cat((new_vector, self.car_embeds(torch.cuda.LongTensor([item[0].item()]))[0]))
                        new_vector = torch.cat((new_vector, self.region_embeds(torch.cuda.LongTensor([item[1].item()]))[0]))
                        new_vector = torch.cat((new_vector, self.poi_embeds(torch.cuda.LongTensor([item[2].item()]))[0]))
                        new_vector = torch.cat((new_vector, self.week_embeds(torch.cuda.LongTensor([item[3].item()]))[0]))
                        new_vector = torch.cat((new_vector, self.time_embeds(torch.cuda.LongTensor([item[4].item()]))[0]))
                    else:
                        new_vector = torch.cat((new_vector, self.car_embeds(torch.LongTensor([item[0].item()]))[0]))
                        new_vector = torch.cat((new_vector, self.region_embeds(torch.LongTensor([item[1].item()]))[0]))
                        new_vector = torch.cat((new_vector, self.poi_embeds(torch.LongTensor([item[2].item()]))[0]))
                        new_vector = torch.cat((new_vector, self.week_embeds(torch.LongTensor([item[3].item()]))[0]))
                        new_vector = torch.cat((new_vector, self.time_embeds(torch.LongTensor([item[4].item()]))[0]))
        x = new_vector.view(-1, INPUT_SIZE)
        if gpu_avaliable:
            x = x.cuda()
        x = self.layers(x)
        x = F.softmax(x, dim=1)
        return x


mlp = MLP()
if gpu_avaliable:
    mlp.cuda()
print(mlp)
elogger.log(str(mlp))

mlp_optimizer = torch.optim.Adam(mlp.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted

# training and testing
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(loader):  # gives batch data
        if gpu_avaliable:
            b_x = b_x.cuda()
            b_y = b_y.cuda()

        b_x = b_x.view(-1, 10, 5)  # reshape x to (batch, time_step, input_size)

        output = mlp(b_x)  # rnn output
        loss = loss_func(output, b_y)  # cross entropy loss
        mlp_optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        mlp_optimizer.step()  # apply gradients
        del b_x, b_y

        if step % 50000 == 0:
            all_pred_y = []
            all_test_y = []
            for t_step, (t_x, t_y) in enumerate(test_loader):
                if gpu_avaliable:
                    t_x = t_x.cuda()
                    t_y = t_y.cuda()

                t_x = t_x.view(-1, 10, 5)
                test_output = mlp(t_x)  # (samples, time_step, input_size)
                if gpu_avaliable:
                    pred_y = torch.max(test_output, 1)[1].cuda().data
                else:
                    pred_y = torch.max(test_output, 1)[1].data.numpy()
                    t_y = t_y.data.numpy()
                all_pred_y.extend(pred_y)
                all_test_y.extend(list(t_y))
            accuracy = torch.sum(torch.LongTensor(all_pred_y) == torch.LongTensor(all_test_y)).type(torch.FloatTensor) / len(all_test_y)
            print_out = 'Epoch: ' + str(epoch) + '| train loss: %.4f' % loss.data.cpu().numpy() + '| test accuracy: %.2f' % accuracy
            print(print_out)
            elogger.log(str(print_out))

