"""
View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou

Dependencies:
torch: 0.4
matplotlib
torchvision
"""
import os
import torch
from torch import nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as Data
import torch.nn.functional as F

# torch.manual_seed(1)    # reproducible
os.environ["CUDA_VISIBLE_DEVICES"] = "4"  # gpu
gpu_avaliable = torch.cuda.is_available()

# Hyper Parameters
EPOCH = 10  # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 64
TIME_STEP = 10  # rnn time step / image height
INPUT_SIZE = 30  # rnn input size / image width
HIDDEN_SIZE = 512
LR = 0.01  # learning rate

labels = list(np.load("/root/data/cluster/destination_labels.npy"))
# label个数
label_size = len(set(labels))


def load_data():
    filepath = "/root/data/trajectory/workday_trajectory_destination/youke_1_result_npy.npy"
    trajectories = list(np.load(filepath))
    count = len(trajectories) * 0.8

    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    car_to_ix = {}
    poi_to_ix = {}
    region_to_ix = {}
    for trajectory, label in trajectories:
        for t in trajectory:
            if t[0] not in car_to_ix:
                car_to_ix[t[0]] = len(car_to_ix)
            if t[-1] not in poi_to_ix:
                poi_to_ix[t[-1]] = len(poi_to_ix)
            if t[-2] not in region_to_ix:
                region_to_ix[t[-2]] = len(region_to_ix)

    def transfer(tra):
        new_tra = []
        for t in tra:
            new_t = []
            new_t.append(car_to_ix[t[0]])
            new_t.append(region_to_ix[t[5]])
            new_t.append(poi_to_ix[t[6]])
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
    for trajectory, label in trajectories:
        new_tra = transfer(trajectory)
        new_tra = filter(new_tra)
        if len(new_tra) < 10:
            c += 1
            continue
        if c < count:
            train_data.append(new_tra[:5] + new_tra[-5:])
            train_labels.append(label)
        else:
            test_data.append(new_tra[:5] + new_tra[-5:])
            test_labels.append(label)
        c += 1
    return train_data, train_labels, test_data, test_labels, car_to_ix, poi_to_ix


# trajectory dataset
train_data, train_labels, test_data, test_labels, car_to_ix, poi_to_ix = load_data()

train_data = torch.FloatTensor(train_data)
train_labels = torch.LongTensor(train_labels)
test_data = torch.FloatTensor(test_data).cuda() if gpu_avaliable else torch.FloatTensor(test_data)
test_labels = torch.LongTensor(test_labels).cuda() if gpu_avaliable else torch.LongTensor(test_labels)

torch_dataset = Data.TensorDataset(train_data, train_labels)
loader = Data.DataLoader(
    dataset=torch_dataset,  # torch TensorDataset format
    batch_size=BATCH_SIZE,  # mini batch size
    shuffle=True  # 要不要打乱数据 (打乱比较好)
)


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(  # if use nn.RNN(), it hardly learns
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,  # rnn hidden unit
            num_layers=2,  # number of rnn layer
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.out = nn.Linear(HIDDEN_SIZE, label_size)
        self.car_embeds = nn.Embedding(len(car_to_ix), 10)
        self.poi_embeds = nn.Embedding(len(poi_to_ix), 10)
        self.region_embeds = nn.Embedding(1067, 10)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        new_vector = None
        for vector in x:
            for item in vector:
                if new_vector is None:
                    new_vector = torch.cat((self.car_embeds(torch.cuda.LongTensor([item[0].item()]))[0],
                                            self.region_embeds(torch.cuda.LongTensor([item[1].item()]))[0]))
                    new_vector = torch.cat((new_vector, self.poi_embeds(torch.cuda.LongTensor([item[2].item()]))[0]))
                else:
                    new_vector = torch.cat((new_vector, self.car_embeds(torch.cuda.LongTensor([item[0].item()]))[0]))
                    new_vector = torch.cat((new_vector, self.region_embeds(torch.cuda.LongTensor([item[1].item()]))[0]))
                    new_vector = torch.cat((new_vector, self.poi_embeds(torch.cuda.LongTensor([item[2].item()]))[0]))
        x = new_vector.view(-1, 10, 30)
        if gpu_avaliable:
            x = x.cuda()
        r_out, (h_n, h_c) = self.rnn(x, None)  # None represents zero initial hidden state

        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
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

        b_x = b_x.view(-1, 10, 3)  # reshape x to (batch, time_step, input_size)

        output = rnn(b_x)  # rnn output
        loss = loss_func(output, b_y)  # cross entropy loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        if step % 50 == 0:
            test_output = rnn(test_data)  # (samples, time_step, input_size)
            if gpu_avaliable:
                pred_y = torch.max(test_output, 1)[1].cuda().data
            else:
                pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = torch.sum(pred_y == test_labels).type(torch.FloatTensor) / test_labels.size(0)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy(), '| test accuracy: %.2f' % accuracy)

# print 10 predictions from test data
test_output = rnn(test_data[:10].view(-1, 10, 3))
if gpu_avaliable:
    pred_y = torch.max(test_output, 1)[1].cuda().data
else:
    pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y, 'prediction number')
print(test_labels[:10], 'real number')