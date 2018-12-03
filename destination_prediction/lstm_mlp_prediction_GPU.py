import torch
from torch import nn
import numpy as np
import torch.utils.data as Data
import torch.nn.functional as F
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # gpu
gpu_avaliable = torch.cuda.is_available()
EPOCH = 10  # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 64
TIME_STEP = 10  # rnn time step / image height
INPUT_SIZE = 30  # rnn input size / image width
HIDDEN_SIZE = 128
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


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.GRU(
            input_size=INPUT_SIZE,
            hidden_size=128,  # rnn hidden unit
            # num_layers=2,  # number of rnn layer
            # batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        # self.embedding = nn.Embedding(input_size, hidden_size)
        self.car_embeds = nn.Embedding(len(car_to_ix), 10)
        self.poi_embeds = nn.Embedding(len(poi_to_ix), 10)
        self.region_embeds = nn.Embedding(1067, 10)

    def forward(self, x, hidden):
        new_vector = None
        new_vector = torch.cat((self.car_embeds(torch.cuda.LongTensor([x[0].item()]))[0],
                                self.region_embeds(torch.cuda.LongTensor([x[1].item()]))[0]))
        new_vector = torch.cat((new_vector, self.poi_embeds(torch.cuda.LongTensor([x[2].item()]))[0]))
        x = new_vector.view(1, 1, 30)

        # embedded = self.embedding(input).view(1, 1, -1)
        if gpu_avaliable:
            x = x.cuda()
        output = x
        output, hidden = self.lstm(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 100),
            nn.ReLU(),
            nn.Linear(100, label_size)
        )

    def forward(self, x):
        # convert tensor (128, 1, 28, 28) --> (128, 1*28*28)
        x = x.view(-1)
        if gpu_avaliable:
            x = x.cuda()
        x = self.layers(x)
        return x


encoder = EncoderRNN(INPUT_SIZE, HIDDEN_SIZE)
print(encoder)
mlp = MLP()
print(mlp)
if gpu_avaliable:
    encoder.cuda()
    mlp.cuda()

encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=LR)
mlp_optimizer = torch.optim.Adam(mlp.parameters(), lr=LR)

loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted


# 对轨迹序列进行循环训练，得到最后时刻的隐藏状态，并计算loss
def train(b_x):
    encoder_hidden = encoder.initHidden()

    batch_output = []
    for i, vector in enumerate(b_x):
        for item in vector:
            encoder_output, encoder_hidden = encoder(item, encoder_hidden)
        mlp_output = mlp(encoder_hidden)
        batch_output.append(mlp_output)
        # loss = loss + loss_func(mlp_output, b_y[i])

    return torch.stack(batch_output)


# training and testing
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(loader):  # gives batch data
        b_x = b_x.view(-1, 10, 3)  # reshape x to (batch, time_step, input_size)
        if gpu_avaliable:
            b_x = b_x.cuda()
            b_y = b_y.cuda()

        output = train(b_x)
        loss = loss_func(output, b_y)  # cross entropy loss
        encoder_optimizer.zero_grad()  # clear gradients for this training step
        mlp_optimizer.zero_grad()
        loss.backward()  # backpropagation, compute gradients
        encoder_optimizer.step()  # apply gradients
        mlp_optimizer.step()

        if step % 50 == 0:
            test_output = train(test_data)  # (samples, time_step, input_size)
            if gpu_avaliable:
                pred_y = torch.max(test_output, 1)[1].cuda().data
            else:
                pred_y = torch.max(test_output, 1)[1].data.numpy()
                accuracy = torch.sum(pred_y == test_labels).type(torch.FloatTensor) / test_labels.size(0)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

# print 10 predictions from test data
test_output = train(test_data[:10].view(-1, 10, 3))
if gpu_avaliable:
    pred_y = torch.max(test_output, 1)[1].cuda().data
else:
    pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y, 'prediction number')
print(test_labels[:10], 'real number')
