import torch
import torch.nn as nn
import os
from demand_prediction.Demand_Conv import Net
import numpy as np
import torch.utils.data as Data

EPOCH = 100  # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 128
TIME_STEP = 6  # rnn time step / image height
INPUT_SIZE = 47  # rnn input size / image width
HIDDEN_SIZE = 256
LR = 0.0001  # learning rate
LAYER_NUM = 2
WEEKDAY_NUM = 7
TIME_SLOT = 48
REGION_NUM = 918
SEQ_LENGTH = 6

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # gpu
gpu_avaliable = torch.cuda.is_available()


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
                before_conv.extend(seqs[-49:].tolist())
        if gpu_avaliable:
            before_conv = torch.cuda.FloatTensor(before_conv)
        else:
            before_conv = torch.FloatTensor(before_conv)
        before_conv = before_conv.view(-1, SEQ_LENGTH, 7, 7)

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


model = RNN()
model.load_state_dict(torch.load('params.pkl'))

dispatcher_data = np.load("K:\毕业论文\TaxiData\demand\dispatcher_data_without_filter_9am.npy").tolist()
ix = np.load("K:\毕业论文\TaxiData\demand\\region_to_ix.npy").item()

def flatten(o):
    new_seq = []

    for d, conv in o[3]:
        new_o = []
        if o[0] < 5:
            new_o.append(0)
        else:
            new_o.append(1)
        new_o.append(o[1])
        new_o.append(o[2])
        new_o.append(d)
        for i in conv:
            for j in i:
                new_o.append(j)
        new_seq.append(new_o)
    return new_seq

test_data = []
test_labels = []

train_count = 0
for obj in dispatcher_data:
    test_data.append(flatten(obj))
    test_labels.append(obj[-1])

test_data = torch.LongTensor(test_data)
test_labels = torch.FloatTensor(test_labels)

test_dataset = Data.TensorDataset(test_data, test_labels)
test_loader = Data.DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE
    # shuffle=True
)

all_pred_y = []
all_test_y = []
region_prediction_dict = {}
for t_step, (t_x, t_y) in enumerate(test_loader):
    if gpu_avaliable:
        t_x = t_x.cuda()
        t_y = t_y.cuda()

    t_x = t_x.view(-1, TIME_STEP, 53)
    test_output = model(t_x)  # (samples, time_step, input_size)
    if gpu_avaliable:
        pred_y = test_output.cuda().data
    else:
        pred_y = test_output.data.numpy()
    all_pred_y.extend(pred_y)
    all_test_y.extend(list(t_y.data.cpu().numpy()))

    for i, x in enumerate(t_x):
        print("region:{}, ix: {}, time_slot: {}, real value: {}, pred value: {}".format(x[0][2], ix[x[0][2].item()], x[0][1].item(), t_y[i], round(test_output[i].item())))
        if t_y[i] < 10:
            region_prediction_dict[x[0][2].item()] = 0
        else:
            region_prediction_dict[x[0][2].item()] = round(test_output[i].item())

np.save("region_prediction_distribution_1pm", region_prediction_dict)

