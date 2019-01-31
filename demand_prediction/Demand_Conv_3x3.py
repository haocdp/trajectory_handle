import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from torch.autograd import Variable


class Net(nn.Module):
    def __init__(self, input_size=1, kernel_size=3, num_filter=5):
        super(Net, self).__init__()

        self.kernel_size = kernel_size
        self.num_filter = num_filter
        self.input_size = input_size

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_size, self.num_filter, 2, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(self.num_filter, 16, 2, stride=1, padding=0),
            nn.ReLU(True),
            nn.AvgPool2d(2, 2)
        )

        self.fc = nn.Sequential(
            nn.Linear(16 * 1 * 1, 8),
            nn.Linear(8, 4)
        )

    def forward(self, traj):
        conv_locs = F.elu(self.conv(traj))
        conv_locs = conv_locs.view(-1, self.num_flat_features(conv_locs))
        conv_locs = self.fc(conv_locs)
        return conv_locs

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
