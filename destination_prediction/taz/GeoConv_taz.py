import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self, input_size=10, kernel_size=3, num_filter=32):
        super(Net, self).__init__()

        self.kernel_size = kernel_size
        self.num_filter = num_filter
        self.input_size = input_size
        self.build()

    def build(self):
        self.conv = nn.Conv1d(self.input_size, self.num_filter, self.kernel_size)

    def forward(self, traj):
        traj = traj.permute(0, 2, 1)
        conv_locs = F.elu(self.conv(traj)).permute(0, 2, 1)
        return conv_locs

