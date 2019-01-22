import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.out = nn.Sequential(
            nn.Linear(64, 1)
        )

model = RNN(30)
print(model.state_dict())
model.load_state_dict(torch.load('params.pkl'))
print(model)
print(model.state_dict())
