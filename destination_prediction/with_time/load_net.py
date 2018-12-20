import torch
import pickle
from destination_prediction.with_time.bi_lstm_prediction_qian_10dian_with_time import RNN

model = RNN()
model.load_state_dict(torch.load('F:/FCD data/models/bi-lstm_prediction_qian_10dian_with_time_1545296047.pt'))
model.eval()
print(model)
#
# with open('F:\FCD data\models\\bi-lstm_prediction_qian_10dian_with_time1545294590.pkl', 'rb') as f:
#     model = pickle.load(f)
#
#     print(model)