from math import radians, atan, tan, sin, acos, cos, sqrt
import torch
from destination_prediction.evaluation import get_cluster_center


class Evaluate:
    gpu_avaliable = torch.cuda.is_available()

    def __init__(self):
        pass

    @staticmethod
    def RMSE(pred_y, test_y):
        sum_exp_distance = 0.
        for i, pred_demand in enumerate(pred_y):
            if Evaluate.gpu_avaliable:
                pred_demand = pred_demand.item()
            test_demand = test_y[i]
            sum_exp_distance += pow(pred_demand - test_demand, 2)
        return sqrt(sum_exp_distance / len(pred_y))

    def recall(self, pred_y, test_y):
        pass

    def precision(self, pred_y, test_y):
        pass

    @staticmethod
    def MAPE(pred_y, test_y):
        sum_distance = 0.
        sum = 0
        for i, pred_demand in enumerate(pred_y):
            if Evaluate.gpu_avaliable:
                pred_demand = pred_demand.item()
            test_demand = test_y[i]
            if test_demand == 0:
                sum_distance += 0
                sum += 1
            else:
                sum_distance += abs(pred_demand - test_demand) / float(test_demand)
        return sum_distance / (len(pred_y) - sum)
