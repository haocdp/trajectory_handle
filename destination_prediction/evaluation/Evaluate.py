from math import radians, atan, tan, sin, acos, cos, sqrt
import torch
from destination_prediction.evaluation import get_cluster_center


class Evaluate:
    # cluter_center_dict = get_cluster_center.get_cluster_center()
    # gpu_avaliable = torch.cuda.is_available()

    def __init__(self):
        pass

    @staticmethod
    def get_distance(latA, lonA, latB, lonB):
        ra = 6378140
        rb = 6356755
        flatten = (ra - rb) / ra
        radLatA = radians(latA)
        radLonA = radians(lonA)
        radLatB = radians(latB)
        radLonB = radians(lonB)

        try:
            pA = atan(rb / ra * tan(radLatA))
            pB = atan(rb / ra * tan(radLatB))
            x = acos(sin(pA) * sin(pB) + cos(pA) * cos(pB) * cos(radLonA - radLonB))
            c1 = (sin(x) - x) * (sin(pA) + sin(pB)) ** 2 / cos(x / 2) ** 2
            c2 = (sin(x) + x) * (sin(pA) - sin(pB)) ** 2 / sin(x / 2) ** 2
            dr = flatten / 8 * (c1 - c2)
            distance = ra * (x + dr)
            return distance
        except:
            return 0.0000001

    @staticmethod
    def RMSE(pred_y, test_y):
        sum_exp_distance = 0.
        for i, pred_point in enumerate(pred_y):
            if Evaluate.gpu_avaliable:
                pred_point = Evaluate.cluter_center_dict[pred_point.item()]
            else:
                pred_point = Evaluate.cluter_center_dict[pred_point]
            test_point = test_y[i]
            sum_exp_distance += pow(Evaluate.get_distance(pred_point[1], pred_point[0], test_point[1], test_point[0]), 2)
        return sqrt(sum_exp_distance / len(pred_y))

    @staticmethod
    def accuracy(pred_y, test_y):
        return torch.sum(torch.LongTensor(pred_y) == torch.LongTensor(test_y)).type(torch.FloatTensor) / len(test_y)

    def recall(self, pred_y, test_y):
        pass

    def precision(self, pred_y, test_y):
        pass

    @staticmethod
    def MAE(pred_y, test_y):
        sum_distance = 0.
        for i, pred_point in enumerate(pred_y):
            if Evaluate.gpu_avaliable:
                pred_point = Evaluate.cluter_center_dict[pred_point.item()]
            else:
                pred_point = Evaluate.cluter_center_dict[pred_point]
            test_point = test_y[i]
            sum_distance += Evaluate.get_distance(pred_point[1], pred_point[0], test_point[1], test_point[0])
        return sum_distance / len(pred_y)
