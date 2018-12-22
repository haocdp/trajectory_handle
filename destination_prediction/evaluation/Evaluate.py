from math import radians, atan, tan, sin, acos, cos, sqrt
import torch

class Evaluate:
    def __init__(self):
        pass

    def get_distance(self, latA, lonA, latB, lonB):
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

    def RMSE(self, pred_y, test_y):
        sum_exp_distance = 0.
        for i, pred_point in enumerate(pred_y):
            test_point = test_y[i]
            sum_exp_distance += pow(self.get_distance(pred_point[1], pred_point[0], test_point[1], test_point[0]), 2)
        return sqrt(sum_exp_distance / len(pred_y))

    @staticmethod
    def accuracy(pred_y, test_y):
        return torch.sum(torch.LongTensor(pred_y) == torch.LongTensor(test_y)).type(torch.FloatTensor) / len(test_y)

    def recall(self, pred_y, test_y):
        pass

    def precision(self, pred_y, test_y):
        pass

    def MAE(self, pred_y, test_y):
        sum_distance = 0.
        for i, pred_point in enumerate(pred_y):
            test_point = test_y[i]
            sum_distance += self.get_distance(pred_point[1], pred_point[0], test_point[1], test_point[0])
        return sum_distance / len(pred_y)
