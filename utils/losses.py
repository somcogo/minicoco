import torch
import torch.nn as nn
import torch.nn.functional as F

class AuxiliaryLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        loss = nn.L1Loss(x,y)
        return loss


class SampleLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.loss1 = nn.L1Loss()
        self.loss2 = AuxiliaryLoss()
        self.lambda1 = 1.0
        self.lambda2 = 1.0

    def forward(self, x, y):
        loss1 = self.lambda1 * self.loss1(x, y)
        loss2 = self.lambda2 * self.loss2(x, y)
        total_loss = loss1 + loss2

        return total_loss, (loss1, loss2)


