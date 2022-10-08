import torch 
import torch.nn as nn


class ConditionNetMetrics(nn.Module):
    def __init__(self):
        super(ConditionNetMetrics, self).__init__()
        self.maploss = nn.L1Loss()

    def forward(self, map_om, M_target):
        return self.maploss(map_om, M_target)
