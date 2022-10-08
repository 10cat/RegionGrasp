import numpy as np
import torch
import torch.nn as nn
from option import MyOptions as cfg


class ConditionNetLoss(nn.Module):
    def __init__(self):
        super(ConditionNetLoss, self).__init__()
        self.maploss_hand = nn.L1Loss()
        self.maploss_om = nn.L1Loss()
        self.featloss = nn.MSELoss()

    def forward(self, feats, maps, M_target):
        dict_loss = {}
        map_h, map_om = maps
        feat_oh, feat_oom = feats
        target = M_target # directly use annotated sdmap -- no sigmoid
        # target = torch.sigmoid(M_target) # original target map 
        loss_hand = self.maploss_hand(map_h, target)
        loss_om = self.maploss_om(map_om, target)
        loss_feat = self.featloss(feat_oom, feat_oh)

        loss = loss_hand + loss_om + loss_feat
        dict_loss['loss_map_hand'] = loss_hand
        dict_loss['loss_map_om'] = loss_om
        dict_loss['loss_feat'] = loss_feat

        return loss, dict_loss


class cGraspvaeLoss(nn.Module):
    def __init__(self):
        super(cGraspvaeLoss, self).__init__()



if __name__ == "__main__":
    loss = ConditionNetLoss()
    

