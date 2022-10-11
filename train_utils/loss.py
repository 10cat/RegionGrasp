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
        map_h, map_om, feat_oh, feat_oom = None, None, None, None
        dict_loss = {}
        if len(maps) > 1:
            map_h, map_om = maps
        else:
            map_om = maps[0]
        if len(feats) > 1:
            feat_oh, feat_oom = feats
        else:
            feat_oom = feats[0]
        target = M_target # directly use annotated sdmap -- no sigmoid
        # target = torch.sigmoid(M_target) # original target map 
        loss_hand = self.maploss_hand(map_h, target) if map_h is not None else 0.0
        loss_om = self.maploss_om(map_om, target)
        loss_feat = self.featloss(feat_oom, feat_oh) if feat_oh is not None else 0.0

        loss = loss_hand + cfg.lambda_om * loss_om + cfg.lambda_feat * loss_feat
        dict_loss['loss_map_hand'] = loss_hand
        dict_loss['loss_map_om'] = loss_om
        dict_loss['loss_feat'] = loss_feat

        return loss, dict_loss


class cGraspvaeLoss(nn.Module):
    def __init__(self):
        super(cGraspvaeLoss, self).__init__()

        


if __name__ == "__main__":
    loss = ConditionNetLoss()
    

