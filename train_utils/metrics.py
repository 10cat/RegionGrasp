import torch 
import torch.nn as nn
from utils.utils import point2point_signed
from pytorch3d.structures import Meshes

class ConditionNetMetrics(nn.Module):
    def __init__(self):
        super(ConditionNetMetrics, self).__init__()
        self.maploss = nn.L1Loss()

    def forward(self, map_om, M_target):
        return self.maploss(map_om, M_target)

class cGraspvaeMetrics(nn.Module):
    def __init__(self, rh_model, rh_f, device):
        super(cGraspvaeMetrics, self).__init__()
        self.rh_model = rh_model
        self.rh_f = rh_f
        self.device = device

    def penetration(self, signed_dists):
        """
        Compute the max penetration depth between predicted hand mesh and object mesh / origin hand mesh and object mesh;
        
        Return:
        - max penetration depth of generated mesh
        - max penetration depth of origin mesh
        - ratio between the two
        """
        o2h_signed_pred, o2h_signed = signed_dists

        max_depth_pred = o2h_signed_pred.min()
        max_depth = o2h_signed.min()
        max_depth_ratio = max_depth_pred / max_depth

        return max_depth_pred, max_depth, max_depth_ratio



    def forward(self, signed_dists):

        """
        :params nearest_ids: (B, P2, 1)
        :region_centers
        """
        dict_metrics = {}

        ##### penetration metrics #####
        max_depth_pred, max_depth, max_depth_ratio = self.penetration(signed_dists)

        ##### simulation metrics #####


        dict_metrics = {'max_depth_ratio': max_depth_ratio, 
                        'max_depth_pred': max_depth_pred}

        return dict_metrics

