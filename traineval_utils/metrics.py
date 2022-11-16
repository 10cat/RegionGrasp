import os
import sys
sys.path.append('.')
sys.path.append('..')
from option import MyOptions as cfg
import torch 
import torch.nn as nn
from utils.utils import point2point_signed
from pytorch3d.structures import Meshes
from traineval_utils import simulation, interpenetraion, contact

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

    def penetration(self, signed_dists, penetrate_th=cfg.penetrate_threshold):
        """
        Compute the max penetration depth between predicted hand mesh and object mesh / origin hand mesh and object mesh;
        
        Return:
        - max penetration depth of generated mesh
        - max penetration depth of origin mesh
        - ratio between the two
        """
        o2h_signed_pred, o2h_signed, h2o_signed, h2o_signed_pred = signed_dists
        # (B, N1, 1) / (B, N2, 1)

        batch_size = o2h_signed.shape[0]
        obj_point_nb = o2h_signed.shape[1]
        hand_point_nb = h2o_signed.shape[1]

        # for signed < 0 as penetrate: take min as the max penetration depth
        func_max_depth = lambda SignedDist: torch.min(SignedDist, dim=1)
        # func_th_depth = lambda max_depths: max_depths[max_depths > penetrate_th] = 0.0
        func_mean_depth = lambda max_depths: torch.mean(max_depths)

        # use h2o or o2h for the penetration depth; h2o can be used only when h2o is signed
        if cfg.use_h2osigned:
            # import pdb; pdb.set_trace()
            signed_min_pred = func_max_depth(h2o_signed_pred).values # (B, 1)
            signed_min = func_max_depth(h2o_signed).values # (B, 1)
            
        else:
            signed_min_pred = func_max_depth(o2h_signed_pred).values # (B, 1)
            signed_min = func_max_depth(o2h_signed).values # (B, 1)

        # import pdb; pdb.set_trace()

        # given penetration threshold, set those under threshold to 0
        signed_min_pred[signed_min_pred > penetrate_th] = 0.0
        signed_min[signed_min > penetrate_th] = 0.0
        
        # TODO take batchmean: means of the penetration depths in this batch
        max_depth_pred = func_mean_depth(signed_min_pred)
        max_depth = func_mean_depth(signed_min)
        # import pdb; pdb.set_trace()

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

    def test_forward(self, sample_info, sample_idx):
        """
        Parameters:
        - sample_info: (dict), value_dtype = numpy_array
        ----------------
        Returns:
        - metrics: (AverageMeters), value_dtype = (AverageMeter)
        """
        save_gif_folder = os.path.join(cfg.output_dir, )
        sim_dist = simulation.process_sample(sample_idx, sample_info, save_gif_folder=cfg.output_dir, save_all_steps=True)
        



        return

