import sys
sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from option import MyOptions as cfg
from utils.utils import get_std, point2point_signed
from pytorch3d.structures import Meshes
from chamfer_distance import ChamferDistance as ch_dist
from utils.utils import edges_for, decode_hand_params_batch



# class ConditionNetLoss(nn.Module):
#     def __init__(self):
#         super(ConditionNetLoss, self).__init__()
#         self.maploss_hand = nn.L1Loss()
#         self.maploss_om = nn.L1Loss()
#         self.featloss = nn.MSELoss()

#     def forward(self, feats, maps, M_target):
#         map_h, map_om, feat_oh, feat_oom = None, None, None, None
#         dict_loss = {}
#         if len(maps) > 1:
#             map_h, map_om = maps
#         else:
#             map_om = maps[0]
#         if len(feats) > 1:
#             feat_oh, feat_oom = feats
#         else:
#             feat_oom = feats[0]
#         target = M_target # directly use annotated sdmap -- no sigmoid
#         # target = torch.sigmoid(M_target) # original target map 
#         loss_hand = self.maploss_hand(map_h, target) if map_h is not None else 0.0
#         loss_om = self.maploss_om(map_om, target)
#         loss_feat = self.featloss(feat_oom, feat_oh) if feat_oh is not None else 0.0

#         loss = loss_hand + cfg.lambda_om * loss_om + cfg.lambda_feat * loss_feat
#         dict_loss['loss_map_hand'] = loss_hand
#         dict_loss['loss_map_om'] = loss_om
#         dict_loss['loss_feat'] = loss_feat

#         return loss, dict_loss


class cGraspvaeLoss(nn.Module):
    def __init__(self, device, cfg):
        super(cGraspvaeLoss, self).__init__()
        self.device = device
        self.LossL1 = torch.nn.L1Loss(reduction='mean')
        self.HOILoss = HOILoss(cfg, device)
        self.v_weights = torch.from_numpy(np.load(cfg.c_weights_path)).to(torch.float32).to(self.device) # 这个到底是啥呀？ 能不能用在其他数据集上？
        self.v_weights2 = torch.pow(self.v_weights, 1.0/2.5) # 这个到底是啥呀？ 能不能用在其他数据集上？
        self.vpe = torch.from_numpy(np.load(cfg.vpe_path)).to(self.device).to(torch.long) # 这个到底是啥呀？ 能不能用在其他数据集上？
        self.cfg = cfg
        self.latent_size = cfg.model.vae.kwargs.latent_size
        self.batch_size = cfg.batch_size
        self.mano_rh_path = cfg.mano_rh_path
        self.coefs = cfg.loss.coef
        

    def dist_loss(self, h2o_signed, h2o_signed_pred, o2h_signed, o2h_signed_pred, region):
        loss_dist_h, loss_dist_o = self.HOILoss(h2o_signed, h2o_signed_pred, o2h_signed, o2h_signed_pred, region)
        return loss_dist_h, loss_dist_o

    def KLLoss(self, rhand_vs, p_mean, log_vars):
        device = rhand_vs.device
        dtype = rhand_vs.dtype
        # import pdb; pdb.set_trace()
        B = rhand_vs.size(0)

        p_std = get_std(log_vars, self.cfg) 
        q_z = torch.distributions.normal.Normal(p_mean, p_std)
        
        p_z = torch.distributions.normal.Normal(
            loc=torch.tensor(np.zeros([B, self.latent_size]), requires_grad=False).to(device).to(dtype),
            scale=torch.tensor(np.ones([B, self.latent_size]), requires_grad=False).to(device).to(dtype)
        )
        loss_kl = self.coefs['kl_coef'] * torch.mean(torch.sum(torch.distributions.kl.kl_divergence(q_z, p_z), dim=[1]))

        return loss_kl

    def forward(self, hand_params, sample_stats, obj_vs, rhand_vs, region, obj_normals=None, obj_mesh_faces=None, mode=None):

        B = rhand_vs.size(0)
        
        rhand_pred, rh_model = decode_hand_params_batch(hand_params, B, self.cfg, self.device)
        
        rhand_vs_pred = rhand_pred.vertices
        if rhand_vs.shape[-1] != 3:
            rhand_vs = rhand_vs.transpose(2, 1)
        rhand_vs = rhand_vs.to(self.device)
        
        if obj_vs.shape[-1] != 3:
            obj_vs = obj_vs.transpose(2, 1)
        obj_vs = obj_vs.to(self.device)
        
        rh_f_single = torch.from_numpy(rh_model.faces.astype(np.int32)).view(1, -1, 3)
        rhand_faces = rh_f_single.repeat(B, 1, 1).to(self.device).to(torch.long)
        # import pdb; pdb.set_trace() # to check decode module
        
        
        
        dict_loss = {}
        loss_cfg = self.cfg.loss[mode]
        #### dist Loss ####
        if loss_cfg.loss_dist_h or loss_cfg.loss_dist_o:
            rh_normals = Meshes(verts=rhand_vs, faces=rhand_faces).to(self.device).verts_normals_packed().view(-1, 778, 3)
            rh_normals_pred = Meshes(verts=rhand_vs_pred, faces=rhand_faces).to(self.device).verts_normals_packed().view(-1, 778, 3) # packed representation of the vertex normals
            
            if obj_mesh_faces is not None:
                # obj_mesh_faces = torch.Tensor(obj_mesh_faces)
                # obj_vs = obj_vs.tolist()
                num_obj_verts = obj_vs.shape[1]
                obj_vs_list = [obj_vs[i] for i in range(obj_vs.shape[0])] # need to be consistent length list with obj_mesh_faces
                obj_normals = Meshes(verts=obj_vs_list, faces=obj_mesh_faces).to(self.device).verts_normals_packed().view(-1, num_obj_verts, 3)
            elif obj_normals is None:
                obj_normals = None
            # import pdb; pdb.set_trace()
            obj_normals = obj_normals.to(torch.float32).to(self.device)
            o2h_signed, h2o_signed, _, _ = point2point_signed(rhand_vs, obj_vs, rh_normals, obj_normals)
            o2h_signed_pred, h2o_signed_pred, _, _ = point2point_signed(rhand_vs_pred, obj_vs, rh_normals_pred, obj_normals)
            loss_dist_h, loss_dist_o = self.dist_loss(h2o_signed, h2o_signed_pred, o2h_signed, o2h_signed_pred, region)
            
            if loss_cfg.loss_dist_h:
                dict_loss.update({'loss_dist_h': loss_dist_h})
            if loss_cfg.loss_dist_o:
                dict_loss.update({'loss_dist_o': loss_dist_o})
        else:
            o2h_signed_pred, o2h_signed, h2o_signed, h2o_signed_pred = None, None, None, None
            

        #### KL Loss ####
        if sample_stats is not None: 
            p_mean, log_vars, Zin = sample_stats
            loss_kl = self.KLLoss(rhand_vs, p_mean, log_vars)
        else:
            # validation / test中loss_kl = 0
            loss_kl = torch.tensor(0.0, dtype=float).to(self.device)
        dict_loss.update({'loss_kl': loss_kl})

        #### verts Loss ####
        loss_mesh_rec = self.coefs['lambda_mesh_rec'] * (1 - self.coefs['kl_coef']) * torch.mean(torch.einsum('ijk,j->ijk', torch.abs((rhand_vs - rhand_vs_pred)), self.v_weights2)) # should be v_weights2 instead of v_weights
        if loss_cfg.loss_mesh_rec:
            dict_loss.update({'loss_mesh_rec': loss_mesh_rec})

        #### edge Loss ####
        loss_edge = self.coefs['lambda_edge'] * (1 - self.coefs['kl_coef']) * self.LossL1(edges_for(rhand_vs_pred, self.vpe), edges_for(rhand_vs, self.vpe))
        if loss_cfg.loss_edge:
            dict_loss.update({'loss_edge': loss_edge})
            
        loss_total = torch.stack(list(dict_loss.values())).sum()

        signed_dists = [o2h_signed_pred, o2h_signed, h2o_signed, h2o_signed_pred]
        
        return loss_total, dict_loss, signed_dists, rhand_vs_pred, rhand_faces

class HOILoss(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.device = device
        self.v_weights = torch.from_numpy(np.load(cfg.c_weights_path)).to(torch.float32).to(self.device) # 这个到底是啥呀？ 能不能用在其他数据集上？
        self.v_weights2 = torch.pow(self.v_weights, 1.0/2.5) # 这个到底是啥呀？ 能不能用在其他数据集上？
        self.vpe = torch.from_numpy(np.load(cfg.vpe_path)).to(self.device).to(torch.long) # 这个到底是啥呀？ 能不能用在其他数据集上？
        self.hparams = cfg.hoi_hparams
        self.coefs = cfg.loss.coef
        
        
    def h2o_weight(self, dists):
        # NOTE: h2o -- penetration weights (up); contact_weights (down)
        w_dist_h = torch.ones([dists.shape[0], dists.shape[1]]).to(self.device)
        p_dist = self.hparams['th_penet']
        c_dist = self.hparams['th_contact']
        w_dist_con = (dists < c_dist) * (dists > p_dist) # inside as negative; outside as positive
        w_dist_pen = dists < p_dist
        weight = w_dist_h.clone()
        weight[w_dist_con] = self.hparams['weight_contact'] # less weights for contact verts
        weight[w_dist_pen] = self.hparams['weight_penet'] # more weights for penetration verts
        return weight
    
    def o2h_weight(self, dists, region):
        # NOTE: o2h -- conditioned region weights
        w_dist_o = torch.ones([dists.shape[0], dists.shape[1]]).to(self.device)
        w_region = region > 0.
        # import pdb; pdb.set_trace()
        weight = w_dist_o.clone()
        weight[w_region.squeeze(1)] = self.hparams['weight_region']
        return weight
    
    def forward(self, h2o_signed, h2o_signed_pred, o2h_signed, o2h_signed_pred, region):
        ## adaptive weight for penetration and contact verts
        
        # weight_o2h = self.dist_loss_weight(o2h_signed)
        weight_o2h = self.o2h_weight(o2h_signed, region)
        weight_h2o = self.h2o_weight(h2o_signed)
        # import pdb; pdb.set_trace()
        

        loss_dist_h = self.coefs['lambda_dist_h'] * (1 - self.coefs['kl_coef']) * torch.mean(torch.einsum('ij,ij->ij', torch.abs(h2o_signed_pred.abs() - h2o_signed.abs()), weight_h2o))
        loss_dist_o = self.coefs['lambda_dist_o'] * (1 - self.coefs['kl_coef']) * torch.mean(torch.einsum('ij,ij->ij', torch.abs(o2h_signed_pred - o2h_signed), weight_o2h))
        return loss_dist_h, loss_dist_o
    
class PointCloudCompletionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.coarse_loss = ChamferDistanceL2Loss()
        self.fine_loss = ChamferDistanceL2Loss()
        
    def forward(self, coarse_pc, fine_pc, gt_pc, dict_loss, param_coarse=1000, param_fine=1000):
        loss_coarse = self.coarse_loss(coarse_pc, gt_pc)
        loss_fine = self.fine_loss(fine_pc, gt_pc)
        
        dict_loss['coarse_loss'] = loss_coarse * param_coarse
        dict_loss['fine_loss'] = loss_fine * param_fine
        return dict_loss
        
class MPMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.pc_loss = ChamferDistanceL2Loss()
        
    def forward(self, pred_pc, gt_pc, dict_loss):
        dict_loss['recon_chamfer_loss'] = self.pc_loss(pred_pc, gt_pc)
        
        return dict_loss


class ChamferDistanceL2Loss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pc_in, pc_gt):
        """_summary_

        Args:
            pc_in (): input point cloud
            pc_gt (): target ground truth point cloud
        """
        B, N1, _ = pc_in.size()
        _, N2, _ = pc_gt.size()
        cham_x, cham_y, _, _ = ch_dist(pc_in, pc_gt)
        # import pdb; pdb.set_trace()
        loss = torch.mean(cham_x) + torch.mean(cham_y)
        
        return loss




if __name__ == "__main__":
    loss = ChamferDistanceL2Loss()
    x = torch.randn(8, 32, 3)
    y = torch.randn(8, 32, 3)
    
    # import pdb; pdb.set_trace()
    out = loss.forward(x, y)

