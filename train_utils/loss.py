import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from option import MyOptions as cfg
from utils.utils import point2point_signed
from pytorch3d.structures import Meshes
import chamfer_distance as chd



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
    def __init__(self, rh_f, rh_model, device):
        super(cGraspvaeLoss, self).__init__()
        self.rh_f = rh_f.transpose(2,1)
        self.rh_model = rh_model
        self.device = device
        self.LossL1 = torch.nn.L1Loss(reduction='mean')
        self.v_weights = torch.from_numpy(np.load(cfg.c_weights_path)).to(torch.float32).to(self.device)
        self.v_weights2 = torch.pow(self.v_weights, 1.0/2.5)
        self.vpe = torch.from_numpy(np.load(cfg.vpe_path)).to(self.device).to(torch.long)

    def edges_for(self, x, vpe):
        return (x[:, vpe[:, 0]] - x[:, vpe[:, 1]])

    def dist_loss(self, h2o, h2o_pred, o2h_signed, o2h_signed_pred, region):
        ## adaptive weight for penetration and contact verts
        
        self.w_dist = torch.ones([h2o.shape[0], cfg.num_obj_verts]).to(self.device)
        p_dist = cfg.th_penet
        c_dist = cfg.th_contact
        w_dist = (o2h_signed < c_dist) * (o2h_signed > p_dist) # inside as negative; outside as positive
        w_dist_neg = o2h_signed_pred < 0.
        weight = self.w_dist.clone()
        weight[~w_dist] = cfg.weight_contact # less weights for contact verts
        weight[w_dist_neg] = cfg.weight_penet # more weights for penetration verts
        # conditioned region weights
        w_region = region > 0.
        # import pdb; pdb.set_trace()
        weight[w_region.squeeze(1)] = cfg.weight_region

        loss_dist_h = cfg.lambda_dist_h * torch.mean(torch.einsum('ij,j->ij', torch.abs(h2o_pred.abs() - h2o.abs()), self.v_weights2))
        loss_dist_o = cfg.lambda_dist_o * torch.mean(torch.einsum('ij,ij->ij', torch.abs(o2h_signed_pred - o2h_signed), weight))
        return loss_dist_h, loss_dist_o

    def KLLoss(self, rhand_vs, p_mean, p_std):
        device = rhand_vs.device
        dtype = rhand_vs.dtype
        # import pdb; pdb.set_trace()
        B = rhand_vs.size(0)
        q_z = torch.distributions.normal.Normal(p_mean, F.softplus(p_std)) # why we have softplus here: turn the negative std components to positive

        # q_z = torch.distributions.normal.Normal(p_mean, torch.exp(0.5 * p_std))
        p_z = torch.distributions.normal.Normal(
            loc=torch.tensor(np.zeros([B, cfg.VAE_enc_out_size]), requires_grad=False).to(device).to(dtype),
            scale=torch.tensor(np.ones([B, cfg.VAE_enc_out_size]), requires_grad=False).to(device).to(dtype)
        )
        loss_kl = cfg.kl_coef * torch.mean(torch.sum(torch.distributions.kl.kl_divergence(q_z, p_z), dim=[1]))

        return loss_kl

    def forward(self, hand_params, sample_stats, obj_vs, rhand_vs, region):

        B = rhand_vs.size(0)

        if B == cfg.batch_size:
            rhand_pred = self.rh_model(**hand_params)
        else:
            import mano
            rh_model = mano.load(model_path=cfg.mano_rh_path,
                                    model_type='mano',
                                      num_pca_comps=45,
                                      batch_size=B,
                                      flat_hand_mean=True)
            rh_model = rh_model.to(self.device)
            rhand_pred = rh_model(**hand_params)
        rhand_vs_pred = rhand_pred.vertices
        rhand_vs = rhand_vs.transpose(2, 1)
        obj_vs = obj_vs.transpose(2, 1)

        # import pdb; pdb.set_trace()

        rh_mesh_pred = Meshes(verts=rhand_vs_pred, faces=self.rh_f).to(self.device).verts_normals_packed().view(-1, 778, 3) # packed representation of the vertex normals
        rh_mesh = Meshes(verts=rhand_vs, faces=self.rh_f).to(self.device).verts_normals_packed().view(-1, 778, 3)

        # import pdb; pdb.set_trace()

        o2h_signed, h2o, o_nearest_ids = point2point_signed(rhand_vs, obj_vs, rh_mesh)
        o2h_signed_pred, h2o_pred, o_nearest_ids_pred = point2point_signed(rhand_vs_pred, obj_vs, rh_mesh_pred)

        #### dist Loss ####
        loss_dist_h, loss_dist_o = self.dist_loss(h2o, h2o_pred, o2h_signed, o2h_signed_pred, region)

        #### KL Loss ####
        if sample_stats is not None: 
            p_mean, p_std, Zin = sample_stats
            loss_kl = self.KLLoss(rhand_vs, p_mean, p_std)
        else:
            loss_kl = torch.tensor(0.0, dtype=float).to(self.device)

        #### verts Loss ####
        loss_mesh_rec_w = cfg.lambda_mesh_rec_w * torch.mean(torch.einsum('ijk,j->ijk', torch.abs((rhand_vs - rhand_vs_pred)), self.v_weights))

        #### edge Loss ####
        loss_edge = cfg.lambda_edge * self.LossL1(self.edges_for(rhand_vs_pred, self.vpe), self.edges_for(rhand_vs, self.vpe))

        dict_loss = {'loss_kl': loss_kl,
                     'loss_edge': loss_edge,
                     'loss_mesh_rec': loss_mesh_rec_w,
                     'loss_dist_h': loss_dist_h,
                     'loss_dist_o': loss_dist_o
                     }
        
        loss_total = torch.stack(list(dict_loss.values())).sum()

        signed_dists = [o2h_signed_pred, o2h_signed]
        
        return loss_total, dict_loss, signed_dists


        




if __name__ == "__main__":
    loss = ConditionNetLoss()
    

