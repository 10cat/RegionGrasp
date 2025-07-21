import sys

sys.path.append('.')
sys.path.append('..')
import config
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from chamfer_distance import ChamferDistance as ch_dist
from dataset.data_utils import faces2verts_no_rep
from pytorch3d.structures import Meshes
# from option import MyOptions as cfg
from utils.utils import (decode_hand_params_batch, edges_for, get_NN, get_std,
                         point2point_signed)


class cGraspvaeLoss(nn.Module):
    def __init__(self, device, cfg):
        super(cGraspvaeLoss, self).__init__()
        self.device = device
        self.LossL1 = torch.nn.L1Loss(reduction='mean')
        if cfg.dloss_type == 'w_o2honly':
            self.HOILoss = HOILoss_o2honly(cfg, device)
        else:
            self.HOILoss = HOILoss(cfg, device)
            
        self.PenetrLoss = inter_penetr_loss()
        self.v_weights = torch.from_numpy(np.load(cfg.c_weights_path)).to(torch.float32).to(self.device) # 这个到底是啥呀？ 能不能用在其他数据集上？
        self.v_weights2 = torch.pow(self.v_weights, 1.0/2.5) # 这个到底是啥呀？ 能不能用在其他数据集上？
        self.vpe = torch.from_numpy(np.load(cfg.vpe_path)).to(self.device).to(torch.long) # 这个到底是啥呀？ 能不能用在其他数据集上？
        
        cam_extr = np.array([[1., 0., 0., 0.], [0., -1., 0., 0.],
                                  [0., 0., -1., 0.]]).astype(np.float32)
        self.cam_extr = torch.from_numpy(cam_extr[:3, :3]).unsqueeze(0)
        
        self.cfg = cfg
        self.latent_size = cfg.model.vae.kwargs.latent_size
        self.batch_size = cfg.batch_size
        self.mano_rh_path = cfg.mano_rh_path
        self.coefs = cfg.loss.coef
        
        import chamfer_distance as chd
        self.chd = chd.ChamferDistance()
        

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


    def forward(self, hand_params, sample_stats, obj_vs, rhand_vs, region, trans=None, cam_extr=None, obj_normals=None, gt_hand_params=None, obj_mesh_faces=None, mode=None):


        B = rhand_vs.size(0)
        
        rhand_pred, rh_model = decode_hand_params_batch(hand_params, B, self.cfg, self.device)
        
        rhand_vs_pred = rhand_pred.vertices
        
        if self.cfg.tta:
            cam_extr = self.cam_extr.repeat(B, 1, 1).to('cuda')
            rhand_vs_pred = torch.matmul(rhand_vs_pred, cam_extr.transpose(1, 2))
        
        
        if self.cfg.use_mano and self.cfg.dataset.name == 'obman':
            assert trans is not None and cam_extr is not None
            cam_extr = cam_extr.to(self.device)
            trans = trans.to(self.device)
            # rhand_vs_pred = cam_extr.dot(rhand_vs_pred.transpose(2, 1)).transpose(2, 1)
            rhand_vs_pred = torch.bmm(cam_extr, rhand_vs_pred.transpose(2, 1)).transpose(2, 1)
            # import pdb; pdb.set_trace()
            rhand_vs_pred -= trans.unsqueeze(1)
            # CHECK: 此处如果写成rhand_vs_pred = rhand_vs_pred - trans.unsqueeze(1), 则在计算knn_points时会报错：RuntimeError: Expected tensor for argument #1 'p1' to have the same type as tensor for argument #2 'p2'; but type torch.cuda.DoubleTensor does not equal torch.cuda.FloatTensor (while checking arguments for KNearestNeighborIdxCuda)
        
        
        if rhand_vs.shape[-1] != 3:
            rhand_vs = rhand_vs.transpose(2, 1)
        rhand_vs = rhand_vs.to(self.device)
        
        if obj_vs.shape[-1] != 3:
            obj_vs = obj_vs.transpose(2, 1)
        obj_vs = obj_vs.to(self.device)
        
        faraway_origin = -50*torch.ones(region.shape)* (1 - region)
        faraway_origin = faraway_origin.to(self.device)
        region = region.to(self.device)
        rh_f_single = torch.from_numpy(rh_model.faces.astype(np.int32)).view(1, -1, 3)
        rhand_faces = rh_f_single.repeat(B, 1, 1).to(self.device).to(torch.long)
        # import pdb; pdb.set_trace() # to check decode module
        
        
        dict_loss = {}
        loss_cfg = self.cfg.loss[mode]
        #### dist Loss ####
        if loss_cfg.loss_dist_h or loss_cfg.loss_dist_o or (loss_cfg.get('metrics_cond') and loss_cfg.metrics_cond):
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
            obj_normals = obj_normals.to(torch.float32).to(self.device) if obj_normals is not None else None
            o2h_signed, h2o_signed, o2h_vid, h2o_vid = point2point_signed(rhand_vs, obj_vs, rh_normals, obj_normals)
            o2h_signed_pred, h2o_signed_pred, o2h_vid_pred, h2o_vid_pred = point2point_signed(rhand_vs_pred, obj_vs, rh_normals_pred, obj_normals)
            loss_dist_h, loss_dist_o = self.dist_loss(h2o_signed, h2o_signed_pred, o2h_signed, o2h_signed_pred, region)
            
            if loss_cfg.loss_dist_h:
                dict_loss.update({'loss_dist_h': loss_dist_h})
            if loss_cfg.loss_dist_o:
                dict_loss.update({'loss_dist_o': loss_dist_o})
            if loss_cfg.get('metrics_cond') and loss_cfg.metrics_cond:
                dict_loss['cond_contact'] = None
                dict_loss['cond_region'] = None
                
                # thumb_indices = torch.Tensor(config.bigfinger_vertices).reshape(1, -1).repeat(B, 1).to(self.device)
                
                # thumb_prox_ids = torch.gather(h2o_vid_pred, dim=-1, index=thumb_indices.to(torch.long)).unsqueeze(-1).repeat(1, 1, 3)
                # # thumb_prox_points = obj_vs[:, thumb_prox_ids]
                # thumb_prox_points = torch.gather(obj_vs, dim=1, index=thumb_prox_ids.to(torch.long))
                # # obj_vids = torch.arange(0, obj_vs.shape[0])
                
                # # import pdb; pdb.set_trace()
                #d region_points = obj_vs * (region.unsqueeze(-1)) + faraway_origin.unsqueeze(-1)
                # _, t2r_signed, _, _ = self.chd(region_points, thumb_prox_points)
                # hit_flag = t2r_signed.abs() < 0.00001
                # # import pdb; pdb.set_trace()
                # cond_hit_rate = (t2r_signed[hit_flag].shape[0] / t2r_signed.reshape(-1).shape[0])
                
                # dict_loss.update({'metrics_cond': torch.Tensor([cond_hit_rate]).to(torch.float32)[0].to(self.device)})
                
                # TODO: ------- new thumb condition hit rate with thumb pulp ------ #
                thumb_pulp_indices = torch.Tensor(faces2verts_no_rep(rh_f_single.squeeze(0)[config.thumb_center]))
                thumb_pulp_indices = thumb_pulp_indices.view(1, -1).repeat(B, 1).to(self.device)
                
                # import pdb; pdb.set_trace() #check: dims
                dists = torch.gather(h2o_signed_pred, dim=-1, index=thumb_pulp_indices.to(torch.long))
                thumb_prox_ids = torch.gather(h2o_vid_pred, dim=-1, index=thumb_pulp_indices.to(torch.long)).unsqueeze(-1).repeat(1, 1, 3)
                
                thumb_prox_points = torch.gather(obj_vs, dim=1, index=thumb_prox_ids.to(torch.long))
                region_points = obj_vs * (region.unsqueeze(-1)) + faraway_origin.unsqueeze(-1)
                _, t2r_signed, _, _ = self.chd(region_points, thumb_prox_points)
                
                hit_flag_in_contact = (dists > self.cfg.hoi_hparams['th_penet']) & (dists < self.cfg.hoi_hparams['th_contact']) & (t2r_signed.abs() < 1e-5)
                hit_in_contact = torch.zeros_like(thumb_pulp_indices).to(self.device)
                hit_in_contact[hit_flag_in_contact] = 1.
                cond_hit_rate_in_contact = hit_in_contact.sum() / hit_in_contact.reshape(-1).shape[0]
                
                hit_flag_in_region = (t2r_signed.abs() < 1e-5)
                hit_in_region = torch.zeros_like(thumb_pulp_indices).to(self.device)
                hit_in_region[hit_flag_in_region] = 1.
                cond_hit_rate_in_region = hit_in_region.sum() / hit_in_region.reshape(-1).shape[0]
                
                dict_loss.update({'cond_contact': torch.Tensor([cond_hit_rate_in_contact]).to(torch.float32)[0].to(self.device)})
                dict_loss.update({'cond_region': torch.Tensor([cond_hit_rate_in_region]).to(torch.float32)[0].to(self.device)})
                
                
        else:
            o2h_signed_pred, o2h_signed, h2o_signed, h2o_signed_pred = None, None, None, None
            
        #### penetration loss ####
        # if loss_cfg.get('loss_penetr') and loss_cfg.loss_penetr:
        #     obj_nn_dist_recon, obj_nn_idx_recon = get_NN(obj_vs, rhand_vs_pred)
        #     loss_penetr = self.coefs['lambda_penetr'] * self.PenetrLoss(rhand_vs_pred, rhand_faces, obj_vs,
        #                                 obj_nn_dist_recon, obj_nn_idx_recon)
        #     # import pdb; pdb.set_trace()
        #     dict_loss.update({'loss_penetr': loss_penetr})
            

        #### KL Loss ####
        if sample_stats is not None and mode != 'test': 
            p_mean, log_vars, Zin = sample_stats
            loss_kl = self.KLLoss(rhand_vs, p_mean, log_vars)
        else:
            # validation / test中loss_kl = 0
            loss_kl = torch.tensor(0.0, dtype=float).to(self.device)
        dict_loss.update({'loss_kl': loss_kl})

        #### verts Loss ####
        if loss_cfg.loss_mesh_rec:
            
            loss_mesh_rec = self.coefs['lambda_mesh_rec'] * (1 - self.coefs['kl_coef']) * torch.mean(torch.einsum('ijk,j->ijk', torch.abs((rhand_vs - rhand_vs_pred)), self.v_weights2)) # should be v_weights2 instead of v_weights
            dict_loss.update({'loss_mesh_rec': loss_mesh_rec})

        #### edge Loss ####
        if loss_cfg.loss_edge:
            loss_edge = self.coefs['lambda_edge'] * (1 - self.coefs['kl_coef']) * self.LossL1(edges_for(rhand_vs_pred, self.vpe), edges_for(rhand_vs, self.vpe))
            dict_loss.update({'loss_edge': loss_edge})
            
            
        if loss_cfg.loss_mano:
            recon_params = torch.cat([hand_params['global_orient'], hand_params['hand_pose'], hand_params['transl']], dim=1)
            gt_hand_params = gt_hand_params.to(self.device)
            loss_mano = self.coefs['lambda_mano'] * (1 - self.coefs['kl_coef']) * F.mse_loss(recon_params, gt_hand_params).sum() / recon_params.size(0)
            dict_loss.update({'loss_mano': loss_mano})
        
        # import pdb; pdb.set_trace()
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
        weight[~w_dist_con] = self.hparams['weight_contact'] # less weights for contact verts
        weight[w_dist_pen] = self.hparams['weight_penet'] # more weights for penetration verts
        return weight
    
    def o2h_weight(self, dists, region):
        # NOTE: o2h -- conditioned region weights
        w_dist_o = torch.ones([dists.shape[0], dists.shape[1]]).to(self.device)
        w_region = region > 0.
        # import pdb; pdb.set_trace()
        weight = w_dist_o.clone()
        weight[w_region.squeeze(1)] = self.hparams['weight_region']
        # p_dist = self.hparams['th_penet']
        # c_dist = self.hparams['th_contact']
        # w_dist_con = (dists < c_dist) * (dists > p_dist) # inside as negative; outside as positive
        # w_dist_pen = dists < p_dist
        # weight = w_dist_o.clone()
        # weight[w_dist_con] = self.hparams['weight_contact'] # less weights for contact verts
        # weight[w_dist_pen] = self.hparams['weight_penet'] # more weights for penetration verts
        return weight
    
    def forward(self, h2o_signed, h2o_signed_pred, o2h_signed, o2h_signed_pred, region):
        ## adaptive weight for penetration and contact verts
        
        # weight_o2h = self.dist_loss_weight(o2h_signed)
        weight_o2h = self.o2h_weight(o2h_signed, region)
        weight_h2o = self.h2o_weight(h2o_signed)
        # import pdb; pdb.set_trace()
        
        loss_dist_h = self.coefs['lambda_dist_h'] * (1 - self.coefs['kl_coef']) * torch.mean(torch.einsum('ij,ij->ij', torch.abs(h2o_signed_pred.abs() - h2o_signed.abs()), weight_h2o))
        # loss_dist_h = self.coefs['lambda_dist_h'] * (1 - self.coefs['kl_coef']) * torch.mean(torch.einsum('ij,ij->ij', torch.abs(h2o_signed_pred - h2o_signed), weight_h2o))
        # loss_dist_h = self.coefs['lambda_dist_h'] * (1 - self.coefs['kl_coef']) * torch.mean(torch.einsum('ij,j->ij', torch.abs(h2o_signed_pred.abs() - h2o_signed.abs()), self.v_weights2))
        
        loss_dist_o = self.coefs['lambda_dist_o'] * (1 - self.coefs['kl_coef']) * torch.mean(torch.einsum('ij,ij->ij', torch.abs(o2h_signed_pred - o2h_signed), weight_o2h))
        return loss_dist_h, loss_dist_o
    
    
class HOILoss_o2honly(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.device = device
        self.v_weights = torch.from_numpy(np.load(cfg.c_weights_path)).to(torch.float32).to(self.device) # 这个到底是啥呀？ 能不能用在其他数据集上？
        self.v_weights2 = torch.pow(self.v_weights, 1.0/2.5) # 这个到底是啥呀？ 能不能用在其他数据集上？
        self.vpe = torch.from_numpy(np.load(cfg.vpe_path)).to(self.device).to(torch.long) # 这个到底是啥呀？ 能不能用在其他数据集上？
        self.hparams = cfg.hoi_hparams
        self.coefs = cfg.loss.coef
    
    def o2h_weight(self, dists, region):
        # NOTE: o2h -- conditioned region weights
        w_dist_o = torch.ones([dists.shape[0], dists.shape[1]]).to(self.device)
        # w_region = region > 0.
        # import pdb; pdb.set_trace()
        # weight = w_dist_o.clone()
        # weight[w_region.squeeze(1)] = self.hparams['weight_region']
        p_dist = self.hparams['th_penet']
        c_dist = self.hparams['th_contact']
        w_dist_con = (dists < c_dist) * (dists > p_dist) # inside as negative; outside as positive
        w_dist_pen = dists < 0.
        weight = w_dist_o.clone()
        weight[~w_dist_con] = self.hparams['weight_contact'] # less weights for contact verts
        weight[w_dist_pen] = self.hparams['weight_penet'] # more weights for penetration verts
        return weight
    
    def forward(self, h2o_signed, h2o_signed_pred, o2h_signed, o2h_signed_pred, region):
        ## adaptive weight for penetration and contact verts
        
        # weight_o2h = self.dist_loss_weight(o2h_signed)
        weight_o2h = self.o2h_weight(o2h_signed, region)
        # weight_h2o = self.h2o_weight(h2o_signed)
        # import pdb; pdb.set_trace()
        
        loss_dist_h = self.coefs['lambda_dist_h'] * (1 - self.coefs['kl_coef']) * torch.mean(torch.einsum('ij,j->ij', torch.abs(h2o_signed_pred.abs() - h2o_signed.abs()), self.v_weights2))
        
        loss_dist_o = self.coefs['lambda_dist_o'] * (1 - self.coefs['kl_coef']) * torch.mean(torch.einsum('ij,ij->ij', torch.abs(o2h_signed_pred - o2h_signed), weight_o2h))
        return loss_dist_h, loss_dist_o
    

class batched_index_select():
    def __init__(self):
        return
    
    def __call__(self, input, index, dim=1):
        views = [input.size(0)] + [1 if i != dim else -1 for i in range(1, len(input.shape))]
        expanse = list(input.shape)
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return index

class get_interior():
    def __init__(self):
        self.batched_index_select = batched_index_select()
        return
    
    def __call__(self, src_face_normal, src_xyz, trg_xyz, trg_NN_idx):
        N1, N2 = src_xyz.size(1), trg_xyz.size(1)

        # get vector from trg xyz to NN in src, should be a [B, 3000, 3] vector
        NN_src_xyz = self.batched_index_select(src_xyz, trg_NN_idx)  # [B, 3000, 3]
        NN_vector = NN_src_xyz - trg_xyz  # [B, 3000, 3]

        # get surface normal of NN src xyz for every trg xyz, should be a [B, 3000, 3] vector
        NN_src_normal = self.batched_index_select(src_face_normal, trg_NN_idx)

        interior = (NN_vector * NN_src_normal).sum(dim=-1) > 0  # interior as true, exterior as false
        return interior


class inter_penetr_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.get_interior = get_interior()
    
    def forward(self, hand_xyz, hand_face, obj_xyz, nn_dist, nn_idx):
        '''
        get penetrate object xyz and the distance to its NN
        :param hand_xyz: [B, 778, 3]
        :param hand_face: [B, 1538, 3], hand faces vertex index in [0:778]
        :param obj_xyz: [B, 3000, 3]
        :return: inter penetration loss
        '''
        B = hand_xyz.size(0)
        mesh = Meshes(verts=hand_xyz, faces=hand_face)
        hand_normal = mesh.verts_normals_packed().view(-1, 778, 3)

        # if not nn_dist:
        #     nn_dist, nn_idx = utils_loss.get_NN(obj_xyz, hand_xyz)
        interior = self.get_interior(hand_normal, hand_xyz, obj_xyz, nn_idx).type(torch.bool)  # True for interior
        penetr_dist = nn_dist[interior].sum() / B  # batch reduction
        return 100.0 * penetr_dist
    
    
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
        self.pc_loss_cham = ChamferDistanceL2Loss()
        
    def forward(self, pred_pc, gt_pc, dict_loss):
        # dict_loss['recon_centers_loss'] = self.pc_loss_cham(pred_centers, gt_centers)
        dict_loss['recon_loss'] = self.pc_loss_cham(pred_pc, gt_pc)
        
        return dict_loss


class ChamferDistanceL2Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cham_dist = ch_dist()
    def forward(self, pc_in, pc_gt):
        """_summary_

        Args:
            pc_in (): input point cloud
            pc_gt (): target ground truth point cloud
        """
        B, N1, _ = pc_in.size()
        _, N2, _ = pc_gt.size()
        cham_x, cham_y, _, _ = self.cham_dist(pc_in, pc_gt)
        # import pdb; pdb.set_trace()
        loss = torch.mean(cham_x) + torch.mean(cham_y)
        
        return loss




if __name__ == "__main__":
    loss = ChamferDistanceL2Loss()
    x = torch.randn(8, 32, 3)
    y = torch.randn(8, 32, 3)
    
    # import pdb; pdb.set_trace()
    out = loss.forward(x, y)

