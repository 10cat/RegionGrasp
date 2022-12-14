import os
import sys
sys.path.append('.')
sys.path.append('..')
from option import MyOptions as cfg
import numpy as np
import torch 
import torch.nn as nn
import trimesh
from tqdm import tqdm
from utils.utils import dataset_object_faces_batch, point2point_signed, makepath, signed_distance_batch
from pytorch3d.structures import Meshes
from traineval_utils import simulation, interpenetraion, contact, condition
from utils.meters import AverageMeter, AverageMeters

to_dev = lambda tensor, device: tensor.to(device)
# to_gpu = lambda tensor: tensor.to('cuda')
to_cpu_np = lambda tensor: tensor.detach().cpu().numpy()

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
        
        # DONE take batchmean: means of the penetration depths in this batch
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
    
class TestMetricsCPU():
    """
    不是batch computation，而是整个数据集串行处理的方式进行的
    """
    def __init__(self, rh_model, dataset):
        self.rh_model = rh_model
        self.rh_faces = self.rh_model.faces
        self.rh_faces_torch = torch.from_numpy(self.rh_faces.astype(np.int32)).view(1, -1, 3)
        self.dataset = dataset
        self.metrics_names = {'CA':0, 'IV':1, 'simulation':2, 'condition':3}
        # import pdb; pdb.set_trace()
        # DONE: 根据cfg中的参数设定构造bool list决定计算的metrics种类
        self.test_metrics = [cfg.metrics_contact, cfg.metrics_inter, cfg.metrics_simul, cfg.metrics_cond]
        
    def contact_area(self, sample_info, signed_dists):
        area = contact.get_contact_area(sample_info, signed_dists)
        return area
    
    def interpenetration(self, sample_info):
        volume = interpenetraion.main(sample_info)
        return volume
    
    def simulation(self, sample_info):
        save_gif_folder = os.path.join(cfg.output_dir, 'gt_sim', 'gif')
        makepath(save_gif_folder)
        save_obj_folder = os.path.join(cfg.output_dir, 'gt_sim', 'obj')
        makepath(save_obj_folder)
        
        sim_dist = simulation.main(sample_idx=sample_info['index'], 
                                   sample=sample_info, 
                                   save_gif_folder=save_gif_folder, 
                                   save_obj_folder=save_obj_folder)
        
        return sim_dist
    
    def condition(self, sample_info, signed_dists, h_nearest_faces=None):
        return condition.main(sample_info, signed_dists, h_nearest_faces)
    
    def signed_distance(self, hand_mesh, obj_mesh):
        signed_dists = trimesh.proximity.signed_distance(obj_mesh, hand_mesh.vertices)
        return signed_dists
    
    def sample_metrics(self, sample_info, metrics):
        # import pdb; pdb.set_trace()
        # DONE: 通过sample_info构造HandMesh, ObjMesh
        HandMesh = trimesh.Trimesh(vertices=sample_info['hand_verts'], faces=sample_info['hand_faces'])
        ObjMesh = trimesh.Trimesh(vertices=sample_info['obj_verts'], faces=sample_info['obj_faces'])
        # import pdb; pdb.set_trace()
        # DONE: 对于每个sample使用trimesh.proximity.signed_distance计算过于慢了 => 改用事先使用gpu计算好的batch
        # signed_dists = self.signed_distance(HandMesh, ObjMesh)
        signed_dists = sample_info['h2o_dists']
        h_nearest_faces = sample_info['h_nearest_ids']
        # import pdb; pdb.set_trace()
        # DONE: 根据定义的bool list -- self.test_metrics决定计算哪些metrics
        names = self.metrics_names
        if self.test_metrics[names['CA']]: # -- 0
            CA = self.contact_area(sample_info, signed_dists)
            metrics.add_value('CA', CA)
        if self.test_metrics[names['IV']]: # -- 1
            IV = self.interpenetration(sample_info)
            metrics.add_value('IV', IV)
        if self.test_metrics[names['condition']]: # -- 3
            hit_success, coverage = self.condition(sample_info, signed_dists, h_nearest_faces)
            metrics.add_value('hit', hit_success)
            metrics.add_value('coverage', coverage)
        if self.test_metrics[names['simulation']]: # -- 2
            sim_dist = self.simulation(sample_info)
            metrics.add_value('sim_dist', sim_dist)
            
        
        return metrics
        
    def __call__(self, rh_verts_pred_batch, data):
        # NOTE: [<-epochbase] rhand_vs_pred -- 
        # NOTE: [<-epochbase] data -- 'region_mask', 'region_centers', 'verts_obj', 'verts_rhand', 'sample_ids'
        # DONE: 由于希望在gpu上计算signed_distance加快计算速度， 因此先不着急to_cpu, 需要先在gpu上转置做batch operation预备工作
        obj_verts_batch = data['verts_obj'].transpose(2,1)
        region_mask_batch = data['region_mask'].transpose(2,1)
        sample_idx_batch = data['sample_ids']
        device = obj_verts_batch.device
        # DONE: 索引batch对应的object_mesh_faces
        obj_mesh_faces = dataset_object_faces_batch(sample_idx_batch, self.dataset, device)
        # import pdb; pdb.set_trace()
        # DONE: generate corresponding batch faces
        B = obj_verts_batch.shape[0]
        rh_faces_batch = self.rh_faces_torch.repeat(B, 1, 1).to(device).to(torch.long)
        # DONE 批处理计算signed distance
        _, h2o_signed_dists, _, h_nearest_ids = signed_distance_batch(device, rh_verts_pred_batch, rh_faces_batch, obj_verts_batch,object_faces=obj_mesh_faces)
        # import pdb; pdb.set_trace()
        
        # DONE：进行对于输入数据to_cpu和转numpy处理
        # DONE 新生成的h2o_signed_dists和h_nearest_ids也加上
        rh_verts_pred_batch = to_cpu_np(rh_verts_pred_batch)
        obj_verts_batch = to_cpu_np(obj_verts_batch)
        region_mask_batch = to_cpu_np(region_mask_batch)
        region_centers_batch = to_cpu_np(data['region_centers'])
        h2o_signed_dists_batch = to_cpu_np(h2o_signed_dists)
        h_nearest_ids_batch = to_cpu_np(h_nearest_ids)
        
        # import pdb; pdb.set_trace()
        # DONE: 输入batch的尺寸是否正确
        batch_size = obj_verts_batch.shape[0]
        batch_metrics_meters = AverageMeters()
        
        # DONE:推理速度太慢了 => 根据pdb调试情况，主要的时间花在计算signed distance上， 考虑是否能用pytorch3d进行批处理计算signed distance
        # DONE: 实现之后只能够提升1.3倍左右，看来在validation阶段还是只能截取部分进行测试
        
        for idx in range(batch_size):
            if cfg.test_part and idx > batch_size * cfg.select_k:
                break
            sample_idx = int(sample_idx_batch[idx])
            # NOTE: verts_rh_pred is output from mano forward, no need to transpose
            rh_verts_pred = rh_verts_pred_batch[idx]
            rh_faces = self.rh_faces
            obj_name = self.dataset.frame_objs[sample_idx]
            ObjMesh = self.dataset.object_meshes[obj_name]
            obj_verts = obj_verts_batch[idx]
            obj_faces = ObjMesh.faces
            
            # NOTE: [->contact/.../...] sample_info -- 'hand_verts', 'hand_faces', 'obj_verts', 'obj_faces', cond_center, cond_region_mask, h2o_dists, h_nearest_ids
            sample_info = {'hand_verts': rh_verts_pred, 'hand_faces':rh_faces, 
                           'obj_verts':obj_verts, 'obj_faces':obj_faces, 
                           'cond_center':region_centers_batch[idx], 'cond_region_mask':region_mask_batch[idx], 
                           'h2o_dists': h2o_signed_dists_batch[idx], 'h_nearest_ids':h_nearest_ids_batch[idx],
                           'index':sample_idx}
            
            batch_metrics_meters = self.sample_metrics(sample_info, batch_metrics_meters) 
       
        # NOTE: batch_metrics.average_meters不能直接回传；要对每一类提取均值之后变成普通字典； 这里用之前实现好的
        batch_metrics = batch_metrics_meters.avg(self.dataset.ds_name)
        return batch_metrics