import os
import sys
sys.path.append('.')
sys.path.append('..')
import pdb
import pickle
from tqdm import tqdm
import config

import numpy as np
import torch
import trimesh
from sklearn.neighbors import KDTree

from utils.utils import func_timer, makepath
from utils.visualization import colors_like
from dataset.data_utils import faces2verts_no_rep, contact_to_dict
from dataset.obman_orig import obman

class ObManResample(obman):
    def __init__(self, ds_root, shapenet_root, split='train', joint_nb=21, mini_factor=None, use_cache=False, root_palm=False, mode='all', segment=False, use_external_points=True, apply_obj_transform=True, expand_times=config.expand_times, resample_num = 8192):
        super().__init__(ds_root, shapenet_root, split, joint_nb, mini_factor, use_cache, root_palm, mode, segment, use_external_points, apply_obj_transform)
        self.unique_objs = np.unique(
                    [(meta_info['obj_class_id'], meta_info['obj_sample_id'])
                     for meta_info in self.meta_infos],
                    axis=0)
        
        self.resample_num = resample_num
        self.obj_resampled = self.resample_obj_mesh(N=resample_num)
        
        
        
    def resampling_surface(self, class_id, sample_id, N):
        obj = self.obj_meshes[class_id][sample_id]
        obj_mesh = trimesh.Trimesh(vertices=obj['vertices'], faces=obj['faces'])
        # areas = obj_mesh.area_faces
        # weight = 1 + areas 
        obj_xyz_resampled, face_id = trimesh.sample.sample_surface(obj_mesh, N)
        return obj_xyz_resampled, face_id
    
    def resample_obj_mesh(self, N):
        resampled = {}
        resampled_paths = []
        for pair in tqdm(self.unique_objs, desc=f"resampling objs to {N} points"):
            class_id, sample_id = pair[0], pair[1]
            resampled_points_path = os.path.join(self.shapenet_root, class_id, sample_id, 'models/resampled_'+str(N)+'.npy')
            resampled_faces_path = os.path.join(self.shapenet_root, class_id, sample_id, 'models/resampled_'+str(N)+'_face_id.npy')
            if os.path.exists(resampled_points_path) and not config.force_resample:
                obj_xyz_resampled = np.load(resampled_points_path)
                face_id = np.load(resampled_faces_path)
            else:
                obj_xyz_resampled, face_id = self.resampling_surface(class_id, sample_id, N)
                np.save(resampled_points_path, obj_xyz_resampled)
                np.save(resampled_faces_path, face_id)
            
            if class_id not in resampled:
                resampled[class_id] = {}
            resampled[class_id].update({sample_id: {'points':obj_xyz_resampled, 'faces':face_id}})
            
        return resampled
    
    
    def get_obj_resampled_trans(self, meta_infos, obj_transforms, idx, obj_centric=False):
        meta_info = meta_infos[idx]
        class_id, sample_id = meta_info['obj_class_id'], meta_info['obj_sample_id']
        # -- for resampled 
        obj_re = self.obj_resampled[class_id][sample_id]
        points = obj_re['points']
        face_ids = obj_re['faces']
        
        obj_transform = obj_transforms[idx]
        hom_points = np.concatenate([points, np.ones([points.shape[0], 1])], axis=1)
        transform_mat = self.cam_extr.dot(obj_transform)
        trans_points = transform_mat.dot(hom_points.T).T
        
        rotmat = transform_mat[:, :3]
        trans = transform_mat[:, -1]
        obj_trans = trans + points.mean(axis=0)
        if obj_centric:
            trans_points -= obj_trans
        
        return np.array(trans_points).astype(np.float32), trans

class ObManObj(ObManResample):
    def __init__(self, ds_root, shapenet_root, split='train', joint_nb=21, mini_factor=None, use_cache=False, root_palm=False, mode='all', segment=False, use_external_points=True, apply_obj_transform=True, expand_times=config.expand_times, resample_num=8192, object_centric=False):
        super().__init__(ds_root, shapenet_root, split, joint_nb, mini_factor, use_cache, root_palm, mode, segment, use_external_points, apply_obj_transform, expand_times, resample_num)
        self.meta_infos_exp, self.obj_transforms_exp = self.expand_set(expand_times)
        self.mask_centers, self.mask_Ks = self.get_mask_ratio()
        self.obj_centric = object_centric
        
    def transform_under_views(self):
        return
    
    def expand_set(self, times=config.expand_times):
        meta_infos_new = []
        obj_transforms_new = []
        for idx, meta_info in enumerate(self.meta_infos):
            for _ in range(times):
                meta_infos_new.append(meta_info)
                obj_transforms_new.append(self.obj_transforms[idx])
                
        return meta_infos_new, obj_transforms_new
    
    def get_mask_ratio(self, ratio_lb=config.ratio_lower, ratio_ub=config.ratio_upper, N = config.num_resample_points, Nm = config.num_mask_points, seed1=0, seed2=1024):
        np.random.seed(seed1)
        mask_centers = np.floor(np.random.random(self.__len__()) * N).astype(np.int32)
        # np.random.seed(seed2)
        # NOTE: 虽然训练阶段是需要固定点数的，但是pretrain阶段可以扩大mask掉的点数，可以一定程度上防止模型过拟合
        mask_Ks = np.round((ratio_lb + (ratio_ub - ratio_lb) * np.random.random(self.__len__())) * N).astype(np.int32)
        # mask_Ks = (Nm *np.ones_like(mask_centers)).astype(np.int32)
        return mask_centers, mask_Ks
    
    def KNNmask(self, points, idx):
        tree = KDTree(points)
        center_id = int(self.mask_centers[idx])
        K = self.mask_Ks[idx]
        # pdb.set_trace()
        distances, indices = tree.query(points[center_id].reshape(1, -1), K)
        
        mask = np.ones_like(points)
        indices = indices.reshape(-1).tolist()
        for index in indices:
            mask[index] = 0.
        return mask
    
    def input_pc_sample(self, idx, PC, M=2048):
        N = PC.shape[0]
        np.random.seed(idx+10)
        indices = np.floor(np.random.random(M)*N).astype(np.int32)
        sample_pc = [PC[int(index)] for index in indices]
        sample_pc = np.array(sample_pc)
        return sample_pc, indices
        
    def __len__(self):
        return len(self.meta_infos_exp)
    
    # @func_timer
    def __getitem__(self, idx):
        sample = {}
        obj_points, obj_trans = self.get_obj_resampled_trans(self.meta_infos_exp, self.obj_transforms_exp, idx, obj_centric=self.obj_centric)
        mask = self.KNNmask(obj_points, idx)
        masked = mask < 1
        remained_points = obj_points[~masked].reshape(-1, 3)
        mask_points = obj_points[masked].reshape(-1, 3)
        # import pdb; pdb.set_trace()
        remained_pc, sample_indices = self.input_pc_sample(idx, remained_points)
        sample['input_points'] = torch.from_numpy(remained_pc)
        sample['gt_points'] = torch.from_numpy(obj_points)
        sample['mask'] = torch.from_numpy(mask)
        sample['sample_id'] = torch.Tensor([idx])
        sample['obj_trans'] = torch.Tensor(obj_trans)
        
        return sample
    
class ObManThumb(ObManResample):
    def __init__(self, ds_root, shapenet_root, split='train', joint_nb=21, mini_factor=None, use_cache=False, root_palm=False, mode='all', segment=False, use_external_points=True, apply_obj_transform=True, expand_times=1, resample_num=8192, object_centric=False):
        super().__init__(ds_root, shapenet_root, split, joint_nb, mini_factor, use_cache, root_palm, mode, segment, use_external_points, apply_obj_transform, expand_times, resample_num)
        
        self.obj_centric = object_centric
        
    def thumb_query_point(self, HandMesh, ObjMesh, pene_th=0.002, contact_th=-0.005):
        thumb_vertices_ids = faces2verts_no_rep(HandMesh.faces[config.thumb_center])
        thumb_vertices = HandMesh.vertices[thumb_vertices_ids]
        ObjQuery = trimesh.proximity.ProximityQuery(ObjMesh)
        #  -- 以thumb_vertices_ids为query计算signed distances并返回相对应的closest faces
        # NOTE: the on_surface return is not signed_dists, 所以需要专门计算signed dists， 再用on_surface返回obj上最近的面
        h2o_signed_dists = ObjQuery.signed_distance(thumb_vertices)
        _, _, h2o_closest_fid = ObjQuery.on_surface(thumb_vertices)
        
        # -- 用sdf_th阈值进一步选取thumb上真正的contact部分
        # NOTE: OUTSIDE mesh -> NEG； INSIDE the mesh -> POS
        penet_flag = h2o_signed_dists < pene_th
        contact_flag = h2o_signed_dists > contact_th
        flag = penet_flag & contact_flag
        obj_contact_fids = h2o_closest_fid[flag]
        # import pdb; pdb.set_trace()
        if obj_contact_fids.shape[0] == 0:
            return None
        elif obj_contact_fids.shape[0] == 1:
            point = ObjMesh.triangles_center[obj_contact_fids[0]]
        else:
            # import pdb; pdb.set_trace()
            tri_centers = np.array([ObjMesh.triangles_center[fid] for fid in obj_contact_fids])
            # TODO mean of the tri_centers
            point =  np.mean(tri_centers, axis=0)
            
        return point
    
    
    def get_KNN_in_pc(self, PC, point_q, K=410):
        PC_tree = KDTree(PC)
        distance, indices = PC_tree.query(point_q.reshape(1, -1), K)
        distance = distance.reshape(-1)
        indices = indices.reshape(-1)
        return distance, indices
    
    def divide_pointcloud(self, PC, indices):
        N = PC.shape[0]
        indices_neg = list(set(range(N)) - set(indices))
        pc = [PC[i] for i in indices]
        rem_pc = [PC[i] for i in indices_neg]
        
        pc = np.array(pc)
        rem_pc = np.array(rem_pc)
        return pc, rem_pc
    
    def input_pc_sample(self, idx, PC, M=2048):
        N = PC.shape[0]
        np.random.seed(idx)
        indices = np.floor(np.random.random(M)*N).astype(np.int32)
        sample_pc = [PC[int(index)] for index in indices]
        sample_pc = np.array(sample_pc)
        return sample_pc, indices
    
    def __getitem__(self, idx):
        annot = {}
        ObjPoints, obj_trans = self.get_obj_resampled_trans(self.meta_infos, idx, obj_centric=self.obj_centric) # np(8192, 3)
        N = ObjPoints.shape[0]
        hand_verts = self.get_verts3d(idx)
        hand_faces = self.get_faces3d(idx)
        if self.obj_centric:
            hand_verts -= obj_trans
        obj_mesh = self.get_sample_obj_mesh(idx)
        obj_verts, _ = self.get_obj_verts_faces(idx)
        
        HandMesh = trimesh.Trimesh(vertices=hand_verts, faces=hand_faces)
        ObjMesh = trimesh.Trimesh(vertices=obj_verts, faces=obj_mesh['faces'])
        
        point_contact = self.thumb_query_point(HandMesh, ObjMesh)
        if point_contact is None:
            # TODO: 筛掉没有手接触的sample
            return None
        
        dists, contact_indices = self.get_KNN_in_pc(ObjPoints, point_contact)
        
        # import pdb; pdb.set_trace()
        
        contact_pc, input_pc_hr = self.divide_pointcloud(ObjPoints, contact_indices)
        
        # CHECK:region_visual --> 820有点太大了，先取一半吧410
        # PC_contact = trimesh.PointCloud(vertices=contact_ps, colors=colors_like(config.colors['yellow']))
        # PC_rem = trimesh.PointCloud(vertices=rem_ps, colors=colors_like(config.colors['green']))
        
        # PC_contact.export('test_pc_contact.ply')
        # PC_rem.export('test_pc_rem.ply')
        # ObjMesh.export('test_mesh.ply')
        # HandMesh.export('hand_mesh.ply')
        
        # import pdb; pdb.set_trace()
        
        # TODO: random sampling Np = 2048 from the rem_ps
        np.random.seed(idx)
        input_pc, sampled_indices = self.input_pc_sample(idx, ObjPoints)
        
        annot['contact_indices'] = contact_indices
        annot['contact_pc'] = contact_pc
        annot['input_pc_hr'] = input_pc_hr
        annot['input_pc'] = input_pc
        annot['sampled_indices'] = sampled_indices
        
        return annot
    

  
def sampling_check(objdataset):
    """
    研究采样方式和点数对于几何体所有面覆盖的情况

    Args:
        objdataset (dataset): obj dataset
    """
    unique_objs = objdataset.unique_objs
    ratio_avg = 0
    for pair in unique_objs:
        class_id, sample_id = pair[0], pair[1]
        obj = objdataset.obj_meshes[class_id][sample_id]
        obj_faces = obj['faces']
        obj_xyz_resampled, face_ids = objdataset.resampling_surface(class_id, sample_id, N=3000)
        num_face_orig = obj_faces.shape[0]
        num_face_sampled = len(set(face_ids))
        face_sample_ratio = num_face_sampled / num_face_orig
        ratio_avg += face_sample_ratio
        print(f"[{class_id}, {sample_id}] original face_num:{num_face_orig}; sampled face_num: {num_face_sampled}; ratio: {face_sample_ratio}")
        
    ratio_avg = ratio_avg / unique_objs.shape[0]
    print(f"avg ratio: {ratio_avg}")
    return

def get_thumb_condition(ds_root):
    dataset = ObManThumb(ds_root=ds_root, 
                           shapenet_root=config.SHAPENET_ROOT,
                           split='train',
                           use_cache=True)
    output_root = os.path.join(dataset.root, 'thumbHOI')
    makepath(output_root)
    samples_list = []
    for idx in tqdm(range(dataset.__len__())):
        annot = dataset.__getitem__(idx)
        if annot is None:
            continue
        output_path = os.path.join(output_root, f'{idx}.pkl')
        with open(output_path, 'wb') as fid:
            pickle.dump(annot, fid)
        samples_list.append(idx)
    
    list_path = os.path.join(output_root, 'samples_id.npy')
    np.save(list_path, np.array(samples_list))
    return

def get_new_objpretrain(ds_root):
    objdataset = ObManObj(ds_root=ds_root,
                          shapenet_root=config.SHAPENET_ROOT,
                          split='train',
                          use_cache=True,
                          expand_times=5,
                          object_centric=True)
    for idx in tqdm(range(objdataset.__len__())):
        sample = objdataset.__getitem__(idx)
    
    
if __name__ == "__main__":
    dataset_root = config.OBMAN_ROOT
    
    get_new_objpretrain(dataset_root)
    
    
    
    
        
        
    
    
    
        
    
    
        
    