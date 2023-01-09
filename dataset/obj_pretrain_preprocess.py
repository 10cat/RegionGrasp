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

from utils.utils import func_timer
from dataset.data_utils import contact_to_dict
from obman_preprocess import obman

class ObManObj(obman):
    def __init__(self, root, shapenet_root, split='train', joint_nb=21, mini_factor=None, use_cache=False, root_palm=False, mode='all', segment=False, use_external_points=True, apply_obj_transform=True):
        super().__init__(root, shapenet_root, split, joint_nb, mini_factor, use_cache, root_palm, mode, segment, use_external_points, apply_obj_transform)
        self.unique_objs = np.unique(
                    [(meta_info['obj_class_id'], meta_info['obj_sample_id'])
                     for meta_info in self.meta_infos],
                    axis=0)
        self.obj_resampled = self.resample_obj_mesh()
        self.meta_infos_expand = self.expand_set()
        self.mask_centers, self.mask_Ks = self.get_mask_ratio()
        
    def resampling_surface(self, class_id, sample_id, N):
        obj = self.obj_meshes[class_id][sample_id]
        obj_mesh = trimesh.Trimesh(vertices=obj['vertices'], faces=obj['faces'])  
        obj_xyz_resampled, face_id = trimesh.sample.sample_surface(obj_mesh, N)
        return obj_xyz_resampled, face_id
    
    def resample_obj_mesh(self, N=config.num_resample_points):
        resampled = {}
        resampled_paths = []
        for pair in tqdm(self.unique_objs, desc=f"resampling objs to {N} points"):
            class_id, sample_id = pair[0], pair[1]
            resampled_points_path = os.path.join(self.shapenet_root, class_id, sample_id, 'models/resampled_'+str(N)+'.npy')
            resampled_faces_path = os.path.join(self.shapenet_root, class_id, sample_id, 'models/resampled_'+str(N)+'_face_id.npy')
            if os.path.exists(resampled_points_path):
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
    
    def expand_set(self, times=config.expand_times):
        meta_infos_new = []
        for idx, meta_info in enumerate(self.meta_infos):
            for _ in range(times):
                meta_infos_new.append(meta_info)
                
        return meta_infos_new
    
    def get_mask_ratio(self, ratio_lb=config.ratio_lower, ratio_ub=config.ratio_upper, N = config.num_resample_points, seed1=0, seed2=1024):
        np.random.seed(seed1)
        mask_centers = np.round(np.random.random(self.__len__()) * N).astype(np.int32)
        np.random.seed(seed2)
        mask_Ks = np.round((ratio_lb + (ratio_ub - ratio_lb) * np.random.random(self.__len__())) * N).astype(np.int32)
        
        return mask_centers, mask_Ks
    
    
    def get_obj_resampled_trans(self, meta_infos, idx):
        meta_info = meta_infos[idx] # NOTE: use the expanded meta_infos list
        class_id, sample_id = meta_info['obj_class_id'], meta_info['obj_sample_id']
        # -- for resampled 
        obj_re = self.obj_resampled[class_id][sample_id]
        points = obj_re['points']
        face_ids = obj_re['faces']
        
        obj_transform = self.obj_transforms[idx]
        hom_points = np.concatenate([points, np.ones([points.shape[0], 1])], axis=1)
        trans_points = obj_transform.dot(hom_points.T).T[:, :3]
        trans_points = self.cam_extr[:3, :3].dot(trans_points.transpose()).transpose()
        
        return np.array(trans_points).astype(np.float32)
    
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
    
    def __len__(self):
        return len(self.meta_infos_expand)
    
    @func_timer
    def __getitem__(self, idx):
        sample = {}
        points = self.get_obj_resampled_trans(self.meta_infos_expand, idx)
        mask = self.KNNmask(points, idx)
        sample['points'] = torch.from_numpy(points)
        sample['mask'] = torch.from_numpy(mask)
        return sample
    
    
    
    
if __name__ == "__main__":
    dataset_root = config.OBMAN_ROOT
    
    objdataset = ObManObj(root=dataset_root, 
                           shapenet_root=config.SHAPENET_ROOT,
                           split='train',
                           use_cache=True)
    
    
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
        
    
    
        
    