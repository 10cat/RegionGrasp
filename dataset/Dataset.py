import os
import sys
sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch
import mano
from torch.utils import data
from torch.utils.data._utils.collate import default_collate
from mano.model import load
import trimesh
import config
from copy import deepcopy

import random
from utils.utils import func_timer, makepath
from dataset.Dataset_origin import GrabNetDataset_orig

set_seed = lambda val: np.random.seed(val)

def select_ids_dataset(ds_names, seeds=[]):
    for i, name in enumerate(ds_names):
        dataset = GrabNetDataset_orig(dataset_root=config.DATASET_ROOT, 
                                    ds_name=name, 
                                    frame_names_file='frame_names_thumb.npz', 
                                    grabnet_thumb=True, 
                                    obj_meshes_folder='decimate_meshes', 
                                    output_root=None, 
                                    dtype=torch.float32, 
                                    only_params=False, 
                                    load_on_ram=False)
        
        len = dataset.__len__()
        set_seed(seeds[i])
        select_ids_norm = np.random.random(len)
        np.save(os.path.join(dataset.ds_path, f'{name}_ids_norm.npy'), select_ids_norm)

class GrabNetDataset(GrabNetDataset_orig):
    def __init__(self, 
                 dataset_root, 
                 ds_name='train', 
                 frame_names_file='frame_names.npz', 
                 grabnet_thumb=False, 
                 obj_meshes_folder='contact_meshes',
                 select_ids = False, 
                 output_root=None, 
                 dtype=torch.float32, 
                 only_params=False, 
                 load_on_ram=False):
        super().__init__(dataset_root, ds_name, frame_names_file, grabnet_thumb, obj_meshes_folder, output_root, dtype, only_params, load_on_ram)
        
        frame_names_sdf_list = [os.path.join(self.ds_root, fname.replace('data', 'data_sdf').replace('.npz', '.npy')) for fname in self.frame_names_orig]
        self.frame_names_sdf = np.asarray(frame_names_sdf_list)
        
        if select_ids:
            self.select_ids_norm = np.load(os.path.join(self.ds_path, ds_name+'_ids_norm.npy'))
        else:
            self.select_ids_norm = None
            
        # for visualization
        self.region_face_ids = {} 
        self.region_center_ids = {}
        
    def region_mask_vertices(self, sample, obj_vertices, obj_faces, idx):
        # TODO: 改为只去thumb_center_ids对应的区域
        # -- 原'obj_faces_ids'键名改成了'candidate_centers'; 'thumb_center_ids'为thumb对应'candidate_centers'和'candidates'的索引号
        # -- val / test中需要指定新的编号并存储下来了
        candidates = sample['candidates']
        candidate_centers  = sample['candidate_centers']
        thumb_center_ids = sample['thumb_center_ids']
        # TODO: 为sample选取输入的condition mask
        if self.select_ids_norm is not None:
            # --- 固定选取id的情况
            select_id = int(self.select_ids_norm[idx] * len(thumb_center_ids))
            
        else:
            if self.ds_name == 'train':
                select_id = int(np.random.choice(len(thumb_center_ids)))
            else:
                raise ValueError("Suppose to select ids for val / test set")
        index = thumb_center_ids[select_id]
        region_center_id = candidate_centers[index]
        region_faces_ids = candidates[index]

        region_faces = obj_faces[region_faces_ids]
        region_vertices_ids = np.unique(region_faces.reshape(-1)).tolist()
        # import pdb; pdb.set_trace()
        region_mask_np = np.zeros_like(obj_vertices[:, :1], dtype=float)
        region_mask_np[region_vertices_ids] = 1.0

        # TODO: update the region face / center dict for visualization
        self.region_face_ids[idx] = region_faces_ids
        self.region_center_ids[idx] = region_center_id
        
        # TODO: 为了配合后面 region_centers = torch.from_numpy(region_centers).float()，将center id的int值转化为numpy array
        region_centers = [region_center_id]
        region_centers = np.array(region_centers)

        return region_mask_np, region_centers, region_faces_ids
        
    def obj_annots_2torch(self, sample, idx):
        """
        Randomly choose a neighbor region center in the loaded ones, get the included vertice indices.
        This operation is inherently facilitated with data-augmentation effect.
        """
        # 读取object template mesh
        obj_name = sample['obj_name']
        self.ObjMesh = self.object_meshes[obj_name]
        obj_faces = self.ObjMesh.faces
        obj_sdf_np = sample['obj_hand_sdf']
        rot_mat_np = np.array(sample['root_orient_obj_rotmat'][0])
        trans_np = np.array(sample['trans_obj'])
        obj_vertices = np.matmul(self.ObjMesh.vertices, rot_mat_np) + trans_np 

        # For the object meshes that have sdf results with lower than 3000 vertices during decimation
        if obj_sdf_np.shape[0] < 3000:
            add_sdf = np.min(obj_sdf_np)
            add_sdf = np.repeat(add_sdf, repeats=(3000 - obj_sdf_np.shape[0]), axis=0)
            obj_sdf_np = np.concatenate([obj_sdf_np.reshape(-1, 1), add_sdf.reshape(-1, 1)])
            # import pdb; pdb.set_trace()

        region_mask_np, region_centers, region_faces_ids = self.region_mask_vertices(sample, obj_vertices, obj_faces, idx)
        # import pdb; pdb.set_trace()

        # numpy 2 torch
        vertices = torch.from_numpy(obj_vertices).float() # --> ["obj_verts"]
        region_mask = torch.from_numpy(region_mask_np).float() # --> ["region_mask"]
        region_centers = torch.from_numpy(region_centers).float() # --> ["region_centers"]
        
        obj_sdf = torch.from_numpy(obj_sdf_np).reshape(-1, 1).float()# --> ["obj_sdf"]
        # obj_faces = torch.from_numpy(obj_faces).float()
        sample_idx = torch.tensor([idx]) # --> ["sample_idx"]

        return vertices, region_mask, region_centers, obj_sdf, sample_idx
    
    def get_frames_sdf_data(self, idx):
        sdf_data_file = np.load(self.frame_names_sdf[idx], allow_pickle=True)
        sdf_data = sdf_data_file.tolist() # dict
        return sdf_data
        
    def __getitem__(self, idx):
        data_out = {k: self.ds[k][idx] for k in self.ds.keys()}
        if not self.only_params:
            data = self.get_frames_data(idx, self.frame_names)
            data_out.update(data)
            # TODO: get_frames_data_sdf
            data_sdf = self.get_frames_sdf_data(idx)
            # import pdb; pdb.set_trace()
            data_sdf.update(data_out)
            data_sdf['obj_name'] = self.frame_objs[idx]
            # import pdb; pdb.set_trace()
            data_out['verts_obj'], data_out['region_mask'], data_out['region_centers'], data_out['obj_sdf'], data_out['sample_idx'] = self.obj_annots_2torch(data_sdf, idx) 
        return data_out
        
        
if __name__=="__main__":
    dataset = GrabNetDataset(dataset_root=config.DATASET_ROOT, 
                            ds_name="train", 
                            frame_names_file='frame_names_thumb.npz', 
                            grabnet_thumb=False, 
                            obj_meshes_folder='decimate_meshes',
                            select_ids=False, 
                            output_root=None, 
                            dtype=torch.float32, 
                            only_params=False, 
                            load_on_ram=False)
    
    
    sample = dataset.__getitem__(200031)

    
    
        