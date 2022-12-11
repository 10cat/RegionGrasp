import math
import sys
sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch
# import cv2
import os
from torch.utils import data
from torch.utils.data._utils.collate import default_collate
# import open3d as o3d
from mano.model import load
# from mano.utils import Mesh
import trimesh
import config
from copy import deepcopy
# from trimesh import viewer
# from pycaster import pycaster

import mano
import time
import random
from utils.utils import func_timer, makepath
from utils.visualization import visual_sdf, visual_obj_contact_regions
# from vedo import show

class GrabNetDataset(data.Dataset):
    @func_timer
    def __init__(self,
                 dataset_dir,
                 ds_name='train',
                 dtype=torch.float32,
                 only_params = False,
                 load_on_ram = False,
                 num_mask = 1):

    
        super().__init__()
        """
        [Preprocess the GrabNet raw data]
        sample should have:
        - orients and trans params of hands: (from preprocess params with 10 different subject templates)
            -- 'global_orient_rhand_rotmat', 'fpose_rhand_rotmat', 'trans_rhand', 
            -- 'global_orient_rhand_rotmat_f', 'fpose_rhand_rotmat_f', 'trans_rhand_f'
        - orients and trans params of objects:
            -- 'trans_obj', 'root_orient_obj_rotmat'
        """
        self.only_params = only_params

        self.ds_name = ds_name
        self.num_mask = num_mask

        self.ds_path = os.path.join(dataset_dir, ds_name)
        self.mano_path = config.mano_dir
        self.obj_mesh_dir = config.obj_mesh_dir
        # self.obj_sample_dir = "/home/datassd/yilin/GrabNet/tools/object_meshes/sample_info/"
        self.output_path = config.dataset_visual_dir

        # self.create_sdf_folders()

        self.ds = self._np2torch(os.path.join(self.ds_path,'grabnet_%s.npz'%ds_name))
        frame_names = np.load(os.path.join(self.ds_path, 'frame_names.npz'))['frame_names']
        # import pdb; pdb.set_trace()
        
        
        # save frame file paths to numpy
        # -- frame_names: original annotations in the GrabNet dataset
        # -- frame_names_sdf: the frame files that are going to be used to save the new annotations
        frame_names_list, frame_names_sdf_list = [], []
        for fname in frame_names:
            frame_names_list.append(os.path.join(dataset_dir, fname))
            frame_names_sdf_list.append(os.path.join(dataset_dir, fname.replace('data', 'data_sdf').replace('npz', 'npy'))) # new annotations以字典形式存在npy文件中，因此需要链式两重替换
        self.frame_names, self.frame_names_sdf = np.asarray(frame_names_list), np.asarray(frame_names_sdf_list)
            
        self.frame_sbjs = np.asarray([name.split('/')[-3] for name in self.frame_names])
        self.frame_objs = np.asarray([name.split('/')[-2].split('_')[0] for name in self.frame_names])
        self.frame_acts = np.asarray([name.split('/')[-2].split('_')[1] for name in self.frame_names])
        self.frame_numbers = np.asarray([name.split('/')[-1].split('.')[0] for name in self.frame_names])

        # subject info
        self.sbjs = np.unique(self.frame_sbjs)
        self.sbj_info = np.load(os.path.join(dataset_dir, 'sbj_info.npy'), allow_pickle=True).item()

        self.sbj_vtemp = torch.from_numpy(np.asarray([self.sbj_info[sbj]['rh_vtemp'] for sbj in self.sbjs]))
        self.sbj_betas = torch.from_numpy(np.asarray([self.sbj_info[sbj]['rh_betas'] for sbj in self.sbjs]))

        # object info
        obj_info_object = np.load(os.path.join(dataset_dir, 'objects_info.npz'), allow_pickle=True)
        self.obj_info = {k: obj_info_object[k] for k in obj_info_object.files}
        self.object_meshes = self.load_obj_meshes()

        # self.frame_sbj_names = self.frame_sbjs

        for idx, name in enumerate(self.sbjs):
            self.frame_sbjs[(self.frame_sbjs == name)] = idx

        self.frame_sbjs=torch.from_numpy(self.frame_sbjs.astype(np.int8)).to(torch.long)

        self.load_on_ram = False
        if load_on_ram:
            self.ds = self[:]
            self.load_on_ram = True

        if self.ds_name != 'train':
            # FIXME: 在 val/test的时候，选取的mask_center_ids应该是确定的, 而不是每次都random生成
            self.mask_center_ids = np.random.random(self.__len__())
        else:
            self.mask_center_ids = None
        # import pdb; pdb.set_trace()
        ####

        self.region_face_ids = {}

    def load_obj_meshes(self):
        obj_meshes = {}
        obj_files = os.listdir(self.obj_mesh_dir)
        for obj_file in obj_files:
            obj_name = obj_file.split('.')[0]
            obj_meshes[obj_name] = trimesh.load(os.path.join(self.obj_mesh_dir, obj_file))

        return obj_meshes

    def _np2torch(self,ds_path):
        data = np.load(ds_path, allow_pickle=True)
        data_torch = {k:torch.tensor(data[k]) for k in data.files}
        return data_torch

    def _data_sdf(self, ds_path):
        data = np.load(ds_path, allow_pickle=True)
        # import pdb; pdb.set_trace()
        return data.tolist()
    
    def to_np(array, dtype=np.float32):
        if 'scipy.sparse' in str(type(array)):
            array = np.array(array.todense())
        elif 'chumpy' in str(type(array)):
            array = np.array(array)
        elif torch.is_tensor(array):
            array = array.detach().cpu().numpy()
        return array.astype(dtype)

    def load_disk(self, idx):

        if isinstance(idx, int):
            data = self._np2torch(self.frame_names[idx])
            # import pdb; pdb.set_trace()
            data_sdf = self._data_sdf(self.frame_names_sdf[idx])
            return data, data_sdf

        frame_names = self.frame_names[idx]
        from_disk = []
        
        for f in frame_names:
            from_disk.append(self._np2torch(f))
            f_sdf = f.replace('data', 'data_sdf').replace('npz', 'npy')
            from_disk.append(self._np2torch(f_sdf))
        from_disk = default_collate(from_disk)
        return from_disk

    def region_mask_vertices(self, sample, obj_vertices, obj_faces, idx):
        
        # import pdb; pdb.set_trace()
        candidates = sample['candidates']
        candidate_centers = obj_faces[sample['obj_faces_ids'].tolist()] # all the candidate centers 
        
        if self.ds_name == "train":
            chosen_idx = int(np.random.choice(candidate_centers.shape[0]))
        else:
            if self.mask_center_ids is None: raise ValueError("In val / test mask_center_ids cannot be None!")
            else:
                chosen_idx = int(self.mask_center_ids[idx] * candidate_centers.shape[0])
        if chosen_idx > self.num_mask:
            if chosen_idx < candidate_centers.shape[0] - self.num_mask:
                start = chosen_idx - int(self.num_mask / 2)
            else:
                start = candidate_centers.shape[0] - self.num_mask

        else:
            start = 0
        end = start + self.num_mask
        
        region_faces_ids = []
        for i in range(start, end):
            region_faces_ids = region_faces_ids + candidates[i]
        # import pdb; pdb.set_trace()
        # centers = np.arange(start, end)
        
        region_centers = sample['obj_faces_ids'][start:end]

        region_faces = obj_faces[region_faces_ids]
        region_vertices_ids = np.unique(region_faces.reshape(-1)).tolist()
        # import pdb; pdb.set_trace()
        region_mask_np = np.zeros_like(obj_vertices[:, :1], dtype=float)
        region_mask_np[region_vertices_ids] = 1.0

        self.region_face_ids[str(idx)] = region_faces_ids

        return region_mask_np, region_centers, region_faces_ids

    def obj_annots_2torch(self, sample, idx):
        """
        Randomly choose a neighbor region center in the loaded ones, get the included vertice indices.
        This operation is inherently facilitated with data-augmentation effect.
        """

        # 读取object template mesh
        obj_name = sample['obj_name']
        # obj_path = os.path.join(self.obj_mesh_dir, obj_name+'.ply')
        # self.ObjMesh = trimesh.load(obj_path)
        self.ObjMesh = self.object_meshes[obj_name]
        # obj_vertices = self.ObjMesh.vertices
        obj_faces = self.ObjMesh.faces
        obj_sdf_np = sample['obj_hand_sdf']
        rot_mat_np = np.array(sample['root_orient_obj_rotmat'][0])
        trans_np = np.array(sample['trans_obj'])
        obj_vertices = np.matmul(self.ObjMesh.vertices, rot_mat_np) + trans_np 

        # For the object meshes that have sdf results with lower than 3000 vertices during decimation
        if obj_sdf_np.shape[0] < 3000:
            # import pdb; pdb.set_trace()
            # origin = np.array([[0., 0., 0.]], dtype=obj_vertices.dtype)
            add_sdf = np.min(obj_sdf_np)
            add_sdf = np.repeat(add_sdf, repeats=(3000 - obj_sdf_np.shape[0]), axis=0)
            # obj_vertices = np.concatenate(obj_vertices, origin)
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


    def __len__(self):
        k = list(self.ds.keys())[0]
        return self.ds[k].shape[0]
        # return len(self.frame_names)

    def __getitem__(self, idx):
        # if self.frame_sbjs[idx] == 9 and self.ds_name != 'train':
        #     return
        data_out = {k: self.ds[k][idx] for k in self.ds.keys()}
        if not self.only_params:
            if not self.load_on_ram:
                data, data_sdf = self.load_disk(idx)
                data_out.update(data)
                # import pdb; pdb.set_trace()
                data_sdf.update(data_out)
                data_sdf['obj_name'] = self.frame_objs[idx]
                # import pdb; pdb.set_trace()
                data_out['verts_obj'], data_out['region_mask'], data_out['region_centers'], data_out['obj_sdf'], data_out['sample_idx'] = self.obj_annots_2torch(data_sdf, idx) 
        return data_out

if __name__ == "__main__":
    from tqdm import tqdm
    import time
    random_seed = 1000
    np.random.seed(random_seed)

    dataset_dir = "/home/datassd/yilin/GrabNet/data/"

    traindataset = GrabNetDataset(dataset_dir=dataset_dir, ds_name='train', num_mask=1)

    valdataset = GrabNetDataset(dataset_dir=dataset_dir, ds_name='val', num_mask=1)

    dataset = GrabNetDataset(dataset_dir=dataset_dir, ds_name='test', num_mask=1)

    traindataset.__getitem__(0)

    dataloader = data.DataLoader(dataset, batch_size=16, shuffle=False)

    start_time = time.time()

    for idx, sample in enumerate(tqdm(dataloader)):
        region_mask = sample['region_mask']
        verts_obj = sample['verts_obj']
        obj_sdf = sample['obj_sdf']
        verts_rhand = sample['verts_rhand']

    idx = 0
    
    
        
