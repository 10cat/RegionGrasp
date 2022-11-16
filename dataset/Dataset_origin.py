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

    
class GrabNetDataset_orig(data.Dataset):
    @func_timer
    def __init__(self,
                 dataset_root,
                 ds_name='train',
                 obj_meshes_folder = 'contact_meshes',
                 output_root = None,
                 dtype=torch.float32,
                 only_params=False,
                 load_on_ram=False):
        super().__init__()
        self.ds_name = ds_name
        self.ds_root = os.path.join(dataset_root, 'data')
        self.ds_path = os.path.join(self.ds_root, ds_name)
        self.ts_root = os.path.join(dataset_root, 'tools')
        self.obj_mesh_dir = os.path.join(self.ts_root, 'object_meshes', obj_meshes_folder)
        self.output_root = output_root
        self.only_params = only_params
        
        frame_names = np.load(os.path.join(self.ds_path, 'frame_names.npz'))['frame_names'] # 所有sample的标注信息.npz文件的相对路径
        # import pdb; pdb.set_trace()
        
        ## 将frame_names相对路径拼接成绝对路径
        frame_names_list = []
        for fname in frame_names:
            frame_names_list.append(os.path.join(self.ds_root, fname))
        self.frame_names = np.asarray(frame_names_list)
        
        ## 从frame_names中提取每个frame的基本信息
        self.frame_sbjs = np.asarray([name.split('/')[-3] for name in self.frame_names])
        self.frame_objs = np.asarray([name.split('/')[-2].split('_')[0] for name in self.frame_names])
        self.frame_acts = np.asarray([name.split('/')[-2].split('_')[1] for name in self.frame_names])
        self.frame_numbers = np.asarray([name.split('/')[-1].split('.')[0] for name in self.frame_names])
        
        ## 获取所有sbject对应的mano参数信息
        self.sbjs = np.unique(self.frame_sbjs)
        self.sbj_info = np.load(os.path.join(self.ds_root, 'sbj_info.npy'), allow_pickle=True).item()
        self.sbj_vtemp = np.asarray([self.sbj_info[sbj]['rh_vtemp'] for sbj in self.sbjs])
        self.sbj_betas = np.asarray([self.sbj_info[sbj]['rh_betas'] for sbj in self.sbjs])
        ## 将self.frame_sbjs中原本存放sbject id的str格式转化成int
        for idx, name in enumerate(self.sbjs):
            self.frame_sbjs[(self.frame_sbjs == name)] = idx
        
        ## 获取所有物体参数，以及转化成相应的mesh
        obj_info = np.load(os.path.join(self.ds_root, 'objects_info.npz'), allow_pickle=True)
        self.obj_info = {k: obj_info[k].item() for k in obj_info.files} 
        # import pdb; pdb.set_trace()
        self.object_meshes = self.load_obj_meshes()
        
        self.to_torch = False
        if dtype == torch.float32:
            self.sbj_vtemp = torch.from_numpy(self.sbj_vtemp)
            self.sbj_betas = torch.from_numpy(self.sbj_betas)
            self.frame_sbjs = torch.from_numpy(self.frame_sbjs.astype(np.int8)).to(torch.long)
            self.to_torch = True
            
        ## 读取grabnet_[ds_name].npz中提供的所有sample基本参数：mano参数 / hand pose参数 / object pose参数
        self.ds = self.get_npz_data(os.path.join(self.ds_path, 'grabnet_%s.npz'%ds_name), to_torch=self.to_torch)
        
        
    def load_obj_meshes(self):
        obj_meshes = {}
        obj_files = os.listdir(self.obj_mesh_dir)
        for obj_file in obj_files:
            obj_name = obj_file.split('.')[0]
            obj_meshes[obj_name] = trimesh.load(os.path.join(self.obj_mesh_dir, obj_file))
        return obj_meshes
    
    def npz_np2torch(self, data):
        data_torch = {k: torch.tensor(data[k]) for k in list(data.keys())}
        return data_torch
    
    def get_npz_data(self, ds_path, to_torch=False):
        data_npz = np.load(ds_path, allow_pickle=True)
        data = {k: data_npz[k] for k in data_npz.files}
        if to_torch:
            data = self.npz_np2torch(data)
        return data
    
    def get_frames_data(self, idx):
        """
        读取frame_names里面提供的直接可用于训练的参数：
        object_verts(下采样至2048) / rhand_verts / rhand_verts_f / bps_object
        """
        # idx不是int就是list
        if isinstance(idx, int):
            data = self.get_npz_data(self.frame_names[idx], to_torch=self.to_torch)
            return data
        
        frame_names = self.frame_names[idx] # 获得多个frame_names
        from_disk = []
        for f in frame_names:
            from_disk.append(self.get_npz_data(f, to_torch=self.to_torch))
        from_disk = default_collate(from_disk)
        return from_disk
    
    def __len__(self):
        k = list(self.ds.keys())[0]
        return self.ds[k].shape[0]
    
    def __getitem__(self, idx):
        ## 直接通过self.ds获取这一个sample对应的data params
        data_out = {k: self.ds[k][idx] for k in self.ds.keys()}
        # import pdb; pdb.set_trace()
        if not self.only_params:
            data = self.get_frames_data(idx)
            # import pdb; pdb.set_trace()
            data_out.update(data)
            
        return data_out
    

        
if __name__ == "__main__":
    dataset_root = config.DATASET_ROOT
    output_root = config.dataset_visual_dir
    
    trainset = GrabNetDataset_orig(dataset_root, ds_name='train', output_root=output_root)
    trainset.__getitem__(0)
        
        
        
        
        