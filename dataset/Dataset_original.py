import sys
from turtle import color, forward
sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch
import cv2
import os
from torch.utils import data
from torch.utils.data._utils.collate import default_collate
# import open3d as o3d
from mano.model import load
from mano.utils import Mesh
import trimesh
from trimesh import viewer
# from pycaster import pycaster

import time
from utils.utils import func_timer, makepath
# from vedo import show

to_cpu = lambda tensor: tensor.detach().cpu()

colors = {
    'pink': [1.00, 0.75, 0.80],
    'skin': [0.96, 0.75, 0.69],
    'purple': [0.63, 0.13, 0.94],
    'red': [1.0, 0.0, 0.0],
    'green': [.0, 1., .0],
    'yellow': [1., 1., 0],
    'brown': [1.00, 0.25, 0.25],
    'blue': [.0, .0, 1.],
    'white': [1., 1., 1.],
    'orange': [1.00, 0.65, 0.00],
    'grey': [0.75, 0.75, 0.75],
    'black': [0., 0., 0.],
}
JOINTS_NUM = 15

########################################################################################################
########################################################################################################


class GrabNetDataset(data.Dataset):
    def __init__(self,
                 dataset_dir,
                 ds_name='train',
                 dtype=torch.float32,
                 only_params = False,
                 load_on_ram = False):

    
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

        self.ds_path = os.path.join(dataset_dir, ds_name)
        self.mano_path = "/home/datassd/yilin/Codes/_toolbox/mano/models/MANO_RIGHT.pkl" #MANO right hand model path
        self.obj_mesh_dir = "/home/datassd/yilin/GrabNet/tools/object_meshes/contact_meshes/"
        self.output_path = "/home/datassd/yilin/Outputs/GrabNet_visual"

        self.ds = self._np2torch(os.path.join(self.ds_path,'grabnet_%s.npz'%ds_name))
        frame_names = np.load(os.path.join(dataset_dir,ds_name, 'frame_names.npz'))['frame_names']
        self.frame_names =np.asarray([os.path.join(dataset_dir, fname) for fname in frame_names])
        self.frame_sbjs = np.asarray([name.split('/')[-3] for name in self.frame_names])
        self.frame_objs = np.asarray([name.split('/')[-2].split('_')[0] for name in self.frame_names])
        self.frame_acts = np.asarray([name.split('/')[-2].split('_')[1] for name in self.frame_names])

        # subject info
        self.sbjs = np.unique(self.frame_sbjs)
        self.sbj_info = np.load(os.path.join(dataset_dir, 'sbj_info.npy'), allow_pickle=True).item()

        self.sbj_vtemp = torch.from_numpy(np.asarray([self.sbj_info[sbj]['rh_vtemp'] for sbj in self.sbjs]))
        self.sbj_betas = torch.from_numpy(np.asarray([self.sbj_info[sbj]['rh_betas'] for sbj in self.sbjs]))

        # object info
        obj_info_object = np.load(os.path.join(dataset_dir, 'objects_info.npz'), allow_pickle=True)
        self.obj_info = {k: obj_info_object[k] for k in obj_info_object.files}

        # self.frame_sbj_names = self.frame_sbjs

        for idx, name in enumerate(self.sbjs):
            self.frame_sbjs[(self.frame_sbjs == name)] = idx

        self.frame_sbjs=torch.from_numpy(self.frame_sbjs.astype(np.int8)).to(torch.long)

        self.load_on_ram = False
        if load_on_ram:
            self.ds = self[:]
            self.load_on_ram = True

        self.ray_directions, self.ray_sources = None, None

    def _np2torch(self,ds_path):
        data = np.load(ds_path, allow_pickle=True)
        data_torch = {k:torch.tensor(data[k]) for k in data.files}
        return data_torch
    
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
            return self._np2torch(self.frame_names[idx])

        frame_names = self.frame_names[idx]
        from_disk = []
        for f in frame_names:
            from_disk.append(self._np2torch(f))
        from_disk = default_collate(from_disk)
        return from_disk

    def __len__(self):
        k = list(self.ds.keys())[0]
        return self.ds[k].shape[0]
        # return len(self.frame_names)

    @func_timer
    def get_ray_params(self, dir_sample_num=10):
        """
        生成从原点发出、全方位扫描的射线
        方向采样数default=100 => 1000,000 rays, stride 0.01 in every direction
        """
        eps = 0.01
        values_dx = (np.arange(-dir_sample_num, dir_sample_num, 1) + eps) / dir_sample_num 
        values_dy = (np.arange(-dir_sample_num, dir_sample_num, 1) + eps) / dir_sample_num
        values_dz = (np.arange(-dir_sample_num, dir_sample_num, 1) + eps) / dir_sample_num
        print(f"direction_x_value: {values_dx}; direction_y_value: {values_dy}; direction_z_value: {values_dz};")
        ray_directions = []
        ray_sources = []
        for xval in values_dx:
            for yval in values_dy:
                for zval in values_dz:
                    ray_directions.append(np.array([[xval, yval, zval]])) #TODO check 坐标order
                    ray_sources.append(np.array([[0, 0, 0]])) #TODO check 这个点需要是物体中心，按道理object centric的原点是物体中心

        self.ray_directions = np.concatenate(ray_directions, axis=0)
        self.ray_sources = np.concatenate(ray_sources, axis=0)
        print(f"ray direction shape: {self.ray_directions.shape}; ray sources shape: {self.ray_sources.shape}")



    def __getitem__(self, idx):

        data_out = {k: self.ds[k][idx] for k in self.ds.keys()}
        if not self.only_params:
            if not self.load_on_ram:
                form_disk = self.load_disk(idx)
                data_out.update(form_disk)
        hoi_objfaces_idxs, hoi_objpoints = self.get_new_annotations(idx, data_out)
        
        #TODO make the hoi annotations to dict form
        data_out.update()
        
        return data_out



if __name__ == "__main__":
    from tqdm import tqdm
    """
    [Predefine the roots to use ]
    - dataset root, 
    - obj_meshes root,
    - hand mano model path
    """
    dataset_dir = "/home/datassd/yilin/GrabNet/data/"
    obj_mesh_dir = "/home/datassd/yilin/GrabNet/tools/object_meshes/contact_meshes/"
    mano_dir = "/home/datassd/yilin"
    
    dataset = GrabNetDataset(dataset_dir=dataset_dir)
    # sample = dataset.__getitem__(0)
    idx = 0
    test_idx = np.arange(0, dataset.__len__(), 100).tolist()
    for idx in tqdm(test_idx):
        sample = dataset.__getitem__(idx)
        

    # ObjMesh, HandMesh, hoi_info = dataset.get_new_annotations(0)
    # hoi_objfaces_idxs, hoi_objpoints = hoi_info

    # scene = trimesh.scene.Scene(geometry=[ObjMesh, HandMesh])

    # ObjMesh.show()
    
    # meshviewer = viewer.SceneViewer(scene)
    # meshviewer.add_geometry('hand', HandMesh)







    

    


    



    