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
import open3d as o3d
from mano.model import load
from mano.utils import Mesh
import trimesh
from trimesh import viewer
from pycaster import pycaster

import time

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

        self.ds = self._np2torch(os.path.join(self.ds_path,'grabnet_%s.npz'%ds_name))
        frame_names = np.load(os.path.join(dataset_dir,ds_name, 'frame_names.npz'))['frame_names']
        self.frame_names =np.asarray([os.path.join(dataset_dir, fname) for fname in frame_names])
        self.frame_sbjs = np.asarray([name.split('/')[-3] for name in self.frame_names])
        self.frame_objs = np.asarray([name.split('/')[-2].split('_')[0] for name in self.frame_names])

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

    def get_ray_params(self, dir_sample_num=100):
        """
        生成从原点发出、全方位扫描的射线
        方向采样数default=100 => 1000,000 rays, stride 0.01 in every direction
        """
        values_dx = np.arange(-dir_sample_num, dir_sample_num, 1) / dir_sample_num
        values_dy = np.arange(-dir_sample_num, dir_sample_num, 1) / dir_sample_num
        values_dz = np.arange(-dir_sample_num, dir_sample_num, 1) / dir_sample_num
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
    
    def get_new_annotations(self, idx, sample):
        # raw_datas = {k: self.ds[k][idx] for k in self.ds.keys()}
        # sample = self.__getitem__(idx)
        sbj_name = self.frame_sbjs[idx]
        obj_name = self.frame_objs[idx]
        
        vtemp = self.sbj_vtemp[sbj_name]
        betas = self.sbj_betas[sbj_name].reshape(1, -1) # shape (1, 10)

        """
        [Obtain object centric coordinates of both hand and objects]
        """
        #TODO: 该subject下的先得到标准template mesh --> rhand_mesh
        rhand_sbj_model = load(model_path = self.mano_path,
                          is_rhand = True,
                          use_pca = False,
                          betas = betas,
                          v_template = vtemp)
        
        
        # hand_pose_mean = rhand_sbj_model.hand_mean.reshape(-1, 1, 3)
        # hand_pose_new = np.matmul(hand_pose_mean, raw_datas['fpose_rhand_rotmat'].transpose(1,2)).reshape(1,-1) #
        # rhand_mesh_output = rhand_sbj_model.forward()
        # # rhand_mesh_vertices = self.to_np(rhand_mesh_output.vertices)
        # rhand_mesh_vertices = rhand_mesh_output.vertices.detach().numpy().squeeze(0)
        # HandMesh = Mesh(vertices=rhand_mesh_vertices, faces=rhand_sbj_model.faces, vc=colors['skin']) # trimesh
        
        HandMesh = Mesh(vertices=sample['verts_rhand'], faces=rhand_sbj_model.faces, vc=colors['skin'])
        # HandMesh = o3d.geometry.TriangleMesh()
        # HandMesh.v = o3d.utility.Vector3dVector(np.copy(rhand_mesh.verti)) # forward后 --> output.vertices
        # HandMesh.triangles = rhand_sbj_model.faces # MANO --> self.faces
        

        #TODO: 读取object template mesh
        obj_path = os.path.join(self.obj_mesh_dir, obj_name+'.ply')
        # ObjMesh = o3d.io.read_triangle_mesh(obj_path) # 先用o3d读再转trimesh？？
        ObjMesh = trimesh.load(obj_path)
        # ObjMesh = trimesh.load(obj_path)
        ObjMesh = Mesh(vertices=ObjMesh.vertices, faces=ObjMesh.faces, vc=colors['grey'])

        #TODO: 在object_centric坐标系下旋转、平移手和物体
        # ObjMesh.vertices = np.matmul(ObjMesh.vertices, raw_datas['root_orient_obj_rotmat'][0].T) + raw_datas['trans_obj']
        # HandMesh.vertices = np.matmul(HandMesh.vertices, raw_datas['global_orient_rhand_rotmat'][0].T) + raw_datas['trans_rhand']
        # 注意: raw_datas中的rotmat shape都是(1,3,3)
        #TODO: [checkpoint!!] visualize and check if it is actually "object-centric"


        """
        [Ray tracing with pycaster]
        - 用trimesh + pyembree: class trimesh.ray.ray_pyembree.RayMeshIntersector(geometry, scale_to_box=True)
        原因是据说加了pyembree之后速度会快 50x
            - class输入：
                - geometry: (Trimesh object)
                - scale_to_box：scale mesh to approximate unit cube
        - 使用类下的函数intersect_id
            -- 输入：
                - ray origins: 射线原点
                - ray directions: 射线方向向量 direction vector of rays --> 如何扫描全方位的方向？
                - return_locations(bool): 返回相交点坐标
            -- 输出：
                - index_tri: 相交的面indexes
                - index_ray: 有相交点的ray的indexes
                - locations: 相交点的坐标

        """

        # caster = pycaster.rayCaster.fromSTL() #???一定要STL吗 救。。。

        # initialize raycaster for both objs and hands, with the defined sources and directions
        ObjRaycaster = trimesh.ray.ray_pyembree.RayMeshIntersector(ObjMesh)
        HandRaycaster = trimesh.ray.ray_pyembree.RayMeshIntersector(HandMesh)

        if self.ray_sources is None and self.ray_directions is None:
            self.get_ray_params()

        hand_tri_idxs, hand_ray_idxs, hand_points = HandRaycaster.intersects_id(ray_origins=self.ray_sources, ray_directions=self.ray_directions, return_locations=True)

        obj_tri_idxs, obj_ray_idxs, obj_points = ObjRaycaster.intersects_id(ray_origins=self.ray_sources, ray_directions=self.ray_directions, return_locations=True)

        # hoi_ray_idxs = [idx for idx in hand_ray_idxs and obj_ray_idxs]
        hoi_ray_idxs, _, hoi_obj_idx = np.intersect1d(hand_ray_idxs, obj_ray_idxs, return_indices=True) # 可以返回相应的编号

        hoi_objfaces_idxs = obj_tri_idxs[hoi_obj_idx]
        hoi_objpoints = obj_points[hoi_obj_idx]

        ObjMesh.set_face_colors(colors['pink'], face_ids=hoi_objfaces_idxs)

        #TODO: 把相交面所经过的顶点都筛选出来？
        

        #TODO: 再下采样
        


        return [hoi_objfaces_idxs, hoi_objpoints]





    def __getitem__(self, idx):

        data_out = {k: self.ds[k][idx] for k in self.ds.keys()}
        if not self.only_params:
            if not self.load_on_ram:
                form_disk = self.load_disk(idx)
                data_out.update(form_disk)
        hoi_objfaces_idxs, hoi_objpoints = self.get_new_annotations(idx, data_out)
        
        data_out.update()
        
        return data_out



if __name__ == "__main__":
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
    sample = dataset.__getitem__(0)

    ObjMesh, HandMesh, hoi_info = dataset.get_new_annotations(0)
    hoi_objfaces_idxs, hoi_objpoints = hoi_info

    # scene = trimesh.scene.Scene(geometry=[ObjMesh, HandMesh])

    ObjMesh.show()
    
    # meshviewer = viewer.SceneViewer(scene)
    # meshviewer.add_geometry('hand', HandMesh)







    

    


    



    