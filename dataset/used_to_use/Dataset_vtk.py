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
import meshparty
from trimesh import viewer
from pycaster import pycaster

import time
from utils.utils import func_timer, makepath
# from utils.vtk_utils import loadPLY
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
        eps = 0 #0.01
        div = 0.1
        values_dx = (np.arange(-dir_sample_num, dir_sample_num, div) + eps) #/ dir_sample_num 
        values_dy = (np.arange(-dir_sample_num, dir_sample_num, div) + eps) #/ dir_sample_num
        values_dz = (np.arange(-dir_sample_num, dir_sample_num, div) + eps) #/ dir_sample_num
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

    def get_obj_Psurface(self):
        hoi_obj_faces = []
        for idx, obj_idx in enumerate(self.hoi_obj_idx):
            hoi_obj_points = self.obj_points[obj_idx] # (n,3) array
            hoi_hand_points = self.hand_points[self.hoi_hand_idx[idx]]
            hand_min_pnorm = min(np.linalg.norm(hoi_hand_points, axis=1)) # float
            # hand_max_pnorm = min
            obj_max_pnorm = max(np.linalg.norm(hoi_obj_points, axis=1)) # float

            if hand_min_pnorm > obj_max_pnorm:
                hoi_obj_tri = self.obj_tri_idxs[obj_idx][-1].reshape(1, -1) # 只保留最外点：(1, 3) array
            else:
                hoi_obj_tri = self.obj_tri_idxs[obj_idx] # 保留所有 - 也就是(2, 3) array
            
            hoi_obj_faces.append(hoi_obj_tri)

        return hoi_obj_faces

    def vtkraycasting(self, vtkObjMesh, vtkHandMesh, HandMesh, ObjMesh):
        # initialize raycaster for both objs and hands, with the defined sources and directions
        ObjPycaster = pycaster.rayCaster(vtkObjMesh)
        HandPycaster = pycaster.rayCaster(vtkHandMesh)
        # initialize pTargets and pSource for the sample
        pTargets = HandMesh.vertices.tolist() # len = 778
        pSource = [0, 0, 0]
        # initialize binary map and distance map based on object mesh faces
        confaces_bmask = np.array((ObjMesh.faces.shape[0],))
        confaces_distmask = np.array((ObjMesh.faces.shape[0],))
        for i, pTarget in enumerate(pTargets):
            pIntersect_obj =  ObjPycaster.castRay(pSource, pTarget)
            pIntersect_hand = HandPycaster.castRay(pSource, pTarget)
            if pIntersect_obj:
                objpoints = np.array(pIntersect_obj)
                if pIntersect_hand:
                    handpoints = np.array(pIntersect_hand)
                    pNearest_hand = handpoints[0]
                    pFarthest_obj = objpoints[-1]
                    

    @func_timer
    def get_new_annotations(self, idx, sample):
        # raw_datas = {k: self.ds[k][idx] for k in self.ds.keys()}
        # sample = self.__getitem__(idx)
        sbj_name = self.frame_sbjs[idx]
        obj_name = self.frame_objs[idx]
        act_name = self.frame_acts[idx]
        
        vtemp = self.sbj_vtemp[sbj_name]
        betas = self.sbj_betas[sbj_name].reshape(1, -1) # shape (1, 10)

        """
        [Obtain object centric coordinates of both hand and objects]
        """
        #TODO: 该subject下的先得到标准template mesh --> rhand_mesh
        rhand_sbj_model = load(model_path = self.mano_path,
                          is_rhand = True,
                          num_pca_comps=45,
                          betas = betas,
                          v_template = vtemp,
                          flat_hand_mean=True)
        
        
        # hand_pose_mean = rhand_sbj_model.hand_mean.reshape(-1, 1, 3)
        # hand_pose_new = np.matmul(hand_pose_mean, raw_datas['fpose_rhand_rotmat'].transpose(1,2)).reshape(1,-1) #
        # rhand_mesh_output = rhand_sbj_model.forward()
        # # rhand_mesh_vertices = self.to_np(rhand_mesh_output.vertices)
        # rhand_mesh_vertices = rhand_mesh_output.vertices.detach().numpy().squeeze(0)
        # HandMesh = Mesh(vertices=rhand_mesh_vertices, faces=rhand_sbj_model.faces, vc=colors['skin']) # trimesh
        
        
        # HandMesh = o3d.geometry.TriangleMesh()
        # HandMesh.v = o3d.utility.Vector3dVector(np.copy(rhand_mesh.verti)) # forward后 --> output.vertices
        # HandMesh.triangles = rhand_sbj_model.faces # MANO --> self.faces
        

        #TODO: 读取object template mesh
        obj_path = os.path.join(self.obj_mesh_dir, obj_name+'.ply')
        # ObjMesh = loadPLY(obj_path)
        # ObjMesh = o3d.io.read_triangle_mesh(obj_path) # 先用o3d读再转trimesh？？
        # ObjMesh = trimesh.load(obj_path)
        ObjMesh = trimesh.load(obj_path)

        # TODO: 在object_centric坐标系下旋转、平移手和物体
        # 坑：GrabNet里面给的rotmat是已经转置过的
        # obj_verts = np.matmul(ObjMesh.vertices, sample['root_orient_obj_rotmat'][0].T) + sample['trans_obj']
        obj_verts = np.matmul(ObjMesh.vertices, sample['root_orient_obj_rotmat'][0]) + sample['trans_obj'] # the rotation matrix is already transposed
        
        hand_verts = sample['verts_rhand']
        # 注意: raw_datas中的rotmat shape都是(1,3,3)

        # have 
        HandMesh = Mesh(vertices=hand_verts, faces=rhand_sbj_model.faces, vc=colors['skin'])
        ObjMesh = Mesh(vertices=obj_verts, faces=ObjMesh.faces, vc=colors['grey'])
        # TODO: [checkpoint!!] visualize and check if it is actually "object-centric"
        # initialize a primitive whose center is at (0,0,0) in the coordinate
        originSphere = trimesh.primitives.Sphere(radius=0.001, center=[0,0,0])

        # TODO: 使用meshparty将hand/objectmesh从trimesh的mesh格式转成vtk中的vtkpolydata格式
        vtkHandMesh = meshparty.trimesh_to_vtk(HandMesh.vertices, HandMesh.faces)
        vtkObjMesh = meshparty.trimesh_to_vtk(ObjMesh.vertices, ObjMesh.faces)

        """
        [Ray tracing with pycaster]
        - 用pycaster: class rayCaster
            - class输入:
                - mesh (vtkPolyData)
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

        if self.ray_sources is None and self.ray_directions is None:
            self.get_ray_params()

       # TODO: obtain intersection points 
       # 不一定需要两个pycaster了！！ 可以直接从手的所有顶点发出射线？？

        self.hand_tri_idxs, self.hand_ray_idxs, self.hand_points = HandRaycaster.intersects_id(ray_origins=self.ray_sources, ray_directions=self.ray_directions, 
                                                                                multiple_hits=True, return_locations=True)

        self.obj_tri_idxs, self.obj_ray_idxs, self.obj_points = ObjRaycaster.intersects_id(ray_origins=self.ray_sources, ray_directions=self.ray_directions, 
                                                                            multiple_hits=True, return_locations=True)
        
        # intersection_locations -- hand_points / obj_points: (m) sequences of (p, 3) float

        ObjRaycaster_tri = trimesh.ray.ray_triangle.RayMeshIntersector(ObjMesh)
        obj_tri_idx, obj_ray_idx, obj_point = ObjRaycaster_tri.intersects_id(ray_origins=self.ray_sources, ray_directions=self.ray_directions, 
                                                                                multiple_hits=True, return_locations=True)

        self.hoi_ray_idxs, self.hoi_hand_idx, self.hoi_obj_idx = np.intersect1d(self.hand_ray_idxs, self.obj_ray_idxs, return_indices=True) 
        # np.intersect1d可以通过return_indices返回相应的编号

        #TODO: 对于不同物体正确筛选obj表面相交面 based on norm distance
        # L2 norm (because the source is (0,0,0))
        hoi_obj_faces = self.get_obj_Psurface()
        
        # hoi_objfaces_idxs = obj_tri_idxs[hoi_obj_idx][-1]
        # hoi_objpoints = obj_points[hoi_obj_idx]
        # ObjMesh.set_face_colors(colors['pink'], face_ids=hoi_objfaces_idxs)
        
        

        #TODO: 计算射线与hand mesh表面到object mesh表面的距离； 以face为单位 
        # 
        # L2 distance ?
        


        #TODO: 将face的distance map可视化为heatmap
        # trimesh.visual.interpolate --> output: RGBA color (A = alpha: degree of untransparency)
        # matplotlib.pyplot.colormaps 


        #TODO: 把相交面所经过的顶点都筛选出来？
        obj_output_path = os.path.join(self.output_path, str(obj_name))
        makepath(obj_output_path)

        name = os.path.join(obj_output_path, str(act_name) + '_' + 's' + str(int(sbj_name)) + '_' +  str(idx))
        HandMesh.export( name + '_hand.ply')
        ObjMesh.export( name + '_obj.ply')
        originSphere.export( name + '_origin.ply')

        # visualize
        
        # show([ObjMesh, HandMesh, originSphere], axes=1) # use vedo
        # combined = trimesh.util.concatenate([HandMesh, ObjMesh, originSphere])
        # scene = trimesh.scene.Scene(geometry=[HandMesh, ObjMesh, originSphere]) # use trimesh scene viewer
        # scene = trimesh.scene.Scene(geometry=combined)
        # meshviewer = viewer.SceneViewer(scene)

        # meshviewer.toggle_axis()
        

        #TODO: 读取时 -- 下采样
        

        return [hoi_objfaces_idxs, hoi_objpoints]





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
        dataset.__getitem__(idx)

    # ObjMesh, HandMesh, hoi_info = dataset.get_new_annotations(0)
    # hoi_objfaces_idxs, hoi_objpoints = hoi_info

    # scene = trimesh.scene.Scene(geometry=[ObjMesh, HandMesh])

    # ObjMesh.show()
    
    # meshviewer = viewer.SceneViewer(scene)
    # meshviewer.add_geometry('hand', HandMesh)







    

    


    



    