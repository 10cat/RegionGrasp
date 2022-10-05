import math
from symbol import continue_stmt
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
# from mano.utils import Mesh
import trimesh
import config
from copy import deepcopy
from trimesh import viewer
from pycaster import pycaster

import time
from utils.utils import func_timer, makepath
from utils.visualization import visual_sdf, visual_obj_contact_regions
# from vedo import show

to_cpu = lambda tensor: tensor.detach().cpu()

colors = {
    'pink': [1.00, 0.75, 0.80, 1],
    'skin': [0.96, 0.75, 0.69, 1],
    'purple': [0.63, 0.13, 0.94, 1],
    'red': [1.0, 0.0, 0.0, 1],
    'green': [.0, 1., .0, 1],
    'yellow': [1., 1., 0, 1],
    'brown': [1.00, 0.25, 0.25, 1],
    'blue': [.0, .0, 1., 1],
    'white': [1., 1., 1., 1],
    'orange': [1.00, 0.65, 0.00, 1],
    'grey': [0.75, 0.75, 0.75, 1],
    'black': [0., 0., 0., 1],
}
JOINTS_NUM = 15

########################################################################################################
########################################################################################################


class GrabNetDataset(data.Dataset):
    def __init__(self,
                 dataset_dir,
                #  config,
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

        self.ds_name = ds_name

        self.ds_path = os.path.join(dataset_dir, ds_name)
        self.mano_path = config.mano_dir
        self.obj_mesh_dir = config.obj_mesh_dir
        # self.obj_sample_dir = "/home/datassd/yilin/GrabNet/tools/object_meshes/sample_info/"
        self.output_path = os.path.join(config.dataset_visual_dir, self.ds_name)

        self.create_sdf_folders()

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

        # self.frame_sbj_names = self.frame_sbjs

        for idx, name in enumerate(self.sbjs):
            self.frame_sbjs[(self.frame_sbjs == name)] = idx

        self.frame_sbjs=torch.from_numpy(self.frame_sbjs.astype(np.int8)).to(torch.long)

        self.load_on_ram = False
        if load_on_ram:
            self.ds = self[:]
            self.load_on_ram = True

        self.ray_directions, self.ray_sources = None, None

    def create_sdf_folders(self):
        self.sdf_data_path = os.path.join(self.ds_path, 'data_sdf')
        self.data_path = os.path.join(self.ds_path, 'data')
        makepath(self.sdf_data_path)
        for i in range(10):
            # create sub folders that indicate the subject names
            # if self.ds_name != 'train' and i == 8:
            #     continue
            # if i == 8:
                # import pdb; pdb.set_trace()
            name_subfolder = 's' + str(i+1)
            sdf_data_sub_path = os.path.join(self.sdf_data_path, name_subfolder)
            
            makepath(sdf_data_sub_path)

            # create sub-sub folders that indicate the obj names and action names
            data_sub_path = os.path.join(self.data_path, name_subfolder)
            for dir in os.listdir(data_sub_path):
                makepath(os.path.join(sdf_data_sub_path, dir))



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


    def get_prox_sdf(self):
        """
        Obtain signed distance map for both object and hand
        """
        #TODO compute the signed distance obj -> hand mesh
        # -- here to decrease the process time, we have decimated the object meshes in advance
        # https://trimsh.org/trimesh.proximity.html#trimesh.proximity.signed_distance
        obj_hand_sdf = trimesh.proximity.signed_distance(self.HandMesh, self.ObjMesh.vertices) 
        # to have POSITIVE for inside points, NEGATIVE for outside points
        
        #TODO compute the signed distance hand -> obj mesh
        hand_obj_sdf = trimesh.proximity.signed_distance(self.ObjMesh, self.HandMesh.vertices)
        return obj_hand_sdf, hand_obj_sdf

    # def get_obj_sdf_threshold(self, hand_minsdf_vid, obj_hand_sdf):
    #     # HandMeshQuery = trimesh.proximity.ProximityQuery(self.HandMesh)
    #     ObjMeshQuery = trimesh.proximity.ProximityQuery(self.ObjMesh)
    #     # _, obj_hand_dist, hand_tri_ids = HandMeshQuery.on_surface(points=self.ObjMesh.vertices) # for obj signed distance
    #     _, hand_obj_dist, obj_tri_ids = ObjMeshQuery.on_surface(points=self.HandMesh.vertices)
    #     obj_th_face_id = int(obj_tri_ids[hand_minsdf_vid])
    #     obj_th_face = self.ObjMesh.faces[obj_th_face_id]
    #     verts_indices = obj_th_face.tolist()
    #     obj_th_sdfs = obj_hand_sdf[verts_indices]
    #     sdf_threshold = max(obj_th_sdfs)

    #     return sdf_threshold

    def hand_sdf_threshold(self, hand_obj_sdf, alpha=0.5):
        """
        Select the top x% of the points 
        """
        mean = hand_obj_sdf.mean()
        sigma = hand_obj_sdf.std()
        threshold = float(mean +  alpha*sigma)
        while threshold > 0:
            alpha = alpha - 0.1
            threshold = float(mean +  alpha*sigma)
        # import pdb; pdb.set_trace()
        return threshold

    def hand_contact_region(self, hand_obj_sdf):
        """
        Obtain the indices of the selected hand vertices
        """
        threshold = self.hand_sdf_threshold(hand_obj_sdf, alpha= config.hand_sdf_th_alpha)
        eps = 0.005
        ids_eps = np.where(hand_obj_sdf < -eps)[0] #TODO 去掉穿模以及几乎将要穿模的情况
        ids_th = np.where(hand_obj_sdf > threshold)[0] 
        ids = np.intersect1d(ids_eps, ids_th)
        # import pdb; pdb.set_trace()
        indices = ids.squeeze().tolist()

        #[checkpoint] visualization check
        visual_sdf(self.HandMesh, hand_obj_sdf, vert_indices=indices, check=config.check)

        return threshold, indices

    def object_contact_centers(self, vert_indices):
        """
        Find the candidate contact region center faces by matching the closest faces on the obj surface that corresponds to the given hand vertices
        """
        hand_verts = self.HandMesh.vertices[vert_indices]
        ObjMeshQuery = trimesh.proximity.ProximityQuery(self.ObjMesh)
        _, dists, face_ids = ObjMeshQuery.on_surface(points=hand_verts)
        face_ids = np.unique(face_ids)
        #TODO: (may enable random sampling of those faces if needed to reduce the scale)

        return dists, face_ids

    def get_bounding_radius(self, points, faces, obj_vertices, scale=1): 
        point1s, point2s = obj_vertices[faces][:, 1], obj_vertices[faces][:, 2]
        dist1 = np.linalg.norm((points - point1s), axis=1)
        dist2 = np.linalg.norm((points - point2s), axis=1) # np.linalg.norm的结果是flatten之后的
        # import pdb; pdb.set_trace()
        dists = np.concatenate((dist1, dist2))
        dists = np.max(dists)

        radius = scale * dists.mean()
        return radius

    def rtree_bounds(self, points, radius, r_depth=None):
        # axis aligned bounds
        if r_depth is not None:
            distance_vector = [r_depth, radius, radius]
            bounds = np.column_stack((points - distance_vector, points + distance_vector))
        else:
            bounds = np.column_stack((points - radius, points + radius))

        return bounds
        

    def object_contact_regions(self, obj_face_ids, radius=0.002):
        """
        Form the contact unit regions on object surface centered at given faces based on searching the neighors.
        """
        rtree = self.ObjMesh.triangles_tree
        faces = self.ObjMesh.faces[obj_face_ids]
        obj_vertices = self.ObjMesh.vertices
        points = obj_vertices[faces[:, 0]] # 先选取每个候选面上的第一个vertex作为query point

        # axis aligned bounds
        bounds = self.rtree_bounds(points, radius, r_depth=config.r_depth) 

        # line segments that intersect axis aligned bounding box
        candidates = [list(rtree.intersection(b)) for b in bounds]
        
        # import pdb; pdb.set_trace()
        # check visualization
        visual_obj_contact_regions(self.ObjMesh, obj_face_ids, candidates, all=True, check=config.check)

        return candidates

    def adjacency_to_faces(self, adj_candidates):
        face_candidates = []
        for adj_indices in adj_candidates:
            adjacency = self.ObjMesh.face_adjacency[adj_indices]
            # import pdb; pdb.set_trace()
            f_indices = adjacency.reshape(-1)
            f_indices = np.unique(f_indices).tolist() # 去掉重复的edge共面
            face_candidates.append(f_indices)
        return face_candidates

    def object_contact_regions_adjaceny(self, obj_face_ids, radius=0.005):
        """
        Form the contact unit regions on object surface centered at given faces based on searching the neighors.
        """
        rtree = self.ObjMesh.face_adjacency_tree
        faces = self.ObjMesh.faces[obj_face_ids]
        obj_vertices = self.ObjMesh.vertices
        points = obj_vertices[faces[:, 0]] # 先选取每个候选面上的第一个vertex作为query point

        # radius = self.get_bounding_radius(points, faces, obj_vertices) # 先统计一个大致经验值
        bounds = self.rtree_bounds(points, radius, r_depth=config.r_depth) 

        # line segments that intersect axis aligned bounding box
        adj_candidates = [list(rtree.intersection(b)) for b in bounds]

        #TODO turn adjacency candidates to faces
        candidates = self.adjacency_to_faces(adj_candidates)
        
        # import pdb; pdb.set_trace()
        # check visualization
        visual_obj_contact_regions(self.ObjMesh, obj_face_ids, candidates, all=True, check=config.check)

        return candidates
    
    # @func_timer
    def sdf_annotations(self):

        # Get signed distance map for object(obj_hand_sdf) and hand(hand_obj_sdf)
        obj_hand_sdf, hand_obj_sdf = self.get_prox_sdf()

        #TODO: Obtain certain proportion of top closest points based on signed distance
        # threshold = self.get_hand_sdf_threshold(hand_obj_sdf)
        threshold, select_indices = self.hand_contact_region(hand_obj_sdf)

        #TODO: Find correspoding closest faces on object mesh
        hand_obj_dists, obj_face_ids = self.object_contact_centers(select_indices)
        # import pdb; pdb.set_trace()
        
        #TODO: Find the corresponding closest faces as the contact unit region centers; Form the contact unit regions based on neighborhood searching 
        # candidates = self.object_contact_regions(obj_face_ids) # TODO triangles-tree based search contact_regions
        candidates = self.object_contact_regions_adjaceny(obj_face_ids) # TODO face_adjacency_tree based search contact_regions

        # hand visualization
        visual_sdf(self.HandMesh, hand_obj_sdf, vert_indices=select_indices)

        return obj_hand_sdf, hand_obj_sdf, candidates, obj_face_ids

    # @func_timer
    def get_new_annotations(self, idx, sample):
        # raw_datas = {k: self.ds[k][idx] for k in self.ds.keys()}
        # sample = self.__getitem__(idx)
        sbj_name = self.frame_sbjs[idx]
        obj_name = self.frame_objs[idx]
        act_name = self.frame_acts[idx]
        number = self.frame_numbers[idx]
        
        
        vtemp = self.sbj_vtemp[sbj_name]
        betas = self.sbj_betas[sbj_name].reshape(1, -1) # shape (1, 10)

        """
        [Obtain object centric coordinates of both hand and objects]
        """
        # 该subject下的先得到标准template mesh --> rhand_mesh
        rhand_sbj_model = load(model_path = self.mano_path,
                          is_rhand = True,
                          num_pca_comps=45,
                          betas = betas,
                          v_template = vtemp,
                          flat_hand_mean=True)
        

        # 读取object template mesh
        obj_path = os.path.join(self.obj_mesh_dir, obj_name+'.ply')
        # ObjMesh = o3d.io.read_triangle_mesh(obj_path) # 先用o3d读再转trimesh？？
        ObjMesh = trimesh.load(obj_path)
        # ObjMesh = trimesh.load(obj_path)

        # 在object_centric坐标系下旋转、平移手和物体
        # obj_verts = np.matmul(ObjMesh.vertices, sample['root_orient_obj_rotmat'][0].T) + sample['trans_obj']
        obj_verts = np.matmul(ObjMesh.vertices, sample['root_orient_obj_rotmat'][0]) + sample['trans_obj'] # the rotation matrix is already transposed
        
        hand_verts = sample['verts_rhand']
        # 注意: raw_datas中的rotmat shape都是(1,3,3)

        # have 
        HandMesh = trimesh.base.Trimesh(vertices=hand_verts, faces=rhand_sbj_model.faces, vc=colors['skin'])
        ObjMesh = trimesh.base.Trimesh(vertices=obj_verts, faces=ObjMesh.faces, vc=colors['grey'])

        self.HandMesh = deepcopy(HandMesh)
        self.ObjMesh = deepcopy(ObjMesh)
        # [checkpoint!!] visualize and check if it is actually "object-centric"
        # initialize a primitive whose center is at (0,0,0) in the coordinate
        originSphere = trimesh.primitives.Sphere(radius=0.001, center=[0,0,0])
        
        """
        Signed distance annotations
        """
        #TODO save the sdf and unit region mask annotations to the npy files 
        data_sdf = {}
        obj_hand_sdf, hand_obj_sdf, candidates, obj_face_ids = self.sdf_annotations()
        data_sdf['obj_hand_sdf'], data_sdf['hand_obj_sdf'], data_sdf['candidates'], data_sdf['obj_faces_ids'] = \
            obj_hand_sdf, hand_obj_sdf, candidates, obj_face_ids
        np.save(self.frame_names_sdf[idx], data_sdf)

        obj_output_path = os.path.join(self.output_path, str(obj_name))
        makepath(obj_output_path)

        if idx % 100 == 0:
            name = os.path.join(obj_output_path, str(act_name) + '_' + 's' + str(int(sbj_name)) + '_' +  str(number))
            self.HandMesh.export( name + '_hand.ply')
            self.ObjMesh.export( name + '_obj.ply')
            originSphere.export( name + '_origin.ply')

        # visualize
        
        # show([ObjMesh, HandMesh, originSphere], axes=1) # use vedo
        # combined = trimesh.util.concatenate([HandMesh, ObjMesh, originSphere])
        # scene = trimesh.scene.Scene(geometry=[HandMesh, ObjMesh, originSphere]) # use trimesh scene viewer
        # scene = trimesh.scene.Scene(geometry=combined)
        # meshviewer = viewer.SceneViewer(scene)

        # meshviewer.toggle_axis()
        

        #return [obj_hand_sdf, hand_obj_sdf]
        return



    def __getitem__(self, idx):
        # if self.frame_sbjs[idx] == 9 and self.ds_name != 'train':
        #     return
        data_out = {k: self.ds[k][idx] for k in self.ds.keys()}
        if not self.only_params:
            if not self.load_on_ram:
                form_disk = self.load_disk(idx)
                data_out.update(form_disk)
        # obj_hand_sdf, hand_obj_sdf = self.get_new_annotations(idx, data_out)
        self.get_new_annotations(idx, data_out)
        
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
    
    dataset = GrabNetDataset(dataset_dir=dataset_dir, ds_name='test')
    # sample = dataset.__getitem__(0)
    idx = 0
    # test_idx = np.arange(0, int(math.ceil(dataset.__len__()/2))).tolist()
    test_idx = np.arange(0, dataset.__len__()).tolist()
    for idx in tqdm(test_idx):
        if idx < 50000: # test 50000
            continue
        dataset.__getitem__(idx)

    # ObjMesh, HandMesh, hoi_info = dataset.get_new_annotations(0)
    # hoi_objfaces_idxs, hoi_objpoints = hoi_info

    # scene = trimesh.scene.Scene(geometry=[ObjMesh, HandMesh])

    # ObjMesh.show()
    
    # meshviewer = viewer.SceneViewer(scene)
    # meshviewer.add_geometry('hand', HandMesh)







    

    


    



    