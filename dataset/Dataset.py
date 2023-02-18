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
import pickle
from tqdm import tqdm
from utils.utils import func_timer, makepath
from dataset.Dataset_origin import GrabNetDataset_orig
from dataset.grabnet_preprocess import GrabNetResample, GrabNetThumb
from dataset.obman_preprocess import ObManResample, ObManThumb

set_seed = lambda val: np.random.seed(val)

def select_ids_dataset(ds_names, seeds=[]):
    for i, name in enumerate(ds_names):
        dataset = GrabNetDataset_orig(dataset_root=None, 
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
   
   
        
class PretrainDataset(data.Dataset):
    def __init__(self, obman_root, shapenet_root, mano_root, grabnet_root, split,  resample_num=2048, rand_each_num=100, use_cache=True):
        
        self.split = split
        # CHECK: get ObMan oject in training set
        self.obman = ObManResample(ds_root=obman_root,
                                  shapenet_root=shapenet_root,
                                  mano_root=mano_root,
                                  split=split,
                                  resample_num=resample_num,
                                  use_cache=True)
        self.obman_objects = self.obman.obj_resampled
        
        
        
        # CHECK: get GrabNet object in training set
        self.grabnet = GrabNetResample(dataset_root=grabnet_root,
                                       ds_name=split,
                                       resample_num=resample_num)
        self.grabnet_objects = self.grabnet.resampled_objs
        
        # CHECK: 合并两个数据集的object
        objects_dict = {}
        objects_dict.update(self.obman_objects)
        objects_dict.update(self.grabnet_objects)
        objects = self.combine(objects_dict)
        self.objects = objects  
        # import pdb; pdb.set_trace() 
        
        # CHECK: 随机生成obj_transform矩阵
        self.rand_each_num = rand_each_num
        objs_num = len(objects)
        total_num = objs_num * rand_each_num
        rotmats = self.rand_rot(total_num)
        self.rotmats = rotmats
        
        if self.split == 'val':
            set_seed(1024 + 2)
            self.trans = self.rand_trans(total_num)
        else:
            self.trans = None
        
        # TODO: permutation
        self.npoints = resample_num
        self.permutation = np.arange(self.npoints)
        
    def combine(self, obj_dict):
        obj_list = []
        for key, val in obj_dict.items():
            if len(val.keys()) > 2:
                for k, v in val.items():
                    obj_list.append(v)
            elif not isinstance(val, dict):
                raise TypeError
            else:
                obj_list.append(val)
            
                
        return obj_list
    
    def rand_rot(self, N):
        if self.split == 'train':
            set_seed(1024)
        elif self.split == 'val':
            set_seed(2048)
        rot_angles = np.random.random([3, N]) * np.pi * 2
        theta_xs, theta_ys, theta_zs = rot_angles[0], rot_angles[1], rot_angles[2]
        RXs = np.stack([np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]]) for x in theta_xs])
        RYs = np.stack([np.array([[np.cos(y), 0, -np.sin(y)], [0, 1, 0], [np.sin(y), 0, np.cos(y)]]) for y in theta_ys])
        RZs = np.stack([np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]]) for z in theta_zs])
        
        Rs = RXs @ RYs @ RZs
        
        return Rs
    
    def rand_trans(self, N):
        trans = np.array([-0.2, -0.2, -0.2]) + np.random.random([N, 3]) * 0.4
        return trans
    
    def pc_centralize(self, pc):
        # TODO: centralize the object_pc by its mean
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        return pc
    
    def pc_transform(self, pc, idx):
        # TODO: rotate the object_pc with pre-defined rotmat
        rotmat = self.rotmats[idx]
        pc = np.matmul(pc, rotmat.T) # [N, 3][3, 3] -> N, 3
        if self.split == 'val':
            tran = self.trans[idx]
            pc = pc + tran
        return pc
    
    def shuffle_points(self, pc, num):
        # TODO: shuffle点的顺序
        np.random.shuffle(self.permutation)
        pc = pc[self.permutation[:num]]
        return pc
    
    def __len__(self):
        # CHECK: 按照obj_transform的长度
        return self.rotmats.shape[0]
    
    def __getitem__(self, idx):
        sample = {}
        # TODO: 根据num_rand_each计算当前idx对应的object
        obj_idx = int(idx / self.rand_each_num)
        obj = self.objects[obj_idx]
        obj_pc = obj['points']
        
        obj_pc = self.pc_centralize(obj_pc)
        obj_pc = self.pc_transform(obj_pc, idx)
        obj_pc = self.shuffle_points(obj_pc, self.npoints)
        sample['input_points'] = torch.from_numpy(obj_pc).to(torch.float32)
        sample['ids'] = idx
        
        return sample
        
        
        
class ObManDataset(ObManThumb):
    def __init__(self, ds_root, shapenet_root, mano_root, split='train', joint_nb=21, mini_factor=None, use_cache=False, root_palm=False, mode='all', segment=False, use_external_points=True, apply_obj_transform=True, expand_times=1, resample_num=2048, object_centric=False, use_mano=False):
        super().__init__(ds_root, shapenet_root, mano_root, split, joint_nb, mini_factor, use_cache, root_palm, mode, segment, use_external_points, apply_obj_transform, expand_times, resample_num, object_centric, use_mano)
        annot_root = os.path.join(self.root, 'thumbHOI_new')
        sample_id = np.load(os.path.join(annot_root, 'samples_id.npy'))
        annotations_thumb = []
        from tqdm import tqdm
        for id in tqdm(sample_id, desc='loading the annotations:'):
            annot_file = os.path.join(annot_root, f'{id}.pkl')
            with open(annot_file, 'rb') as f:
                annotation = pickle.load(f)
            annotations_thumb.append(annotation)
        self.samples_selected = sample_id
        self.annotations_thumb = annotations_thumb
        self.use_mano = use_mano
        
    def get_verts3d_mano(self, idx):
        
        hand_trans = torch.tensor(self.mano_trans[idx])
        hand_rot = torch.tensor(self.mano_rot[idx])
        hand_pose = torch.tensor(self.hand_poses[idx])
        hand_shape = torch.tensor(self.hand_shapes[idx])
        hand_verts = self.rh_mano(betas=hand_shape.reshape(1, -1), global_orient=hand_rot.reshape(1, -1),
                                  hand_pose=hand_pose.reshape(1, -1), transl=hand_trans.reshape(1, -1)).vertices.squeeze(0)
        # import pdb; pdb.set_trace()
        hand_params = torch.cat([hand_rot, hand_pose, hand_trans], dim=0)
        
        return hand_verts, hand_params
        
    def __len__(self):
            return len(self.samples_selected)
    
    
    def __getitem__(self, idx):
        annot = self.annotations_thumb[idx]
        index = self.samples_selected[idx]
        
        sample = {}
        obj_points, obj_trans, face_ids = self.get_obj_resampled_trans(self.meta_infos, self.obj_transforms, index, obj_centric=self.obj_centric)
        
        # DONE: 获取用于计算point2point_signed的obj_point_normals
        # NOTE: 由于obj_points是由原mesh进行了resample之后得到的，所以这里索引采样点所在的面的face_normals作为点的normals
        obj_mesh = self.get_sample_obj_mesh(index)
        obj_verts, _ = self.get_obj_verts_faces(index)
        if self.use_mano:
            hand_verts_torch, hand_params_torch = self.get_verts3d_mano(index)
            hand_verts = hand_verts_torch.numpy().astype(np.float32)
            hand_verts = self.cam_extr[:3, :3].dot(hand_verts.transpose()).transpose()
            hand_faces = self.rh_mano.faces.astype(np.int32)
        else:
            hand_verts = self.get_verts3d(index)
        hand_faces = self.get_faces3d(index)
        if self.obj_centric:
            obj_verts -= obj_trans
            hand_verts -= obj_trans
        
        ObjMesh = trimesh.Trimesh(vertices=obj_verts, faces=obj_mesh['faces'])
        obj_point_normals = ObjMesh.face_normals[face_ids]
        
        contact_point = annot['center_point']
        # mask_center = np.mean(contact_points, axis=0)
        
        sample['sample_id'] = torch.Tensor([index])
        sample['input_pc'] = torch.from_numpy(obj_points).to(torch.float32)
        sample['contact_center'] = torch.from_numpy(contact_point).to(torch.float32)
        sample['obj_point_normals'] = torch.from_numpy(obj_point_normals).to(torch.float32)
        # sample['region_mask'] = torch.from_numpy(region_mask)
        sample['obj_trans'] = torch.from_numpy(obj_trans).to(torch.float32)
        sample['cam_extr'] = torch.from_numpy(self.cam_extr[:3, :3]).to(torch.float32)
        sample['hand_verts'] = torch.from_numpy(hand_verts).to(torch.float32)
        if self.use_mano:
            sample['hand_params'] = hand_params_torch.to(torch.float32)
        
        return sample
    
class ObManDataset_test(ObManDataset):
    def __init__(self, ds_root, shapenet_root, mano_root, split='train', joint_nb=21, mini_factor=None, use_cache=False, root_palm=False, mode='all', segment=False, use_external_points=True, apply_obj_transform=True, expand_times=1, resample_num=2048, object_centric=False, use_mano=False, cfg=None):
        super().__init__(ds_root, shapenet_root, mano_root, split, joint_nb, mini_factor, use_cache, root_palm, mode, segment, use_external_points, apply_obj_transform, expand_times, resample_num, object_centric, use_mano)
        
        test_pred_path = os.path.join(cfg.output_dir, f'{cfg.chkpt}_{cfg.run_mode}set_pred.pkl')
        
        with open(test_pred_path, 'rb') as f:
            data = pickle.load(f)
        self.pred_hand_params = data['recon_params']
            
    def __getitem__(self, idx):
        
        hand_params_pred = self.pred_hand_params[idx]
        
        index = self.samples_selected[idx]
        
        sample = {}
        obj_points, obj_trans, face_ids = self.get_obj_resampled_trans(self.meta_infos, self.obj_transforms, index, obj_centric=self.obj_centric)
        
        # DONE: 获取用于计算point2point_signed的obj_point_normals
        # NOTE: 由于obj_points是由原mesh进行了resample之后得到的，所以这里索引采样点所在的面的face_normals作为点的normals
        obj_mesh = self.get_sample_obj_mesh(index)
        obj_verts, _ = self.get_obj_verts_faces(index)
        if self.use_mano:
            hand_verts_torch, hand_params_torch = self.get_verts3d_mano(index)
            hand_verts = hand_verts_torch.numpy().astype(np.float32)
            hand_verts = self.cam_extr[:3, :3].dot(hand_verts.transpose()).transpose()
            hand_faces = self.rh_mano.faces.astype(np.int32)
        else:
            hand_verts = self.get_verts3d(index)
        hand_faces = self.get_faces3d(index)
        if self.obj_centric:
            obj_verts -= obj_trans
            hand_verts -= obj_trans
            
        sample['hand_verts'] = torch.from_numpy(hand_verts)
        sample['hand_params_pred'] = torch.from_numpy(hand_params_pred)
        sample['obj_trans'] = torch.from_numpy(obj_trans)
        sample['sample_id'] = torch.Tensor([index])
        
        return sample
        
        
class ObManDataset_obj_comp(ObManThumb):
    def __init__(self, ds_root, shapenet_root, mano_root, split='train', joint_nb=21, mini_factor=None, use_cache=False, root_palm=False, mode='all', segment=False, use_external_points=True, apply_obj_transform=True, expand_times=1, resample_num=8192, object_centric=False):
        super().__init__(ds_root, shapenet_root, mano_root, split, joint_nb, mini_factor, use_cache, root_palm, mode, segment, use_external_points, apply_obj_transform, expand_times, resample_num, object_centric)
        annot_root = os.path.join(self.root, 'thumbHOI')
        sample_id = np.load(os.path.join(annot_root, 'samples_id.npy'))
        annotations_thumb = []
        from tqdm import tqdm
        for id in tqdm(sample_id, desc='loading the annotations:'):
            annot_file = os.path.join(annot_root, f'{id}.pkl')
            with open(annot_file, 'rb') as f:
                annotation = pickle.load(f)
            annotations_thumb.append(annotation)
        self.samples_selected = sample_id
        self.annotations_thumb = annotations_thumb
        np.random.seed(1024)
        self.choice = np.random.random(2048)
        # import pdb; pdb.set_trace()
        
    def get_obj_input(self, obj_pc, contact_mask):
        contact = contact_mask > 0
        remained_pc = obj_pc[~contact]
        N = remained_pc.shape[0]
        indices = np.floor(self.choice * N).astype(np.int32)
        sample_pc = [remained_pc[int(index)] for index in indices]
        sample_pc = np.array(sample_pc)
        return sample_pc
        
    def __len__(self):
        return len(self.samples_selected)
    
    def __getitem__(self, idx):
        annot = self.annotations_thumb[idx]
        index = self.samples_selected[idx]
        
        sample = {}
        obj_points, obj_trans, face_ids = self.get_obj_resampled_trans(self.meta_infos, self.obj_transforms, index, obj_centric=self.obj_centric)
        
        # DONE: 获取用于计算point2point_signed的obj_point_normals
        # NOTE: 由于obj_points是由原mesh进行了resample之后得到的，所以这里索引采样点所在的面的face_normals作为点的normals
        obj_mesh = self.get_sample_obj_mesh(index)
        obj_verts, _ = self.get_obj_verts_faces(index)
        hand_verts = self.get_verts3d(index)
        hand_faces = self.get_faces3d(index)
        if self.obj_centric:
            obj_verts -= obj_trans
            hand_verts -= obj_trans
        ObjMesh = trimesh.Trimesh(vertices=obj_verts, faces=obj_mesh['faces'])
        obj_point_normals = ObjMesh.face_normals[face_ids]
        #import pdb; pdb.set_trace()
        # DONE: gt_visualization
        # HandMesh = trimesh.Trimesh(vertices=hand_verts, faces=hand_faces)
        # root = '/home/yilin/Codes/test_visuals/train_trans'
        # makepath(root)
        # ObjMesh.export(os.path.join(root, f'{index}_obj.ply'))
        # HandMesh.export(os.path.join(root, f'{index}_hand.ply'))
            
        # DONE: generate region_mask for o2h
        contact_indices = annot['contact_indices']
        region_mask = np.zeros(obj_points.shape[0])
        region_mask[contact_indices] = 1.
        
        input_pc = self.get_obj_input(obj_points, region_mask)
        
        sample['obj_points'] = torch.from_numpy(obj_points)
        sample['obj_point_normals'] = torch.from_numpy(obj_point_normals)
        sample['hand_verts'] = torch.from_numpy(hand_verts)
        sample['obj_trans'] = torch.from_numpy(obj_trans)
        
        # sample['contact_pc'] = torch.from_numpy(annot['contact_pc'])
        sample['region_mask'] = torch.from_numpy(region_mask)
        
        sample['input_pc'] = torch.from_numpy(input_pc)
        sample['sample_id'] = torch.Tensor([index])
        
        # import pdb; pdb.set_trace()
        
        return sample
    
class GrabNetDataset(GrabNetThumb):
    def __init__(self, dataset_root, ds_name='train', batch_size=32, sample_same=False, mano_path=None, frame_names_file='frame_names.npz', grabnet_thumb=False, obj_meshes_folder='contact_meshes', output_root=None, dtype=torch.float32, only_params=False, load_on_ram=False, resample_num=8192):
        super().__init__(dataset_root, ds_name, mano_path, frame_names_file, grabnet_thumb, obj_meshes_folder, output_root, dtype, only_params, load_on_ram, resample_num)
        self.obj_rotmat = self.ds['root_orient_obj_rotmat']
        self.obj_trans = self.ds['trans_obj']
        
        self.sample_same = sample_same
        self.batch_size = batch_size
        
        # test
        # self.ds = {k: v[:32] for k,v in self.ds.items()}
        
        # frame_data_name = ['verts_rhand']
        # for i, frame_name in enumerate(tqdm(self.frame_names, desc='Loading frames data')):
        #     data = self.get_npz_data(frame_name, to_torch=True)
        #     # import pdb; pdb.set_trace()
        #     for key in frame_data_name:
        #         if key not in self.ds:
        #             self.ds[key] = [data[key]]
        #         else:
        #             self.ds[key].append(data[key])
        #     self.obj_rotmat.append(self.ds['root_orient_obj_rotmat'][i][0])
        #     self.obj_trans.append(self.ds['trans_obj'][i])
        
                
        # self.obj_verts_trans = []
        
        
    def get_obj_verts_faces(self, idx):
        obj_name = self.frame_objs[idx]
        obj_mesh = self.object_meshes[obj_name]
        obj_verts = obj_mesh.vertices
        
        rot_mat_np = np.array(self.obj_rotmat[idx][0])
        trans_np = np.array(self.obj_trans[idx])
        
        obj_verts_trans = np.matmul(obj_verts, rot_mat_np) + trans_np
        
        return obj_verts_trans, obj_mesh.faces
    
    def get_obj_trans(self, idx):
        obj_name = self.frame_objs[idx]
        obj_mesh = self.object_meshes[obj_name]
        
        obj_resp_points = self.resampled_objs[obj_name]['points']
        obj_resp_faces = self.resampled_objs[obj_name]['faces']
        obj_point_normal = obj_mesh.face_normals[obj_resp_faces]
        rot_mat_np = np.array(self.obj_rotmat[idx][0])
        trans_np = np.array(self.obj_trans[idx])
        obj_resp_points_trans = np.matmul(obj_resp_points, rot_mat_np) + trans_np
        obj_point_normal_trans = np.matmul(obj_point_normal, rot_mat_np) + trans_np
        
        return obj_resp_points_trans, obj_point_normal_trans
        
        
    
    def __getitem__(self, idx):
        if self.sample_same:
            idx = idx % self.batch_size
        sample = {}
        data_out = {k: self.ds[k][idx] for k in self.ds.keys()}
        if not self.only_params:
            data = self.get_frames_data(idx, self.frame_names)
            # import pdb; pdb.set_trace()
            data_out.update(data)
        # import pdb; pdb.set_trace() 
        obj_resp_points_trans, obj_point_normal_trans = self.get_obj_trans(idx)
        
        # global_orient, pose, transl = sample['global_orient_rhand_rotmat_f'], sample['fpose_rhand_rotmat_f'], sample['trans_rhand_f']
        # import pdb; pdb.set_trace()
        
        sample['sample_id'] = torch.Tensor([idx]).to(torch.int32)
        sample['input_pc'] = torch.from_numpy(obj_resp_points_trans).to(self.dtype)
        # import pdb; pdb.set_trace()
        sample['contact_center'] = data_out['contact_center'].to(self.dtype)
        sample['obj_point_normals'] = torch.from_numpy(obj_point_normal_trans).to(self.dtype)
        sample['hand_verts'] = data_out['verts_rhand'].to(self.dtype)
        
        # NOTE: 手的顶点对应的键名从verts_rhand变为hand_verts, 并删除掉原来的键名verts_rhand
        # del sample['verts_rhand']
        # self.obj_verts_trans.append(obj_verts_trans)
        # self.obj_rotmat.append(sample['root_orient_obj_rotmat'][0])
        # self.obj_trans.append(sample['trans_obj'])
        # import pdb; pdb.set_trace()
        
        return sample
    
    
        
class GrabNetDataset_sdf(GrabNetDataset_orig):
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
    # dataset = GrabNetDataset(dataset_root=config.DATASET_ROOT, 
    #                         ds_name="train", 
    #                         frame_names_file='frame_names_thumb.npz', 
    #                         grabnet_thumb=False, 
    #                         obj_meshes_folder='decimate_meshes',
    #                         select_ids=False, 
    #                         output_root=None, 
    #                         dtype=torch.float32, 
    #                         only_params=False, 
    #                         load_on_ram=False)
    
    
    # sample = dataset.__getitem__(200031)
    
    dataset = ObManDataset(ds_root=config.OBMAN_ROOT,
                           shapenet_root=config.SHAPENET_ROOT,
                           split='train',
                           use_cache=True,
                           object_centric=True)
    
    length = dataset.__len__()
    
    for idx in range(length):
        sample = dataset.__getitem__(idx)
        

    
    
        