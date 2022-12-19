import os
import sys
sys.path.append('.')
sys.path.append('..')
import pickle
import numpy as np
import torch
import mano
from torch.utils import data
from torch.utils.data._utils.collate import default_collate
from mano.model import load
import trimesh
import config
from tqdm import tqdm
from copy import deepcopy
from collections import Counter

import random
from tqdm import tqdm
from utils.utils import func_timer, makepath
from tools.obman_utils import fast_load_obj
from tools.condition_utils import thumb_query_points
        
        
class ObMan_preprocess(data.Dataset):
    
    def __init__(self,
                 root,
                 shapenet_root,
                 split='train',
                 joint_nb=21,
                 mini_factor=None,
                 use_cache=False,
                 root_palm=False,
                 mode='all',
                 segment=False,
                 use_external_points=True,
                 apply_obj_transform=True):
        self.split = split
        self.mode = mode
        self.root_all = root
        self.root = os.path.join(root, split)
        self.root_palm = root_palm
        
        if not shapenet_root.endswith('/'):
            # shapenet_root = shapenet_root[:-1]
            shapenet_root = shapenet_root + '/'
        self.shapenet_root = shapenet_root
        
        self.use_external_points = use_external_points
        if mode == 'all':
            self.rgb_folder = os.path.join(self.root, "rgb")
        elif mode == 'obj':
            self.rgb_folder = os.path.join(self.root, "rgb_obj")
        elif mode == 'hand':
            self.rgb_folder = os.path.join(self.root, "rgb_hand")
        else:
            raise ValueError(
                'Mode should be in [all|obj|hand], got {}'.format(mode))

        # Cache information
        self.use_cache = use_cache
        self.name = 'obman'
        self.cache_folder = os.path.join('data', 'cache',
                                         '{}'.format(self.name))
        makepath(self.cache_folder)
        self.mini_factor = mini_factor
        self.cam_intr = np.array([[480., 0., 128.], [0., 480., 128.],
                                  [0., 0., 1.]]).astype(np.float32)
        
        self.cam_extr = np.array([[1., 0., 0., 0.], [0., -1., 0., 0.],
                                  [0., 0., -1., 0.]]).astype(np.float32)
        self.joint_nb = joint_nb
        self.segm_folder = os.path.join(self.root, 'segm')

        self.prefix_template = '{:08d}'
        self.meta_folder = os.path.join(self.root, "meta")
        self.coord2d_folder = os.path.join(self.root, "coords2d")

        # Define links on skeleton
        self.links = [(0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12),
                      (0, 13, 14, 15, 16), (0, 17, 18, 19, 20)]

        # Load mano faces
        self.faces = {}
        for side in ['left', 'right']:
            with open(os.path.join(config.mano_root,'mano_faces_{}.pkl'.format(side)), 'rb') as p_f:
                self.faces[side] = pickle.load(p_f)

        # NOTE:shapenet model_normalized.pkl path
        self.shapenet_template = self.shapenet_root + '{}/{}/models/model_normalized.pkl'
        self.load_dataset()
                
        return
    
    def _get_image_path(self, prefix):
        image_path = os.path.join(self.rgb_folder, '{}.jpg'.format(prefix))

        return image_path
        
    def _get_obj_path(self, class_id, sample_id):
        shapenet_path = self.shapenet_template.format(class_id, sample_id)
        return shapenet_path
    
    def load_dataset(self):
        # NOTE:MANO right hand model path -- use config
        pkl_path = config.mano_dir 
        if not os.path.exists(pkl_path):
            pkl_path = '../' + pkl_path
        cache_path = os.path.join(
            self.cache_folder, '{}_{}_mode_{}.pkl'.format(
                self.split, self.mini_factor, self.mode))
        cache_path_3d = os.path.join(
            self.cache_folder, '{}_3D_{}_mode_{}.pkl'.format(
                self.split, self.mini_factor, self.mode))
        if os.path.exists(cache_path) and self.use_cache:
            with open(cache_path, 'rb') as cache_f:
                annotations = pickle.load(cache_f)
            print('Cached information for dataset {} loaded from {}'.format(
                self.name, cache_path))
            with open(cache_path_3d, 'rb') as cache_f:
                annotations_3D = pickle.load(cache_f)
            print('Cached 3D annotation information for dataset {} loaded from {}'.format(
                self.name, cache_path_3d))
            self.annotations = annotations
            
        else:
            annotations_3D = None
            idxs = [
                int(imgname.split('.')[0])
                for imgname in sorted(os.listdir(self.meta_folder))
            ]

            if self.mini_factor:
                mini_nb = int(len(idxs) * self.mini_factor)
                idxs = idxs[:mini_nb]

            prefixes = [self.prefix_template.format(idx) for idx in idxs]
            print('Got {} samples for split {}'.format(len(idxs), self.split))

            image_names = []
            all_joints2d = []
            all_joints3d = []
            hand_sides = []
            hand_poses = []
            hand_pcas = []
            hand_verts3d = []
            obj_paths = []
            obj_transforms = []
            meta_infos = []
            depth_infos = []
            for idx, prefix in enumerate(tqdm(prefixes)):
                meta_path = os.path.join(self.meta_folder,
                                         '{}.pkl'.format(prefix))
                with open(meta_path, 'rb') as meta_f:
                    meta_info = pickle.load(meta_f)
                    image_path = self._get_image_path(prefix)
                    image_names.append(image_path)
                    all_joints2d.append(meta_info['coords_2d'])
                    all_joints3d.append(meta_info['coords_3d'])
                    hand_verts3d.append(meta_info['verts_3d'])
                    hand_sides.append(meta_info['side'])
                    hand_poses.append(meta_info['hand_pose'])
                    hand_pcas.append(meta_info['pca_pose'])
                    depth_infos.append({
                        'depth_min':
                        meta_info['depth_min'],
                        'depth_max':
                        meta_info['depth_max'],
                        'hand_depth_min':
                        meta_info['hand_depth_min'],
                        'hand_depth_max':
                        meta_info['hand_depth_max'],
                        'obj_depth_min':
                        meta_info['obj_depth_min'],
                        'obj_depth_max':
                        meta_info['obj_depth_max']
                    })
                    obj_path = self._get_obj_path(meta_info['class_id'],
                                                  meta_info['sample_id'])

                    obj_paths.append(obj_path)
                    obj_transforms.append(meta_info['affine_transform'])

                    meta_info_full = {
                        'obj_scale': meta_info['obj_scale'],
                        'obj_class_id': meta_info['class_id'],
                        'obj_sample_id': meta_info['sample_id']
                    }
                    if 'grasp_quality' in meta_info:
                        meta_info_full['grasp_quality'] = meta_info[
                            'grasp_quality']
                        meta_info_full['grasp_epsilon'] = meta_info[
                            'grasp_epsilon']
                        meta_info_full['grasp_volume'] = meta_info[
                            'grasp_volume']
                    meta_infos.append(meta_info_full)

            annotations = {
                'depth_infos': depth_infos,
                'image_names': image_names,
                'joints2d': all_joints2d,
                'joints3d': all_joints3d,
                'hand_sides': hand_sides,
                'hand_poses': hand_poses,
                'hand_pcas': hand_pcas,
                'hand_verts3d': hand_verts3d,
                'obj_paths': obj_paths,
                'obj_transforms': obj_transforms,
                'meta_infos': meta_infos
            }
            print('class_nb: {}'.format(
                np.unique(
                    [(meta_info['obj_class_id']) for meta_info in meta_infos],
                    axis=0).shape))
            unique_obj = np.unique(
                    [(meta_info['obj_class_id'], meta_info['obj_sample_id'])
                     for meta_info in meta_infos],
                    axis=0)
            print('sample_nb : {}'.format(unique_obj.shape))
            obj_meshes = self.load_obj_meshes(unique_obj)
            # import pdb; pdb.set_trace()
            annotations['obj_meshes'] = obj_meshes
            
            with open(cache_path, 'wb') as fid:
                pickle.dump(annotations, fid)
            print('Wrote cache for dataset {} to {}'.format(
                self.name, cache_path))

        # Set dataset attributes
        #import pdb; pdb.set_trace() #(CHECK what is the loaded annotations
        all_objects = [
            obj[:-7].split('/')[-1].split('_')[0]
            for obj in annotations['obj_paths']
        ]
        selected_idxs = list(range(len(all_objects)))
        obj_paths = [annotations['obj_paths'][idx] for idx in selected_idxs]
        image_names = [
            annotations['image_names'][idx] for idx in selected_idxs
        ]
        joints3d = [annotations['joints3d'][idx] for idx in selected_idxs]
        joints2d = [annotations['joints2d'][idx] for idx in selected_idxs]
        hand_sides = [annotations['hand_sides'][idx] for idx in selected_idxs]
        hand_pcas = [annotations['hand_pcas'][idx] for idx in selected_idxs]
        hand_verts3d = [
            annotations['hand_verts3d'][idx] for idx in selected_idxs
        ]
        obj_transforms = [
            annotations['obj_transforms'][idx] for idx in selected_idxs
        ]
        meta_infos = [annotations['meta_infos'][idx] for idx in selected_idxs]
        obj_meshes = annotations['obj_meshes']
        if 'depth_infos' in annotations:
            has_depth_info = True
            depth_infos = [
                annotations['depth_infos'][idx] for idx in selected_idxs
            ]
        else:
            has_depth_info = False
        objects = [
            obj.split('_')[0]
            for obj in set([obj[:-7].split('/')[-1] for obj in obj_paths])
        ]
        unique_objects = set(objects)
        print('Got {} out instances of {} unique objects {}'.format(
            len(objects), len(unique_objects), unique_objects))
        freqs = Counter(objects)
        print(freqs)
        if has_depth_info:
            self.depth_infos = depth_infos
        self.image_names = image_names
        self.joints2d = joints2d
        self.joints3d = joints3d
        self.hand_sides = hand_sides
        self.hand_pcas = hand_pcas
        self.hand_verts3d = hand_verts3d
        self.obj_paths = obj_paths
        self.obj_transforms = obj_transforms
        self.meta_infos = meta_infos
        self.obj_meshes = obj_meshes
        self.annotations_3D = annotations_3D
        # Initialize cache for center and scale in case objects are used
        self.center_scale_cache = {}
        
    def load_obj_meshes(self, unique_obj):
        obj_meshes = {}
        for idx in tqdm(range(unique_obj.shape[0]), desc="Loading object meshes"):
            class_id = unique_obj[idx][0]
            sample_id = unique_obj[idx][1]
            mesh = self.get_obj_mesh(class_id, sample_id)
            if class_id not in obj_meshes:
                obj_meshes[class_id] = {}
            obj_meshes[class_id][sample_id] = mesh
        return obj_meshes
    
    def get_obj_mesh(self, obj_class_id, obj_sample_id):
        # model_path = self.obj_paths[idx]
        # model_path = model_path.replace(
        #     config.SHAPENET_ROOT, # NOTE:shapenet path
        #     self.shapenet_root)
        model_path = os.path.join(self.shapenet_root, obj_class_id, obj_sample_id, 'models/model_normalized.pkl')
        model_path_obj = model_path.replace('.pkl', '.obj')
        if os.path.exists(model_path):
            with open(model_path, 'rb') as obj_f:
                mesh = pickle.load(obj_f)
        elif os.path.exists(model_path_obj):
            with open(model_path_obj, 'r') as m_f:
                mesh = fast_load_obj(m_f)[0]
        else:
            raise ValueError(
                'Could not find model pkl or obj file at {}'.format(
                    model_path.split('.')[-2]))
        return mesh
    
    def get_verts3d(self, idx):
        # NOTE: [hand verts3d]
        # --self.hand_verts3d
        # --self.cam_extr
        verts3d = self.hand_verts3d[idx]
        verts3d = self.cam_extr[:3, :3].dot(verts3d.transpose()).transpose()
        return verts3d
    
    def get_sides(self, idx):
        return self.hand_sides[idx]
    
    def get_faces3d(self, idx):
        # NOTE: [hand faces]
        faces = self.faces[self.get_sides(idx)]
        return faces
    
    def get_sample_obj_info(self, idx):
        model_path = self.obj_paths[idx]
        model_path_split = model_path.split('/')
        class_id = model_path_split[-4]
        sample_id = model_path_split[-3]
        return class_id, sample_id
    
    def get_sample_obj_mesh(self, idx):
        class_id, sample_id = self.get_sample_obj_info(idx)
        mesh = self.obj_meshes[class_id][sample_id]
        return mesh
    
    def get_obj_verts_faces(self, idx):
        # NOTE: [obj_verts, obj_faces] -> obj_meshes
        # model_path = self.obj_paths[idx]
        # model_path = model_path.replace(
        #     config.SHAPENET_ROOT, # NOTE:shapenet path
        #     self.shapenet_root)
        # model_path_obj = model_path.replace('.pkl', '.obj')
        # if os.path.exists(model_path):
        #     with open(model_path, 'rb') as obj_f:
        #         mesh = pickle.load(obj_f)
        # elif os.path.exists(model_path_obj):
        #     with open(model_path_obj, 'r') as m_f:
        #         mesh = fast_load_obj(m_f)[0]
        # else:
        #     raise ValueError(
        #         'Could not find model pkl or obj file at {}'.format(
        #             model_path.split('.')[-2]))
        mesh = self.get_sample_obj_mesh(idx)
        verts = mesh['vertices']
        # Apply transforms
        obj_transform = self.obj_transforms[idx]
        hom_verts = np.concatenate([verts, np.ones([verts.shape[0], 1])],
                                   axis=1)
        trans_verts = obj_transform.dot(hom_verts.T).T[:, :3]
        trans_verts = self.cam_extr[:3, :3].dot(
            trans_verts.transpose()).transpose()
        return np.array(trans_verts).astype(np.float32), np.array(
            mesh['faces']).astype(np.int16)
        
    
    def __len__(self):
        return len(self.hand_verts3d)
        
    def __getitem__(self, idx):
        # TODO: code the __getitem__() to encode the hand_verts and object meshes
        sample = {}
        if self.use_cache:
            hand_verts3d = self.annotations_3D['hand_verts']
        else:
            hand_verts3d = self.get_verts3d(idx)
            hand_faces = self.get_faces3d(idx)
            obj_verts, obj_faces = self.get_obj_verts_faces(idx)
            sample_id = idx
        return sample
    

def preprocess(dataset_root, split='train'):
    
    
    trainset = ObMan_preprocess(root = dataset_root, 
                                shapenet_root = config.SHAPENET_ROOT, 
                                split=split)
    count_2048 = 0
    count_3000 = 0
    right_hand = 0
    
    pbar = tqdm(range(trainset.__len__()))
    annotations_3D = {'hand_verts':[],
                      'obj_verts':[]}
    
    for idx in pbar:
        obj_path = trainset.obj_paths[idx]
        hand_verts3d = trainset.get_verts3d(idx)
        hand_faces = trainset.get_faces3d(idx)
        annotations_3D['hand_verts'].append(hand_verts3d)
        # annotations_3D['hand_faces'].append(hand_faces) # NOTE:不需要专门解码'hand_faces'， 因为经统计，train - 全是right
        obj_verts, obj_faces = trainset.get_obj_verts_faces(idx)
        annotations_3D['obj_verts'].append(obj_verts)
        # 不需要专门解码obj_faces, 因为只会在可视化的时候使用，而annotations中有存meshes，可以直接从meshes中找
        if obj_verts.shape[0] < 3000:
            count_3000 = count_3000 + 1
        if obj_verts.shape[0] < 2048:
            count_2048 = count_2048 + 1
        if trainset.hand_sides[idx] == 'right':
            right_hand = right_hand + 1
        pbar.set_postfix_str(f"obj_verts<3000: num={count_3000};  obj_verts<2048: num={count_2048}; right_hand: num={right_hand}")
        # HandMesh.export()
    cache_path = os.path.join(
            trainset.cache_folder, '{}_3D_{}_mode_{}.pkl'.format(
                trainset.split, trainset.mini_factor, trainset.mode))
    with open(cache_path, 'wb') as fid:
        pickle.dump(annotations_3D, fid)
    print('Wrote cache for dataset {} to {}'.format(
        trainset.name, cache_path))
    print(f"obj_verts < 3000: {count_3000}; obj_verts < 2048: {count_2048}; right_hand: {right_hand}")
    return

    
    
def thumb_condition(dataset_root, output_path, split='train'):
    dataset = ObMan_preprocess(root=dataset_root, 
                               shapenet_root=config.SHAPENET_ROOT,
                               split=split,
                               use_cache=True)
    
    pbar = tqdm(range(dataset.__len__()))
    
    for idx in pbar:
        hand_verts = dataset.annotations_3D['hand_verts'][idx]
        hand_faces = dataset.faces['right']
        
        obj_mesh = dataset.get_sample_obj_mesh(idx)
        obj_verts = dataset.annotations_3D['obj_verts'][idx]
        obj_faces = np.array(obj_mesh['faces']).astype(np.int16)
        import pdb; pdb.set_trace()
        
        HandMesh = trimesh.Trimesh(vertices=hand_verts, faces=hand_faces)
        ObjMesh = trimesh.Trimesh(vertices=obj_verts, faces=obj_faces)
        thumb_query_points(HandMesh, ObjMesh)
    
    
    
    return

        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--preprocess', action='store_true')
    parser.add_argument('--condition', action='store_true')
    args = parser.parse_args()
    
    dataset_root = config.OBMAN_ROOT
    output_root = config.obman_visual_dir
    output_folder = "Dataset_sample"
    output_path = os.path.join(output_root, output_folder)
    makepath(output_path)
    
    if args.preprocess:
        preprocess(dataset_root, split=args.split)
    if args.condition:
        thumb_condition(dataset_root, output_path, split=args.split)
    