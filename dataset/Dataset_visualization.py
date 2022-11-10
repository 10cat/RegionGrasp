import math
from multiprocessing.sharedctypes import Value
from symbol import dotted_as_name
import sys
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
# from mano.utils import Mesh
import trimesh
import config
from copy import deepcopy
from trimesh import viewer
# from pycaster import pycaster
from tqdm import tqdm

import mano

import time
import random
from utils.utils import func_timer, makepath
from utils.visualization import colors_like, visual_sdf, visual_obj_contact_regions
# from vedo import show

list_union = lambda l1, l2: list(set(l1 + l2))
list_diff = lambda l1, l2: list(set(l1) - set(l2))

class GrabNetObjVisual(data.Dataset):
    @func_timer
    def __init__(self,
                 dataset_dir,
                #  config,
                 ds_name='train',
                 dtype=torch.float32,
                 only_params = False,
                 load_on_ram = False,
                 num_mask = 1,
                 one_sample=False,
                 one_region=False,
                 visual_centers=True):

    
        super().__init__()

        self.only_params = only_params

        self.ds_name = ds_name
        self.num_mask = num_mask
        self.ds_path = os.path.join(dataset_dir, ds_name)
        self.mano_path = config.mano_dir
        self.obj_mesh_dir = config.obj_mesh_dir
        # self.obj_sample_dir = "/home/datassd/yilin/GrabNet/tools/object_meshes/sample_info/"
        self.output_path = config.dataset_visual_dir

        # TODO: obtain object meshes dict (k: object_name, v: corresponding decimated mesh)
        self.obj_meshes_dict = self.load_obj_meshes() # 
        
        # TODO: obtain frame_sample dict (k: object_name, v: [frame_samples] of the corresponding object)
        ## frame_samples: {'obj_candidates':[], 'obj_centered_face_ids':[]} => 只需要读取data_sdf
        frame_names = np.load(os.path.join(self.ds_path, 'frame_names.npz'))['frame_names']
        self.frame_names = frame_names
        # import pdb; pdb.set_trace()
        
        frame_sample_dict = {}
        for fname in frame_names:
            name = fname.split('/')[-2]
            obj_name = name.split('_')[0]
            if obj_name not in frame_sample_dict:
                frame_sample_dict[obj_name] = []
            frame_sample_dict[obj_name].append(os.path.join(dataset_dir, fname.replace('data', 'data_sdf').replace('npz', 'npy')))

        self.frame_sample_dict = frame_sample_dict

        # import pdb; pdb.set_trace()
        self.obj_names_list = list(self.frame_sample_dict.keys())

        self.one_region = one_region
        self.one_sample = one_sample
        self.visual_centers = visual_centers
        

    def load_obj_meshes(self):
        """
        object meshes dict (k: object_name, v: corresponding decimated mesh)
        """
        obj_meshes = {}
        obj_files = os.listdir(self.obj_mesh_dir)
        for obj_file in obj_files:
            obj_name = obj_file.split('.')[0]
            obj_meshes[obj_name] = trimesh.load(os.path.join(self.obj_mesh_dir, obj_file))

        return obj_meshes

    def obj_grasp_regions(self, obj_name):
        """
        Returns:
        - obj_centers (-> paint yellow): (list)
        - obj_regions (-> paint blue): (list)
        """
        frame_sample_paths = self.frame_sample_dict[obj_name]
        obj_region_dict = {}
        obj_centers_list = []
        for idx, path in enumerate(tqdm(frame_sample_paths, desc=f'{obj_name}')):
            data = np.load(path, allow_pickle=True)
            # import pdb; pdb.set_trace()
            sample = data.tolist()
            surrounds = sample['candidates']
            centers = sample['obj_faces_ids'].tolist()
            if self.one_region:
                obj_centers_list = [centers[0]]
                obj_region_dict = {centers[0]: surrounds[0]}
                break

            elif obj_centers_list is None:
                obj_centers_list = centers
                obj_region_dict = {c: surrounds[idx] for idx, c in enumerate(centers)}
                
            else:
                
                diff = list(set(centers) - set(obj_centers_list))
                for c in diff:
                    idx = centers.index(c)
                    obj_region_dict[c] = surrounds[idx]
                obj_centers_list = obj_centers_list + diff

            if self.one_sample:
                break  
            
        obj_regions = []

        for centers, surrounds in obj_region_dict.items():
            obj_regions = list(set(obj_regions + surrounds))

        obj_centers = obj_centers_list
        
        return obj_centers, obj_regions

    def visual_obj_regions(self, obj_name):
        obj_centers, obj_regions = self.obj_grasp_regions(obj_name)
        ObjMesh = self.obj_meshes_dict[obj_name]
        colors = config.colors

        # import pdb; pdb.set_trace()

        ObjMesh.visual.face_colors = colors_like(colors['grey'])
        ObjMesh.visual.face_colors[obj_regions] = colors_like(colors['blue'])
        output_folder = 'obj_regions_' + self.ds_name
        if self.visual_centers:
            output_folder += '_w_centers'
            ObjMesh.visual.face_colors[obj_centers] = colors_like(colors['yellow'])
        if self.one_region:
            output_folder += '_one_region'
        if self.one_sample:
            output_folder += '_one_sample'

        output_dir = os.path.join(self.output_path, output_folder)
        makepath(output_dir)
        ObjMesh.export(os.path.join(output_dir, obj_name + '.ply'))

        
        
    
    def __len__(self):
        return len(self.frame_names)
    

if __name__ == "__main__":
    import argparse
    dataset_dir = "/home/datassd/yilin/GrabNet/data/"

    parser = argparse.ArgumentParser()
    parser.add_argument('--ds_name', default='train')
    parser.add_argument('--visual_centers', action='store_true', default=False)
    # parser.set_defaults(visual_centers=True)
    parser.add_argument('--one_sample', action='store_true', default=False)
    parser.add_argument('--one_region', action='store_true', default=False)
    args = parser.parse_args()

    dataset = GrabNetObjVisual(dataset_dir=dataset_dir, ds_name=args.ds_name, num_mask=1, one_sample=args.one_sample, one_region=args.one_region, visual_centers=args.visual_centers)

    obj_names = dataset.obj_names_list

    for obj_name in obj_names:
        dataset.visual_obj_regions(obj_name)

