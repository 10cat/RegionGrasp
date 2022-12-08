import os
import sys
sys.path.append('.')
sys.path.append('..')
import config
import numpy as np
import torch
import trimesh
import mano
from mano.model import load

from dataset.Dataset_origin import GrabNetDataset_orig
from dataset.data_utils import m2m_intersect, visual_inter, visual_mesh, visual_mesh_region
from utils.utils import func_timer, makepath
import matplotlib.pyplot as plt
from tqdm import tqdm

if __name__ == "__main__":
    import argparse
    import random
    # random.seed(1024)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--ds_name', type=str, required=True)
    
    
    args = parser.parse_args()
    
    if args.dataset== "GrabNet":
        dataset_root = config.GrabNet_ROOT
    output_root = os.path.join(config.visual_root, f"{args.dataset}_visual") 
    mano_path = config.mano_dir
    
    rh_model = load(model_path=mano_path, 
                    is_rhand=True, 
                    num_pca_comps=45, 
                    flat_hand_mean=True)
    if args.dataset == "GrabNet":
        dataset = GrabNetDataset_orig(dataset_root, ds_name=args.ds_name, obj_meshes_folder='decimate_meshes', dtype=np.float32)
    visual_folder = os.path.join(output_root, 'thumb_condition', 'Annotations_wtf', args.ds_name)
    makepath(visual_folder)
    
    annot_frame_names = [os.path.join(dataset.ds_root, fname.replace('data', 'data_sdf').replace('npz', 'npy')) for fname in dataset.frame_names_orig]
    
    total_samples = np.array(range(dataset.__len__()))
    # import pdb; pdb.set_trace()
    random_visual_samples = np.random.choice(total_samples, size=100)
    # random_visual_samples = [0, 1, 2, 3]
    
    for idx in tqdm(random_visual_samples, desc=f'{args.ds_name}'):
        idx = int(idx)
        # import pdb; pdb.set_trace()
        sample = dataset.__getitem__(idx)
        hand_faces = rh_model.faces
        hand_verts = sample['verts_rhand']
        HandMesh = trimesh.Trimesh(vertices=hand_verts, faces=hand_faces)
        
        obj_name = dataset.frame_objs[idx]
        
        ObjMesh = dataset.object_meshes[obj_name]
        
        obj_verts_orig = ObjMesh.vertices
        obj_trans = sample['trans_obj']
        obj_rotmat = sample['root_orient_obj_rotmat'][0]
        obj_verts = np.matmul(obj_verts_orig, obj_rotmat) + obj_trans
        obj_faces = ObjMesh.faces
        
        sdf_annot_file = np.load(annot_frame_names[idx], allow_pickle=True)
        sdf_annot = sdf_annot_file.tolist()
        
        
        center_ids = sdf_annot['thumb_center_ids']
        centers = sdf_annot['candidate_centers']
        candidates = sdf_annot['candidates']
        import pdb; pdb.set_trace()
        
        visual_mesh(ObjMesh, bg_color='grey', mark_region=candidates[0], mark_color='pink')
        visual_mesh_region(ObjMesh, fs=centers[0], color='red')
        
        obj_out_path = os.path.join(visual_folder, f'{obj_name}.ply')
        ObjMesh.export(obj_out_path)
            
        
        
        
    