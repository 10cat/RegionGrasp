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
from dataset.data_utils import MeshInitialize, MeshTransform, m2m_intersect 
from utils.visualization import visual_inter, visual_mesh, visual_mesh_region
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
        dataset = GrabNetDataset_orig(dataset_root, 
                                      ds_name=args.ds_name, 
                                      dtype=np.float32, 
                                      frame_names_file='frame_names.npz',
                                      grabnet_thumb=False,
                                      obj_meshes_folder='decimate_meshes')
    ObjTransform = MeshTransform(args.dataset) 
    MeshInit = MeshInitialize(dataset, args.dataset)
        
    visual_folder = os.path.join(output_root, 'thumb_condition', 'Annotations', args.ds_name)
    makepath(visual_folder)
    
    annot_frame_names = [os.path.join(dataset.ds_root, fname.replace('data', 'data_sdf').replace('npz', 'npy')) for fname in dataset.frame_names_orig]
    
    total_samples = np.array(range(5000))
    # import pdb; pdb.set_trace()
    random_visual_samples = np.random.choice(total_samples, size=100)
    # random_visual_samples = [0, 1, 2, 3]
    
    for idx in tqdm(random_visual_samples, desc=f'{args.ds_name}'):
        idx = int(idx)
        # import pdb; pdb.set_trace()
        sample = dataset.__getitem__(idx)
        obj_name = dataset.frame_objs[idx]
        # hand_faces = rh_model.faces
        # hand_verts = sample['verts_rhand']
        # HandMesh = trimesh.Trimesh(vertices=hand_verts, faces=hand_faces)
        
        # obj_name = dataset.frame_objs[idx]
        
        # ObjMesh = dataset.object_meshes[obj_name]
        # TODO: ----- obtain original hand mesh and object mesh using class 'MeshInitialize'--- #
        HandMesh, obj_mesh = MeshInit(sample)
        
        # obj_verts_orig = ObjMesh.vertices
        # obj_trans = sample['trans_obj']
        # obj_rotmat = sample['root_orient_obj_rotmat'][0]
        # obj_verts = np.matmul(obj_verts_orig, obj_rotmat) + obj_trans
        # obj_faces = ObjMesh.faces
        # ObjMesh = trimesh.Trimesh(vertices=obj_verts, faces=obj_faces)
        # TODO: ----- plug in the object transformation function using class 'ObjTransform'----#
        ObjMesh = ObjTransform(obj_mesh, sample)
        
        
        # ---- Read origin data_sdf file -----#
        sdf_annot_file = np.load(annot_frame_names[idx], allow_pickle=True)
        sdf_annot = sdf_annot_file.tolist()
        # import pdb; pdb.set_trace()
        center_ids = sdf_annot['thumb_center_ids']
        centers = sdf_annot['candidate_centers']
        candidates = sdf_annot['candidates']
        
        if args.ds_name == 'train':
            visual_folder_idx = os.path.join(visual_folder, f'frame_{idx}_{obj_name}')
            makepath(visual_folder_idx)
        visual_mesh(HandMesh, bg_color='skin')
        HandMesh.export(os.path.join(visual_folder_idx, f'{idx}_hand.ply'))
        
        if center_ids:
            for cid, center in enumerate(center_ids):
                visual_mesh(ObjMesh, bg_color='grey', mark_region=candidates[center], mark_color='pink')
                visual_mesh_region(ObjMesh, centers[center], 'red')
                ObjMesh.export(os.path.join(visual_folder_idx, f'{cid}_{obj_name}.ply'))
        # import pdb; pdb.set_trace()
        else:
            visual_mesh(ObjMesh, bg_color='grey')
            ObjMesh.export(os.path.join(visual_folder_idx, f'{obj_name}.ply'))
            print("No thumb region contact annotations!")
            
        
        
        
    