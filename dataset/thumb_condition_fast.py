import os
import sys
sys.path.append('.')
sys.path.append('.')
import config
import numpy as np
import torch
import trimesh
import mano
from mano.model import load
from dataset.Dataset_origin import GrabNetDataset_orig
from dataset.data_utils import faces2verts_no_rep, MeshTransform, MeshInitialize
from utils.visualization import visual_inter, visual_mesh, visual_mesh_region
from utils.utils import func_timer, makepath
import matplotlib.pyplot as plt
from tqdm import tqdm
                
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--ds_name', type=str, required=True)
    parser.add_argument('--sdf_th', type=float, default=-0.005)
    parser.add_argument('--visual_freq', type=int, default=500)
    parser.add_argument('--plot', action='store_true', default=False)
    args = parser.parse_args()
    
    sdf_th = args.sdf_th
    dataset_root = config.DATASET_ROOT
    output_root = config.dataset_visual_dir
    mano_path = config.mano_dir
    rh_model = load(model_path = mano_path,
                             is_rhand = True,
                             num_pca_comps=45,
                             flat_hand_mean=True)
    if args.dataset == "GrabNet":
        dataset = GrabNetDataset_orig(dataset_root, 
                                    ds_name=args.ds_name, 
                                    obj_meshes_folder='decimate_meshes', 
                                    dtype=np.float32)
        
    ObjTransform = MeshTransform(args.dataset) 
    MeshInit = MeshInitialize(dataset, args.dataset)

    visual_folder = os.path.join(config.dataset_visual_dir, 'thumb_condition', 'candidate')
    makepath(visual_folder)
    
    annot_frame_names = [os.path.join(dataset.ds_root, fname.replace('data', 'data_sdf').replace('npz', 'npy')) for fname in dataset.frame_names_orig]
    
    thumb_contact_frame_names = []
    ds_orig = dataset.ds
    ds_thumb = {k:[] for k in list(ds_orig.keys())}
    
    for idx in tqdm(range(dataset.__len__()), desc=f'{args.ds_name}'):
        
        sample = dataset.__getitem__(idx)
        obj_name = dataset.frame_objs[idx]
        # TODO: ----- obtain original hand mesh and object mesh using class 'MeshInitialize'--- #
        HandMesh, obj_mesh = MeshInit(sample)
        
        # import pdb; pdb.set_trace()
        # TODO: ----- plug in the object transformation function using class 'ObjTransform'----#
        ObjMesh = ObjTransform(obj_mesh, sample)
        
        filename = annot_frame_names[idx]
        
        sdf_annot_file = np.load(filename, allow_pickle=True)
        
        sdf_annot = sdf_annot_file.tolist()
        # import pdb; pdb.set_trace()
        if 'obj_faces_ids' in list(sdf_annot.keys()):
            centers = sdf_annot['obj_faces_ids']
        else:
            centers = sdf_annot['candidate_centers']
        # query_points = ObjMesh.triangles_center[centers]
        # HandMeshQuery = trimesh.proximity.ProximityQuery(HandMesh)
        # _, _, closest_faces = HandMeshQuery.on_surface(points=query_points)
        
        # TODO: ----- query points不应该是object的顶点，而应该是thumb顶点 ---- #
        # thumb_vertices = config.hand_comp['thumb'][1]
        thumb_vertices_ids = faces2verts_no_rep(HandMesh.faces[config.thumb_center]) # to ensure the selected region in very close to the thumb contact region
        # thumb_vertices = HandMesh.vertices[thumb_vertices_ids]
        # TODO: ---- select the thumb_vertices using a set sdf threshold! ---- #
        # import pdb; pdb.set_trace()
        hand_sdfs = sdf_annot['hand_obj_sdf']
        contact_ids = np.where(hand_sdfs > sdf_th)[0].tolist()
        thumb_contact_ids = list(set(thumb_vertices_ids) & set(contact_ids))
        thumb_vertices = HandMesh.vertices[thumb_contact_ids]
        
        if not thumb_contact_ids:
            continue
        
        # import pdb; pdb.set_trace()
        query_points = thumb_vertices
        ObjMeshQuery = trimesh.proximity.ProximityQuery(ObjMesh)
        _, distances, closest_faces = ObjMeshQuery.on_surface(points=query_points)
        condition_centers = []
        condition_centers_dists = []
        condition_candidates = []
        center_list = centers.tolist()
        for fid, face in enumerate(closest_faces):
            if face in centers:
                index = center_list.index(face)
                dist = distances[fid]
                if index not in condition_centers: 
                    condition_centers.append(index)
                    condition_centers_dists.append(dist)
                # condition_centers.append(fid)
        # import pdb; pdb.set_trace()
        
        if condition_centers:
            thumb_contact_frame_names.append(dataset.frame_names_orig[idx])
            # update the ds_thumb dict with list first
            for key in list(ds_thumb.keys()):
                ds_thumb[key].append(ds_orig[key][idx])
            
        # sdf_annot['centers'] = sdf_annot['obj_faces_ids']
        if 'obj_faces_ids' in list(sdf_annot.keys()):
            sdf_annot['candidate_centers'] = sdf_annot['obj_faces_ids']
            del sdf_annot['obj_faces_ids']
        sdf_annot['thumb_center_ids'] = condition_centers
        sdf_annot['thumb_center_dists'] = condition_centers_dists
        
        np.save(filename, sdf_annot)
        
        # if idx > 0: 
        #     break
        
    frame_names_thumb = np.asarray(thumb_contact_frame_names)
    output_path = os.path.join(dataset.ds_path, 'frame_names_thumb.npz') 
    # TODO: save the frame_name array with thumb contact to ds_path
    np.savez(output_path, frame_names=frame_names_thumb)
    
    # TODO: save the grabnet ds files
    ds_thumb = {k: np.array(ds_thumb[k]) for k in list(ds_thumb.keys())}
    ds_thumb_path = os.path.join(dataset.ds_path, f'grabnet_{args.ds_name}_thumb.npz')
    np.savez(ds_thumb_path, 
             global_orient_rhand_rotmat = ds_thumb['global_orient_rhand_rotmat'], 
             fpose_rhand_rotmat = ds_thumb['fpose_rhand_rotmat'], 
             trans_rhand = ds_thumb['trans_rhand'], 
             trans_obj = ds_thumb['trans_obj'], 
             root_orient_obj_rotmat = ds_thumb['root_orient_obj_rotmat'], 
             global_orient_rhand_rotmat_f = ds_thumb['global_orient_rhand_rotmat_f'], 
             fpose_rhand_rotmat_f = ds_thumb['fpose_rhand_rotmat_f'], 
             trans_rhand_f = ds_thumb['trans_rhand_f'])
    
    frame_names = np.load(output_path)
    
    # import pdb; pdb.set_trace()
    
    frame_names_arr = frame_names['frame_names']
    
    print(f"Frames with thumb contact: {frame_names_arr.shape[0]}")
        
        # sdf_annot_file_0 = np.load(annot_frame_names[idx], allow_pickle=True)
        # sdf_annot_0 = sdf_annot_file_0.tolist()
        # # import pdb; pdb.set_trace()
        # assert 'thumb_center_ids' in list(sdf_annot.keys())
        # sdf_annot.keys()
                
        
        
        
    