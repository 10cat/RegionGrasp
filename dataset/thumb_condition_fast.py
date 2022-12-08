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
from dataset.data_utils import m2m_intersect, visual_hist, visual_inter, visual_mesh, visual_sort, faces2verts_no_rep, inner_verts_detect, visual_mesh_region
from utils.utils import func_timer, makepath
import matplotlib.pyplot as plt
from tqdm import tqdm
                
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds_name', type=str, required=True)
    parser.add_argument('--visual_freq', type=int, default=500)
    parser.add_argument('--plot', action='store_true', default=False)
    args = parser.parse_args()
    
    dataset_root = config.DATASET_ROOT
    output_root = config.dataset_visual_dir
    mano_path = config.mano_dir
    rh_model = load(model_path = mano_path,
                             is_rhand = True,
                             num_pca_comps=45,
                             flat_hand_mean=True)
    
    dataset = GrabNetDataset_orig(dataset_root, ds_name=args.ds_name, obj_meshes_folder='decimate_meshes', dtype=np.float32)
    visual_folder = os.path.join(config.dataset_visual_dir, 'thumb_condition', 'candidate')
    makepath(visual_folder)
    
    annot_frame_names = [os.path.join(dataset.ds_root, fname.replace('data', 'data_sdf').replace('npz', 'npy')) for fname in dataset.frame_names_orig]
    
    thumb_contact_frame_names = []
    
    for idx in tqdm(range(dataset.__len__()), desc=f'{args.ds_name}'):
        
        sample = dataset.__getitem__(idx)
        hand_faces = rh_model.faces
        hand_verts = sample['verts_rhand']
        HandMesh = trimesh.Trimesh(vertices=hand_verts, faces=hand_faces)
        
        obj_name = dataset.frame_objs[idx]
        # import pdb; pdb.set_trace()
        #NOTE:下采样到2048点的物体顶点不能直接使用，需要从原模型获得全部顶点坐标
        ObjMesh = dataset.object_meshes[obj_name]
        ## 先从原物体模型获得原物体顶点
        obj_verts_orig = ObjMesh.vertices 
        obj_trans = sample['trans_obj']
        obj_rotmat = sample['root_orient_obj_rotmat'][0]
        ## 通过旋转、平移转换的矩阵操作获得实际顶点的坐标
        obj_verts = np.matmul(obj_verts_orig, obj_rotmat) + obj_trans
        obj_faces = ObjMesh.faces
        
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
        
        # TODO: query points不应该是object的顶点，而应该是thumb顶点
        # thumb_vertices = config.hand_comp['thumb'][1]
        thumb_vertices_ids = faces2verts_no_rep(HandMesh.faces[config.thumb_center]) # TODO: to ensure the selected region in very close to the thumb contact region
        thumb_vertices = HandMesh.vertices[thumb_vertices_ids]
        # import pdb; pdb.set_trace()
        query_points = thumb_vertices
        ObjMeshQuery = trimesh.proximity.ProximityQuery(ObjMesh)
        _, _, closest_faces = ObjMeshQuery.on_surface(points=query_points)
        condition_centers = []
        condition_candidates = []
        center_list = centers.tolist()
        for fid, face in enumerate(closest_faces):
            if face in centers:
                index = center_list.index(face)
                if index not in condition_centers: 
                    condition_centers.append(index)
                # condition_centers.append(fid)
        # import pdb; pdb.set_trace()
        
        if condition_centers:
            thumb_contact_frame_names.append(dataset.frame_names_orig[idx])
        # sdf_annot['centers'] = sdf_annot['obj_faces_ids']
        if 'obj_faces_ids' in list(sdf_annot.keys()):
            sdf_annot['candidate_centers'] = sdf_annot['obj_faces_ids']
            del sdf_annot['obj_faces_ids']
        sdf_annot['thumb_center_ids'] = condition_centers
        
        np.save(filename, sdf_annot)
        
        # if idx > 0: 
        #     break
        
    frame_names_thumb = np.asarray(thumb_contact_frame_names)
    output_path = os.path.join(dataset.ds_path, 'frame_names_thumb.npz') 
    # TODO: save the frame_name array with thumb contact to ds_path
    np.savez(output_path, frame_names=frame_names_thumb)
    
    frame_names = np.load(output_path)
    
    # import pdb; pdb.set_trace()
    
    frame_names_arr = frame_names['frame_names']
    
    print(f"Frames with thumb contact: {frame_names_arr.shape[0]}")
        
        # sdf_annot_file_0 = np.load(annot_frame_names[idx], allow_pickle=True)
        # sdf_annot_0 = sdf_annot_file_0.tolist()
        # # import pdb; pdb.set_trace()
        # assert 'thumb_center_ids' in list(sdf_annot.keys())
        # sdf_annot.keys()
                
        
        
        
    