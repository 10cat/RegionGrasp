import os
import sys
from xml.sax.handler import DTDHandler
sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch
import trimesh
import config
from option import MyOptions as cfg
from dataset.data_utils import faces2verts_no_rep
from utils.utils import makepath

def get_thumb_contact(HandMesh, signed_dists, threshold=cfg.condition_dist):
    thumb_vertices_ids = faces2verts_no_rep(HandMesh.faces[config.thumb_center])
    contact_ids = np.where(signed_dists > threshold)[0].tolist()
    thumb_contact_ids = list(set(thumb_vertices_ids) & set(contact_ids)) # 取交集获得thumb中sdf > th的点
    thumb_vertices = HandMesh.vertices[thumb_contact_ids] if thumb_contact_ids else None
    return thumb_vertices, thumb_contact_ids

def adjacency_to_faces(ObjMesh, adj_candidates):
    face_candidates = []
    for adj_indices in adj_candidates:
        adjacency = ObjMesh.face_adjacency[adj_indices]
        # import pdb; pdb.set_trace()
        f_indices = adjacency.reshape(-1)
        f_indices = np.unique(f_indices).tolist() # 去掉重复的edge共面
        face_candidates.append(f_indices)
    return face_candidates

def get_contact_region(ObjMesh, obj_face_id, radius=config.rtree_radius):
    rtree = ObjMesh.face_adjacency_tree
    face = ObjMesh.faces[obj_face_id]
    
    points = ObjMesh.vertices[face[0]].reshape(1, -1) 
    # NOTE: points的维度必须是 (n, 3) 才能产生正确的bounds维度(n, 6); 否则报错：rtree.exceptions.RTreeError: Coordinates must be in the form (minx, miny, maxx, maxy) or (x, y) for 2D indexes
    
    bounds = np.column_stack((points - radius, points + radius))
    
    adj_candidates = [list(rtree.intersection(b)) for b in bounds]
    face_candidates = adjacency_to_faces(ObjMesh, adj_candidates)
    
    return face_candidates[0] # adjacency_to_faces输出的face_candidates是[[]]形式，需要用[0]取出
    

def main(sample_info, signed_dists=None, h_nearest_faces=None):
    hand_verts = sample_info['hand_verts']
    HandMesh = trimesh.Trimesh(vertices=hand_verts, faces=sample_info['hand_faces'])
    ObjMesh = trimesh.Trimesh(vertices=sample_info['obj_verts'], faces=sample_info['obj_faces'])
    
    if signed_dists is None:
        signed_dists = trimesh.proximity.signed_distance(ObjMesh, hand_verts) # map signed distance to each hand verts
    
    thumb_contact_verts, thumb_contact_ids = get_thumb_contact(HandMesh, signed_dists)
    
    
    if thumb_contact_verts is None:
        return 0, 0.0
    
    # import pdb; pdb.set_trace()
    center = int(sample_info['cond_center']) #需要从tensor转换回int
    region_mask = sample_info['cond_region_mask']
    
    # import pdb; pdb.set_trace()
    # DONE: transfer condition region_mask to vertices
    region_faces = np.where(region_mask > 0)[0].tolist() # -- list
    
    # import pdb; pdb.set_trace()
    # DONE: 如果输入了h_nearest_faces,则直接根据输入筛选，不需要重新计算closest faces
    if h_nearest_faces is not None:
        closest_faces = h_nearest_faces[thumb_contact_ids]
    else:
        # DONE: find the closest faces on object -- based on signed distance
        query_points = thumb_contact_verts
        ObjMeshQuery = trimesh.proximity.ProximityQuery(ObjMesh)
        _, distances, closest_faces = ObjMeshQuery.on_surface(points=query_points)
    
    # DONE: 如果closest_faces中有condition region center:
    # import pdb; pdb.set_trace()
    # CHECK: -- coverage = 100%
    closest_faces = closest_faces.tolist() # closest_faces需要tolist
    if center in closest_faces:
        coverage = 1
    # DONE: 如果closest_faces中没有condition region center:
    else:
        # import pdb; pdb.set_trace()
        # DONE: -- find among those the closest one to the condition region center -- based on triangle center distance 
        # triangles = ObjMesh.triangles[closest_faces] 
        tri_centers = np.array([ObjMesh.triangles_center[fid] for fid in closest_faces])
        c_triangle_center = ObjMesh.triangles_center[center]
        dists = np.linalg.norm((tri_centers - c_triangle_center), axis=1)
        # import pdb; pdb.set_trace()
        dists_min_id = int(np.argmin(dists))
        pred_center = closest_faces[dists_min_id]
        # import pdb; pdb.set_trace()
        # DONE: -- obtain the neighborhood by certain sphere radius as generated thumb contact region -- based on Rtree
        pred_region_faces = get_contact_region(ObjMesh, pred_center)
        # import pdb; pdb.set_trace()
        # DONE: -- calculate the coverage (obtained region & condition region)
        cover_faces = list(set(pred_region_faces) & set(region_faces))
        coverage = len(cover_faces) / len(region_faces)
        
    hit_success = 1 if coverage > cfg.coverage_th else 0
    
    return hit_success, coverage