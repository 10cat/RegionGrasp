import os
import sys
sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch
import trimesh
import config
from dataset.data_utils import faces2verts_no_rep, MeshTransform, MeshInitialize
from utils.visualization import visual_inter, visual_mesh, visual_mesh_region
from utils.utils import func_timer, makepath


def thumb_query_points(HandMesh, ObjMesh, sdf_th=-0.005):
    thumb_vertices_ids = faces2verts_no_rep(HandMesh.faces[config.thumb_center])
    thumb_vertices = HandMesh.vertices[thumb_vertices_ids]
    ObjQuery = trimesh.proximity.ProximityQuery(ObjMesh)
    # TODO: 以thumb_vertices_ids为query计算signed distances并返回相对应的closest faces
    h2o_signed_dists = ObjQuery.signed_distance(thumb_vertices)
    _, dists, h2o_closest_fid = ObjQuery.on_surface(thumb_vertices)
    
    import pdb; pdb.set_trace() # CHECK: if the on_surface returns signed_dists
    
    
    # TODO: 用sdf_th阈值进一步选取thumb上真正的contact部分
    flag = dists > sdf_th
    contact_num = h2o_closest_fid[flag].shape[0]
    