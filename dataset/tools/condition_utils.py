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


def thumb_query_points(HandMesh, ObjMesh, pene_th=-0.002, contact_th=0.005):
    thumb_vertices_ids = faces2verts_no_rep(HandMesh.faces[config.thumb_center])
    thumb_vertices = HandMesh.vertices[thumb_vertices_ids]
    ObjQuery = trimesh.proximity.ProximityQuery(ObjMesh)
    # TODO: 以thumb_vertices_ids为query计算signed distances并返回相对应的closest faces
    h2o_signed_dists = ObjQuery.signed_distance(thumb_vertices)
    _, dists, h2o_closest_fid = ObjQuery.on_surface(thumb_vertices)
    
    import pdb; pdb.set_trace() # CHECK: if the on_surface returns signed_dists
    # TODO: 用sdf_th阈值进一步选取thumb上真正的contact部分
    
    penet_flag = dists > pene_th
    contact_flag = dists < contact_th
    flag = penet_flag & contact_flag
    obj_contact_fids = h2o_closest_fid[flag]
    