import os
import sys
sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch
import trimesh
import config
from utils.visualization import colors_like


def visual_inter(hand_mesh, rh_fs, obj_mesh, obj_fs, output_folder, frame_name):
    hand_mesh.visual.face_colors = colors_like(config.colors['skin'])
    obj_mesh.visual.face_colors = colors_like(config.colors['grey'])
    
    hand_mesh.visual.face_colors[rh_fs] = colors_like(config.colors['red'])
    obj_mesh.visual.face_colors[obj_fs] = colors_like(config.colors['blue'])
    
    output_path_hand = os.path.join()
    
    return
    

def m2m_intersect(m1, name1, m2, name2):
    """
    m1: mesh1
    name1: name of mesh1 in the CollisionManager
    m2: mesh2
    name2: name of mesh2 in the CollisionManager
    
    """
    CollisionSys = trimesh.collision.CollisionManager()
    CollisionSys.add_object(name1, m1)
    
    
    return

