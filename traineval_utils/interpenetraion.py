import os
import sys
sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch
import trimesh

def intersect_vox(hand_mesh, obj_mesh,pitch=0.01):
    obj_vox = obj_mesh.voxelized(pitch=pitch)
    obj_points = obj_vox.points
    inside = hand_mesh.contains(obj_points)
    volume = inside.sum() * np.power(pitch, 3)
    return volume


def get_interpenetration_volume(sample_info, mode='voxels'):
    hand_mesh = trimesh.Trimesh(vertices=sample_info['hand_verts'], faces=sample_info['hand_faces'])
    obj_mesh = trimesh.Trimesh(vertices=sample_info['obj_verts'], faces=sample_info['obj_faces'])

    if mode == 'voxels':
        volume = intersect_vox(hand_mesh, obj_mesh)

    return volume

if __name__ == "__main__":
    from dataset.Dataset import GrabNetDataset
    import config