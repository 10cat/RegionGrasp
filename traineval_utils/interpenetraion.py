import os
import sys
sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch
import trimesh

def intersect_vox_obj(hand_mesh, obj_mesh, pitch=0.01):
    obj_vox = obj_mesh.voxelized(pitch=pitch)
    obj_points = obj_vox.points
    inside = hand_mesh.contains(obj_points)
    volume = inside.sum() * np.power(pitch, 3)
    return volume

def intersect_vox_hand(hand_mesh, obj_mesh, pitch=0.01):
    hand_vox = hand_mesh.voxelized(pitch=pitch)
    hand_points = hand_vox.points
    inside = obj_mesh.contains(hand_points)
    volume = inside.sum() * np.power(pitch, 3)
    return volume


def get_interpenetration_volume(sample_info, mode='voxels_obj'):
    hand_mesh = trimesh.Trimesh(vertices=sample_info['hand_verts'], faces=sample_info['hand_faces'])
    obj_mesh = trimesh.Trimesh(vertices=sample_info['obj_verts'], faces=sample_info['obj_faces'])

    if mode == 'voxels_obj':
        volume = intersect_vox_obj(hand_mesh, obj_mesh)
    elif mode == 'voxels_hand':
        volume = intersect_vox_hand(hand_mesh, obj_mesh)

    return volume

if __name__ == "__main__":
    from dataset.Dataset import GrabNetDataset
    import config
    from option import MyOptions as cfg
    import mano
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--machine', type=str, default='41')

    args = parser.parse_args()

    cfg.machine = args.machine

    rh_model = mano.load(model_path=cfg.mano_rh_path,
                         model_type='mano',
                         num_pca_comps=45,
                         batch_size=cfg.batch_size,
                         flat_hand_mean=True)
    valset = GrabNetDataset(dataset_dir=config.dataset_dir, ds_name='val')
    sample = valset.__getitem__(0)
    hand_mesh = trimesh.Trimesh(vertices=np.array(sample['verts_rhand']), faces=rh_model.faces)

    hand_vox = hand_mesh.voxelized(pitch=0.01)
    # hand_vox.show()
    hand_vox.as_boxes(colors=config.colors['skin']).export('vox_hand.obj')

