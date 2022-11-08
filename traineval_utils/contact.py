import os
import sys
sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch
import trimesh

def get_contact_area(sample_info, pene_th=0.005, contact_th=-0.005):
    """
    Parameters:
    - sample_info: (dict) includes 4 numpy array items -- 'hand_verts', 'hand_faces', 'obj_verts', 'obj_faces'
    - pene_th: penetration threshold value (positive, since trimesh.proximity.signed_distance output positive penetration depth)
    - contact_th: contact threshold value
    ---------------------
    Returns:
    area: total area of faces in contact
    """
    hand_verts = sample_info['hand_verts']
    HandMesh = trimesh.Trimesh(vertices=hand_verts, faces=sample_info['hand_faces'])
    ObjMesh = trimesh.Trimesh(vertices=sample_info['obj_verts'], faces=sample_info['obj_faces'])

    signed_dists = trimesh.proximity.signed_distance(ObjMesh, hand_verts) # map signed distance to each hand verts
    pene_th = 0.005
    contact_th = -0.005
    # indices = np.where(signed_dists < pene_th & signed_dists > contact_th).tolist() #TODO check if tolist is needed
    bools = (signed_dists < pene_th) * (signed_dists > contact_th)
    # import pdb; pdb.set_trace()
    indices = np.where(bools > 0)[0].tolist()
    v2faces = HandMesh.vertex_faces[indices] #TODO check if tolist is needed
    
    # import pdb; pdb.set_trace()
    ## TODO need to 1)select valid face index from redundant list  2)take union
    bools = v2faces > 0
    face_indices = list(set(v2faces[bools])) 
    # get the area of each faces and sum up
    contact_area = HandMesh.area_faces[face_indices]
    # import pdb; pdb.set_trace()
    contact_area = np.sum(contact_area)

    return contact_area


def get_CA_IV_ratio(contact_area, intersection_volume, eps=1e-6):
    if intersection_volume == 0:
        intersection_volume += eps
    return contact_area / intersection_volume


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
    idx = 220

    sample = valset.__getitem__(idx)

    hand_verts = np.array(sample['verts_rhand'])
    hand_faces = rh_model.faces
    HandMesh = trimesh.Trimesh(vertices=hand_verts, faces=hand_faces)

    obj_verts = np.array(sample['verts_obj'])
    obj_name = valset.frame_objs[idx]
    ObjMesh = valset.object_meshes[obj_name]
    ObjMesh.vertices = obj_verts

    sample_info = {}
    sample_info['hand_verts'] = sample['verts_rhand']
    sample_info['hand_faces'] = rh_model.faces
    sample_info['obj_verts'] = sample['verts_obj']
    sample_info['obj_faces'] = ObjMesh.faces

    # -- hand -> obj signed distance -> face area on hand mesh
    contact_area = get_contact_area(sample_info=sample_info)
    print(f"the contact_area is {contact_area}")





