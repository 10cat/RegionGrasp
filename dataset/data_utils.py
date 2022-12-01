import os
import sys
sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch
import trimesh
import config
from utils.visualization import colors_like
import matplotlib.pyplot as plt

class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)
            
def load_mano_model(model_path, is_rhand=True, ext='pkl'):
    import os.path as osp
    import pickle
    
    # Load the model
    if osp.isdir(model_path):
        model_fn = 'MANO_{}.{ext}'.format('RIGHT' if is_rhand else 'LEFT', ext=ext)
        mano_path = os.path.join(model_path, model_fn)
    else:
        mano_path = model_path
        
    assert osp.exists(mano_path), 'Path {} does not exist!'.format(
        mano_path)

    if ext == 'pkl':
        with open(mano_path, 'rb') as mano_file:
            model_data = pickle.load(mano_file, encoding='latin1')
    elif ext == 'npz':
        model_data = np.load(mano_path, allow_pickle=True)
    else:
        raise ValueError('Unknown extension: {}'.format(ext))
    data_struct = Struct(**model_data)
    return data_struct

def visual_mesh_region(mesh, fs, color):
    mesh.visual.face_colors[fs] = colors_like(config.colors[color])

def visual_mesh(mesh, bg_color='grey', mark_region=None, mark_color='yellow'):
    mesh.visual.face_colors = colors_like(config.colors[bg_color])
    if mark_region:
        if isinstance(mark_region[0], list):
            assert isinstance(mark_color, list), "Parameter 'mark_color' must match the type of 'mark_region'!"
            for idx, region in enumerate(mark_region):
                visual_mesh_region(mesh, region, mark_color[idx])
                # mesh.visual.face_colors[region] = colors_like(config.colors[mark_color[idx]])
                
        else:
            assert isinstance(mark_color, str), "Parameter 'mark_color' must match the type of 'mark_region'!"
            visual_mesh_region(mesh, mark_region, mark_color)
            # mesh.visual.face_colors[mark_region] = colors_like(config.colors[mark_color])

def visual_inter(hand_mesh, rh_fs, h_mark_color,
                 obj_mesh, obj_fs, o_mark_color,
                 output_folder, 
                 frame_name):
    # TODO: component related visualization
    visual_mesh(hand_mesh, bg_color='skin', mark_region=rh_fs, mark_color=h_mark_color)
    visual_mesh(obj_mesh, bg_color='grey', mark_region=obj_fs, mark_color=o_mark_color)
    
    output_path_hand = os.path.join(output_folder, frame_name+'_hand.ply')
    output_path_obj = os.path.join(output_folder, frame_name+'_obj.ply')
    
    hand_mesh.export(output_path_hand)
    obj_mesh.export(output_path_obj)
    
    return

def visual_hist(array):
    plt.figure()
    plt.hist(array)
    plt.axvline(x=np.median(array), color='b', label='median value')
    plt.axvline(x=np.mean(array), color='r', label='mean value')
    plt.show()
    plt.close()
    
def visual_sort(array, plot):
    # array_sort = np.sort(array)
    array_uni = np.array(list(set(array)))
    array_sort = np.sort(array_uni)
    th = cluster_threshold(array_sort)
    if plot:
        plt.figure()
        plt.plot(array_sort)
        plt.axhline(y=th, color='r', label='threshold')
        plt.show()
        plt.close()

def cluster_threshold(sorted_array):
    threshold = None
    diff_max = 0
    for idx, value in enumerate(sorted_array):
        pre_value = sorted_array[ idx - 1 ] if idx > 0 else value
        post_value = sorted_array[ idx + 1 ] if idx < (sorted_array.shape[0] - 1) else value
        pre_diff = value - pre_value
        post_diff = post_value - value
        diff = abs( post_diff - pre_diff )
        diff_pos = (post_diff - pre_diff) > 0
        if diff > diff_max: 
            diff_max = diff
            diff_pos = diff_pos
            diff_max_idx = idx
    # import pdb; pdb.set_trace()
    if diff_pos:
        threshold = (sorted_array[diff_max_idx] + sorted_array[diff_max_idx+1]) / 2
    else:
        threshold = (sorted_array[diff_max_idx] + sorted_array[diff_max_idx-1]) / 2
        
    return threshold

    
def contact_to_dict(dict, contact, name1, name2):
    """
    Returns: ContactDict
    - 'face_index'
        - name1
        - name2
    - 'normal'
    - 'point'
    """
    
    dict['face_index'][name1].append(contact.index(name1))
    dict['face_index'][name2].append(contact.index(name2))
    dict['depth'].append(contact.depth)
    dict['normal'].append(contact.normal)
    dict['point'].append(contact.point)
    
    return 


def m2m_intersect(m1, name1, m2, name2):
    """
    m1: mesh1
    name1: name of mesh1 in the CollisionManager
    m2: mesh2
    name2: name of mesh2 in the CollisionManager
    ----
    Returns:
    ContactDict (dict):
    - 'face_index'
        - name1
        - name2
    - 'normal'
    - 'point'
    
    """
    # NOTE: CollisionManager需要调用python-fcl库：https://github.com/BerkeleyAutomation/python-fcl
    CollisionSys = trimesh.collision.CollisionManager()
    CollisionSys.add_object(name1, m1)
    CollisionSys.add_object(name2, m2)
    is_collision, names, ContactDatas_list = CollisionSys.in_collision_internal(return_names=True, return_data=True)
    # import pdb; pdb.set_trace()
    ContactDict = {'face_index':{name1:[], name2:[]}, 'depth':[], 'normal':[], 'point':[]}
    
    for idx, contact in enumerate(ContactDatas_list):
        contact_to_dict(ContactDict, contact, name1, name2)
        
    return ContactDict
    

def faces2verts_no_rep(faces):
    verts = faces.reshape(-1).tolist()
    verts_no_rep = list(set(verts))
    return verts_no_rep

def inner_verts_detect(mesh_dest, mesh_orig, vert_ids):
    verts = mesh_orig.vertices[vert_ids]
    signed_dists = trimesh.proximity.signed_distance(mesh_dest, verts)
    inner_indices = np.where(signed_dists > 0)[0].tolist()
    return inner_indices

def find_surrounded_faces(vids, mesh):
    face_ids = []
    for vid in vids:
        fids_arr = mesh.vertex_faces[vid]
        fids_arr = fids_arr[fids_arr > 0] # unpadded
        face_ids += fids_arr.tolist()
        face_ids = list(set(face_ids))
    return face_ids


if __name__ == "__main__":
    mano_path = config.mano_dir
    mano_model = load_mano_model(mano_path)
    
    rh_verts = mano_model.v_template
    rh_faces = mano_model.f
    hand_mesh = trimesh.Trimesh(vertices=rh_verts, faces=rh_faces)
    
    thumb_faces = find_surrounded_faces(config.thumb_vertices, hand_mesh)
    index_faces = find_surrounded_faces(config.indexfinger_vertices, hand_mesh)
    middle_faces = find_surrounded_faces(config.middlefinger_vertices, hand_mesh)
    fourth_faces = find_surrounded_faces(config.fourthfinger_vertices, hand_mesh)
    small_faces = find_surrounded_faces(config.smallfinger_vertices, hand_mesh)
    
    visual_mesh(hand_mesh, 
                bg_color='skin', 
                mark_region=[thumb_faces, index_faces, middle_faces, fourth_faces, small_faces], 
                mark_color=['green', 'blue', 'purple', 'yellow', 'orange'])
    
    # hand_mesh.show()
    
    hand_comp = {'thumb':[thumb_faces, config.thumb_vertices], 
                 'index': [index_faces, config.indexfinger_vertices], 
                 'middle': [middle_faces, config.middlefinger_vertices], 
                 'fourth': [fourth_faces, config.fourthfinger_vertices], 
                 'small': [small_faces, config.smallfinger_vertices]}
    
    output_path = 'dataset/tools/hand_comp.npz'
    np.savez(output_path, hand_comp, allow_pickle=True)
    
    
    
    
    

