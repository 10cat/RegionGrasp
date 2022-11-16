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


def visual_inter(hand_mesh, rh_fs, obj_mesh, obj_fs, output_folder, frame_name):
    hand_mesh.visual.face_colors = colors_like(config.colors['skin'])
    obj_mesh.visual.face_colors = colors_like(config.colors['grey'])
    
    hand_mesh.visual.face_colors[rh_fs] = colors_like(config.colors['red'])
    obj_mesh.visual.face_colors[obj_fs] = colors_like(config.colors['blue'])
    
    output_path_hand = os.path.join(output_folder, frame_name+'_hand.ply')
    output_path_obj = os.path.join(output_folder, frame_name+'_obj.ply')
    
    return

def visual_hist(array):
    plt.figure()
    plt.hist(array)
    plt.axvline(x=np.median(array), color='b', label='median value')
    plt.axvline(x=np.mean(array), color='r', label='mean value')
    plt.show()
    plt.close()
    
def visual_sort(array):
    array_sort = np.sort(array)
    plt.figure()
    plt.plot(array_sort)
    plt.show()
    plt.close()

def cluster_threshold(sorted_array):
    threshold = None
    diff_max = 0
    for idx, value in enumerate(sorted_array):
        pre_value = sorted_array[ idx - 1 ] if idx > 0 else value
        post_value = sorted_array[ idx + 1 ] if idx < (sorted_array.shape[0] - 1) else value
        pre_diff = value - pre_value
        post_diff = post_diff - value
        diff = abs( post_diff - pre_diff )
        diff_sign = np.sign(post_diff - pre_diff)
        if diff > diff_max: 
            diff_max = diff
            diff_sign = diff_sign
            diff_max_idx = idx
    if diff_sign > 0:
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

