# from operator import mod
import os
import sys
sys.path.append('.')
sys.path.append('..')
import numpy as np
import trimesh
from trimesh import viewer
from mano.utils import Mesh
from utils.utils import func_timer, makepath

colors = {
    'pink': [1.00, 0.75, 0.80, 1],
    'skin': [0.96, 0.75, 0.69, 1],
    'purple': [0.63, 0.13, 0.94, 1],
    'red': [1.0, 0.0, 0.0, 1],
    'green': [.0, 1., .0, 1],
    'yellow': [1., 1., 0, 1],
    'brown': [1.00, 0.25, 0.25, 1],
    'blue': [.0, .0, 1., 1],
    'white': [1., 1., 1., 1],
    'orange': [1.00, 0.65, 0.00, 1],
    'grey': [0.75, 0.75, 0.75, 1],
    'black': [0., 0., 0., 1],
}

def colors_like(color):
    color = np.array(color)

    if color.max() <= 1.:
        color = color * 255
    color = color.astype(np.int8)
    return color


def visual_sdf(Mesh, sdf, base_color = 'skin',vert_indices=None, check=False):
    """ 
    Visualize the hand mesh in the heatmap form interpolated based on hand_obj_sdf.
    If the input hand_vert_indices are not None, the vertices in the complementary set are set to grey color
    """
    vert_colors = trimesh.visual.interpolate(sdf, 'plasma')
    if vert_indices is not None:
        total = range(sdf.shape[0])
        grey_indices = list(set(total) - set(vert_indices))
        vert_colors[grey_indices] = colors_like(colors[base_color])
    Mesh.visual.vertex_colors = vert_colors
    if check:
        Mesh.show()

def visual_obj_contact_regions(Mesh, obj_face_ids, candidates, all=False, check=True):
    """
    Parameters:
    - Mesh:(trimesh.Trimesh object) Object mesh.
    - candidates: (list)
    - all: (bool) if set to be true, visualize the whole region with numpy array operation
    - check: (bool) if check is set to be true, then Mesh.show()
    ---------------------------
    Returns:
    -
    """
    Mesh.visual.face_colors = colors_like(colors['grey'])
    # import pdb; pdb.set_trace()
    color_id = ['blue', 'green', 'purple']
    for idx, R in enumerate(candidates):
        id = idx % 3
        Mesh.visual.face_colors[R] = colors_like(colors['blue'])
        # center = obj_face_ids[idx]
        # Mesh.visual.face_colors[center:center+1]= colors_like(colors['yellow'])
        # import pdb; pdb.set_trace()
        if not all and check:
            Mesh.show()

    Mesh.visual.face_colors[obj_face_ids] = colors_like(colors['yellow'])
    
    if all and check:
        Mesh.show()

def visual_obj(Mesh, region_centers=None, region_faces_ids=None):
    Mesh.visual.face_colors = colors_like(colors['grey'])
    if region_faces_ids is not None: Mesh.visual.face_colors[region_faces_ids] = colors_like(colors['blue'])
    if region_centers is not None: Mesh.visual.face_colors[region_centers] = colors_like(colors['yellow'])
    return

def visual_hand(Mesh):
    Mesh.visual.face_colors = colors_like(colors['skin'])
    return