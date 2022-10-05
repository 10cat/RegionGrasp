import numpy as np
import sys
sys.path.append('.')
sys.path.append('..')
import os
from mano.utils import Mesh
import trimesh
import pymeshlab as pyml
from quad_mesh_simplify import simplify_mesh
from tqdm import tqdm
from utils.utils import makepath

colors = {
    'pink': [1.00, 0.75, 0.80],
    'skin': [0.96, 0.75, 0.69],
    'purple': [0.63, 0.13, 0.94],
    'red': [1.0, 0.0, 0.0],
    'green': [.0, 1., .0],
    'yellow': [1., 1., 0],
    'brown': [1.00, 0.25, 0.25],
    'blue': [.0, .0, 1.],
    'white': [1., 1., 1.],
    'orange': [1.00, 0.65, 0.00],
    'grey': [0.75, 0.75, 0.75],
    'black': [0., 0., 0.],
}

def decimate_to_given_vertices(mesh, num_target_verts, num_add_faces):
    """
    Refer to the solution here: https://stackoverflow.com/a/65424578
    Input:
    - mesh(pymeshlab.MeshSet())
    """

    m = mesh.current_mesh()
    TARGET = num_target_verts
    numFaces = 2*TARGET + num_add_faces

    while (mesh.current_mesh().vertex_number() > TARGET):
        mesh.apply_filter('simplification_quadric_edge_collapse_decimation', targetfacenum=numFaces, preservenormal=True)
        numFaces = numFaces - (mesh.current_mesh().vertex_number() - TARGET)
    
    m = mesh.current_mesh()
    if m.vertex_number() < TARGET:
        print(f"Attention: got {m.vertex_number()}  instead of {TARGET} vertices.")

    return mesh




if __name__ == "__main__":
    root = "/home/datassd/yilin/GrabNet/tools/object_meshes/"
    obj_mesh_path = os.path.join(root, "contact_meshes")
    # output_path = os.path.join(root, "decimate_info")
    visual_path = os.path.join(root, "origin_meshes")
    output_path = os.path.join(root, "decimate_meshes")
    makepath(output_path)
    makepath(visual_path)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_verts", type=int, default=3000)
    # parser.add_argument("--even", action="store_true")
    parser.add_argument("--num_add_faces", type=int, default=100, help="add some points to the desired final num of points as tolerance to extrac rejection")
    args = parser.parse_args()

    obj_meshfile_list = os.listdir(obj_mesh_path)
    for i, file in enumerate(tqdm(obj_meshfile_list)):
        obj_name = file.split('.')[0]
        filepath = os.path.join(obj_mesh_path, file)

        # load and downsample the object model
        obj_mesh = trimesh.load(filepath)
        ObjMesh = Mesh(vertices=obj_mesh.vertices, faces=obj_mesh.faces, fc=colors['grey'])

        ObjMesh_pyml = pyml.MeshSet()
        ObjMesh_pyml.load_new_mesh(filepath)

        ObjMesh_new_ms = decimate_to_given_vertices(ObjMesh_pyml, args.num_verts, args.num_add_faces)
        # new_vertices, new_faces = simplify_mesh(positions=vertices, face=faces, num_nodes=args.num_verts)
        # ObjMesh_new = Mesh(vertices=new_vertices, faces=new_faces, fc=colors['brown'])

        output_file = obj_name + '.ply'
        output_filepath = os.path.join(output_path, output_file)
        visual_filepath = os.path.join(visual_path, output_file)

        ObjMesh.export(visual_filepath)
        ObjMesh_new_ms.save_current_mesh(output_filepath)
        # ObjMesh_new.export(output_filepath)
        
