from fileinput import filename
import numpy as np
import sys
sys.path.append('.')
sys.path.append('..')
import os
from mano.utils import Mesh
import trimesh
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

if __name__ == "__main__":
    root = "/home/datassd/yilin/GrabNet/tools/object_meshes/"
    obj_mesh_path = os.path.join(root, "contact_meshes")
    output_path = os.path.join(root, "sample_info")
    visual_path = os.path.join(root, "sample_visual")
    makepath(output_path)
    makepath(visual_path)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_points", type=int, default=3000)
    parser.add_argument("--even", action="store_true")
    parser.add_argument("--add_points", type=int, default=500, help="add some points to the desired final num of points as tolerance to extrac rejection")
    args = parser.parse_args()



    obj_meshfile_list = os.listdir(obj_mesh_path)
    for i, file in enumerate(tqdm(obj_meshfile_list)):
        sample_info = {}
        obj_name = file.split('.')[0]
        filepath = os.path.join(obj_mesh_path, file)

        # load and downsample the object model
        obj_mesh = trimesh.load(filepath)
        ObjMesh = Mesh(vertices=obj_mesh.vertices, faces=obj_mesh.faces, fc=colors['grey'])
        if args.even:
            sample_points, face_indices = trimesh.sample.sample_surface_even(mesh=obj_mesh, count=(args.num_points + args.add_points))
        else:
            sample_points, face_indices = trimesh.sample.sample_surface(mesh=obj_mesh, count=args.num_points)
        indices = np.arange(0, sample_points.shape[0])
        sample_points = np.random.choice(indices, size=args.num_points)
        sample_info['points'] = sample_points
        sample_info['faces'] = face_indices

        # Visualization of the sampled object model
        ObjMesh.set_face_colors(fc=colors['yellow'], face_ids=face_indices)
        visual_file = obj_name + '.ply'
        visual_filepath = os.path.join(visual_path, visual_file)
        ObjMesh.export(visual_filepath)
        
        # Output the sample_info dictionary to the .npy file
        output_file = obj_name + '.npy'
        output_filepath = os.path.join(output_path, output_file)
        np.save(output_filepath, sample_info)
