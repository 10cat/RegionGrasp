import numpy as np
import sys
sys.path.append('.')
sys.path.append('..')
import os
import trimesh
from tqdm import tqdm
from utils.utils import makepath

def obj_to_ply(mesh_ply_path, mesh_obj_path):
    obj_path_list = os.listdir(mesh_obj_path)
    for i, path in tqdm(enumerate(obj_path_list)):
        obj_name = path.split('.')[0]
        filepath = os.path.join(mesh_obj_path, path)
        ObjMesh = trimesh.load(filepath)
        obj_path = os.path.join(mesh_ply_path, obj_name + '.ply')
        ObjMesh.export(obj_path)

def ply_to_obj(mesh_ply_path, mesh_obj_path):
    ply_path_list = os.listdir(mesh_ply_path)
    for i, path in tqdm(enumerate(ply_path_list)):
        obj_name = path.split('.')[0]
        filepath = os.path.join(mesh_ply_path, path)
        ObjMesh = trimesh.load(filepath)
        obj_path = os.path.join(mesh_obj_path, obj_name + '.obj') 
        ObjMesh.export(obj_path)


if __name__ == "__main__":
    root = "/home/datassd/yilin/GrabNet/tools/object_meshes/"

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ply_to_obj", action="store_true")
    parser.add_argument("--obj_to_ply", action="store_true")
    parser.add_argument("--ply_path", type=str, required=True)
    parser.add_argument("--obj_path", type=str, required=True)

    args = parser.parse_args()

    mesh_ply_path = os.path.join(root, args.ply_path)
    mesh_obj_path = os.path.join(root, args.obj_path)
    makepath(mesh_ply_path)
    makepath(mesh_obj_path)

    if args.ply_to_obj:
        ply_to_obj(mesh_ply_path, mesh_obj_path)
    
    elif args.obj_to_ply:
        obj_to_ply(mesh_ply_path, mesh_obj_path)

    

    


