#%%
import sys
sys.path.append("..")
import os
import string
import config
import torch
from torch.utils import data
import numpy as np
from data_utils import mode_path
import utils
from utils.io_utils import load_npy_file, load_npz_file, load_pt_file
from utils.utils import to_cpu


#TODO: construct GRAB dataset class
class GRAB(data.Dataset):
    def __init__(self, root, mode):
        #super().__init__()
        self.root = root
        self.mode = mode
        self.folder_root = mode_path(dataset="GRAB", root=self.root, mode=self.mode)
        #TODO: load the preprocessed data from .npz file and .pt files

        self.frame_names_array = load_npz_file(os.path.join(self.folder_root, "frame_names.npz"), one_key="frame_names")
        
        object_info = load_npy_file(os.path.join(self.root, "obj_info.npy"))
        self.object_info = object_info.item()
        self.rhand_tensor = load_pt_file(os.path.join(self.folder_root, "rhand_data.pt"))
        self.lhand_tensor = load_pt_file(os.path.join(self.folder_root, "lhand_data.pt"))
        self.object_tensor = load_pt_file(os.path.join(self.folder_root, "object_data.pt"))
    
    def get_ids_names(self, path, sample):
        name_list = path.split('/')
        name_seq = name_list[-1]
        sample["path"] = path
        sample["subject_id"] = name_list[-2]
        sample["gender"] = config.gender_map[sample["subject_id"]]
        interact_seq = name_seq.split('_')
        sample["object_name"] = interact_seq[0]
        sample["action_name"] = interact_seq[1]
        if len(interact_seq) > 3:
            sample["index_1"] = interact_seq[2]
            sample["index_2"] = interact_seq[3]
        else:
            sample["index_1"] = 0
            sample["index_2"] = eval(interact_seq[-1])
        return sample
    
    def get_hand_data(self, tensor, idx):
        verts = tensor["verts"][idx]
        global_orient = tensor["global_orient"][idx]
        hand_pose = tensor["hand_pose"][idx]
        fullpose = tensor["fullpose"][idx]
        return verts, global_orient, hand_pose, fullpose
        

    def __len__(self):
        return self.frame_names_array.shape[0]

    def __getitem__(self, idx):
        # input: idx
        # output: sample = {keys}; keys include: -- hand vertices
        sample = {}
        sample = self.get_ids_names(self.frame_names_array[idx], sample)
        print(sample)
        sample["rhand_verts"], sample["rhand_global_orient"], sample["rhand_hand_pose"], sample["rhand_fullpose"] = self.get_hand_data(self.rhand_tensor, idx)
        print(sample)
        sample["lhand_verts"], sample["lhand_global_orient"], sample["lhand_hand_pose"], sample["lhand_fullpose"] = self.get_hand_data(self.lhand_tensor, idx)
        sample["obj_verts"] = self.object_tensor["verts"][idx] # torch
        sample["obj_info"] = self.object_info[sample["object_name"]] # np
        sample["obj_global_orient"] = self.object_tensor["global_orient"][idx] # torch
        sample["obj_transl"] = self.object_tensor["transl"][idx] # torch
        sample["contact"] = self.object_tensor["contact"][idx] # torch
        print(sample)
        return sample



if __name__ == '__main__':
    import smplx
    import trimesh
    import cv2

    root = config.dataset_root
    model_path = config.model_root
    obj_model_path = os.path.join(root, "object_meshes/contact_meshes")

    mode = "train"

    trainset = GRAB(root=root, mode=mode)
    print(f"The {mode} set length: {trainset.__len__()}")
    # visualize one frame
    idx = 0
    sample = trainset.__getitem__(idx)
    from utils.meshviewer import Mesh, MeshViewer, points2sphere, colors
    from utils.objectmodel import ObjectModel
    # rhand_m = smplx.create(model_path=model_path,
    #                         model_type='mano',
    #                         gender=)

    #TODO: visualize hand mesh
    n_comps = 24
    rh_mesh = os.path.join(root, 'subject_meshes', sample['gender'], sample["subject_id"]+'_rhand.ply')
    rh_vtemp = np.array(Mesh(filename=rh_mesh).vertices)

    rh_m = smplx.create(model_path=model_path,
                        model_type='mano',
                        is_rhand=True,
                        v_template=rh_vtemp,
                        num_pca_comps=n_comps,
                        flat_hand_mean=True,
                        batch_size=1)
    
    rh_verts_np = to_cpu(sample["rhand_verts"])
    
    rhand_mesh = Mesh(vertices=sample["rhand_verts"], faces=rh_m.faces, vc=colors['pink'], smooth=True)
    rhand_mesh.set_vertex_colors(vc=colors['red'])
    #rhand_mesh.show()
    #trimesh.exchange.ply.export_ply(rhand_mesh)


    #TODO: visualize object mesh
    print("model faces: ", sample["obj_info"]["faces"])
    # 有bug：现在downsample的结果没法直接可视化
    #TODO 直接读取object model进行contact标注
    obj_mesh_path = os.path.join(obj_model_path, sample["object_name"]+'.ply')
    obj_mesh = trimesh.load(obj_mesh_path)
    obj_verts = np.matmul(obj_mesh.vertices, cv2.Rodrigues(sample["obj_global_orient"])[0].T) + sample["obj_transl"]
    # obj_verts_np = to_cpu(sample["obj_verts"])
    # contact_np = to_cpu(sample["contact"])

    object_mesh = Mesh(vertices=obj_verts, faces=sample["obj_info"]["faces"], vc=colors['yellow'])
    # object_mesh.set_vertex_colors(vc=colors['red'], vertex_ids=contact_np > 0)
    # object_mesh.show()



    



    
    


# %%

