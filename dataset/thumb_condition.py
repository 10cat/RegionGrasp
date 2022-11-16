import os
import sys
sys.path.append('.')
sys.path.append('.')
import config
import numpy as np
import torch
import trimesh
import mano
from mano.model import load
from dataset.Dataset_origin import GrabNetDataset_orig
from dataset.data_utils import m2m_intersect
import matplotlib.pyplot as plt



class ThumbConditionator():
    def __init__(self):
        super().__init__()
        self.thumb_verts = config.thumb_vertices
        self.Contactor = ContactDetector()
        
    def contact_annot(self, hand_mesh, obj_mesh):
        self.Contactor.run(hand_mesh, obj_mesh)
        
        return
    
    def center_point(self):
        return
    
    def condition_region(self):
        return
    
    def save_annotate(self):
        return
    
    def run(self, hand_verts, hand_faces, obj_verts, obj_faces):
        HandMesh = trimesh.Trimesh(vertices=hand_verts, faces=hand_faces)
        ObjMesh = trimesh.Trimesh(vertices=obj_verts, faces=obj_faces)
        
        # TODO: I.Contact Region Determination
        self.contact_annot(HandMesh, ObjMesh)
        
        # TODO: II. Select a 'center' of the contact region
        self.center_point()
        
        # TODO: III. Select the 'smallest' 'neighborhood' as the condition region
        self.condition_region()
        
        # TODO: IV. 转换成可以存储的标注形式 + 存储标注
        self.save_annotate()
        
        
        return
    
class ContactDetector():
    def __init__(self):
        super().__init__()
        
    def intersection_detect(self, hand_mesh, obj_mesh):
        ContactDict = m2m_intersect(hand_mesh, 'hand', obj_mesh, 'obj')
        
        return ContactDict
    
    def cluster_rings(self, ContactDict):
        depth_array = np.array(ContactDict['depth'])
        import pdb;pdb.set_trace()
        
        
        return
        
    def run(self, hand_mesh, obj_mesh):
        
        # NOTE: I.-1.1 Mesh intersection detection on thumb region
        contact_dict = self.intersection_detect(hand_mesh, obj_mesh)
        # NOTE: I.-1.2 Clustering intersection points into intersection rings
        self.cluster_rings(contact_dict)
        
        # TODO: I.-1.3 Hand Mesh Segmentation based on rings
        
        # TODO: I.-1.4 Select the outer surface ring; all the surrounded -> contact faces
        
        
        # TODO: I.-2 If not intersecting with object, threshold signed distance -> contact faces
        
        
        return

if __name__ == "__main__":
    dataset_root = config.DATASET_ROOT
    output_root = config.dataset_visual_dir
    mano_path = config.mano_dir
    
    rh_model = load(model_path = mano_path,
                             is_rhand = True,
                             num_pca_comps=45,
                             flat_hand_mean=True)
    
    trainset = GrabNetDataset_orig(dataset_root, ds_name='train', dtype=np.float32)
    
    Conditionator = ThumbConditionator()
    
    for idx in range(trainset.__len__()):
        sample = trainset.__getitem__(idx)
        hand_faces = rh_model.faces
        hand_verts = sample['verts_rhand']
        
        #NOTE:下采样到2048点的物体顶点不能直接使用，需要从原模型获得全部顶点坐标
        obj_name = trainset.frame_objs[idx]
        # import pdb; pdb.set_trace()
        ObjMesh = trainset.object_meshes[obj_name]
        ## 先从原物体模型获得原物体顶点
        obj_verts_orig = ObjMesh.vertices 
        obj_trans = sample['trans_obj']
        obj_rotmat = sample['root_orient_obj_rotmat'][0]
        ## 通过旋转、平移转换的矩阵操作获得实际顶点的坐标
        obj_verts = np.matmul(obj_verts_orig, obj_rotmat) + obj_trans
        obj_faces = ObjMesh.faces
        
        Conditionator.run(hand_verts, hand_faces, obj_verts, obj_faces)
        
        
        
        
    