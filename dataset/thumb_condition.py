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
from dataset.data_utils import m2m_intersect, visual_hist, visual_inter, visual_mesh, visual_sort, faces2verts_no_rep, inner_verts_detect, visual_mesh_region
from utils.utils import func_timer, makepath
import matplotlib.pyplot as plt
from tqdm import tqdm

def create_annot_folders(ds):
    ds.annotate_path = os.path.join(ds.ds_path, 'data_annot')
    ds.data_path = os.path.join(ds.ds_path, 'data')
    makepath(ds.annotate_path)
    for i in range(10):
        # create sub folders that indicate the subject names
        # if self.ds_name != 'train' and i == 8:
        #     continue
        # if i == 8:
            # import pdb; pdb.set_trace()
        name_subfolder = 's' + str(i+1)
        annotate_sub_path = os.path.join(ds.annotate_path, name_subfolder)
        
        makepath(annotate_sub_path)

        # create sub-sub folders that indicate the obj names and action names
        data_sub_path = os.path.join(ds.data_path, name_subfolder)
        for dir in os.listdir(data_sub_path):
            makepath(os.path.join(annotate_sub_path, dir))
                
class ThumbConditionator():
    def __init__(self, visual_folder, annot_fnames, plot=False, visual_freq=500):
        super().__init__()
        # self.thumb_verts = config.thumb_vertices
        self.visual_folder = visual_folder
        self.annot_fnames = annot_fnames
        self.plot = plot
        self.visual_freq = visual_freq
        self.Contactor = ContactDetector(visual_folder=visual_folder, plot=plot, visual_freq=visual_freq)
        
    def contact_annot(self, hand_mesh, obj_mesh, frame_name, idx):
        obj_contact_face_ids = self.Contactor.run(hand_mesh, obj_mesh, frame_name=frame_name, idx=idx)
        
        return obj_contact_face_ids
    
    @func_timer
    def thumb_condition_region(self, obj_mesh, face_ids_dict, frame_name, idx, region_depth=3):
        # 'center'-> contact region detect返回的obj_face_ids列表最后一个
        # NOTE: contact region determination采用BFS向内查询添加面，所以返回的obj_face_ids列表最后的一个面一定是区域中心
        thumb_obj_faces = face_ids_dict['thumb']
        center_face_id = thumb_obj_faces[-1]
        center_face = obj_mesh.faces[center_face_id]
        vertex_ids = center_face.reshape(-1).tolist()
        face_ids = [center_face_id]
        depth = 0
        vids_search = vertex_ids
        while(depth < region_depth):
            new_face_ids = []
            for vid in vids_search:
                fids = obj_mesh.vertex_faces[vid]
                fids = fids[fids > 0].tolist()
                # import pdb; pdb.set_trace()
                new_face_ids = list(set(new_face_ids + fids) - set(face_ids))
            face_ids += new_face_ids
            new_vert_ids = faces2verts_no_rep(obj_mesh.faces[new_face_ids])
            new_vert_ids = list(set(new_vert_ids) - set(vertex_ids))
            vertex_ids += new_vert_ids
            vids_search = new_vert_ids
            depth += 1
        
        thumb_condition = {
            'center': center_face_id,
            'faces': face_ids,
            'verts': vertex_ids
        }
        # import pdb; pdb.set_trace()
        # test visualization result
        if idx % 500 == 0:
            visual_mesh_region(obj_mesh, thumb_condition['faces'], 'pink')
            visual_mesh_region(obj_mesh, thumb_condition['center'], 'yellow')
            output_path = os.path.join(self.visual_folder, frame_name + '_filled_obj_cond.ply')
            obj_mesh.export(output_path)
        # import pdb; pdb.set_trace()
        
        return thumb_condition
    
    @func_timer
    def save_annotate(self, contact_faces_dict, thumb_condition_dict, idx):
        save_dict = {
            'contact': contact_faces_dict,
            'thumb_cond': thumb_condition_dict
        }
        annot_fname = self.annot_fnames[idx]
        np.save(annot_fname, save_dict, allow_pickle=True)
        
        return
    
    def run(self, hand_verts, hand_faces, obj_verts, obj_faces, frame_name, idx):
        HandMesh = trimesh.Trimesh(vertices=hand_verts, faces=hand_faces)
        ObjMesh = trimesh.Trimesh(vertices=obj_verts, faces=obj_faces)
        
        # I.Contact Region Determination
        o_contact_face_ids_dict = self.contact_annot(HandMesh, ObjMesh, frame_name, idx)
        
        thumb_condition_dict = self.thumb_condition_region(ObjMesh, o_contact_face_ids_dict, frame_name, idx)
        
        # II. Select one 'center' of the contact region
        # self.center_point()
        
        # III. Select the 'smallest' 'neighborhood' as the condition region
        # self.condition_region()
        
        # IV. 转换成可以存储的标注形式 + 存储标注
        self.save_annotate(o_contact_face_ids_dict, thumb_condition_dict, idx)
        
        
        return
    
class ContactDetector():
    def __init__(self, visual_folder, plot, visual_freq):
        super().__init__()
        self.visual_folder = visual_folder
        self.plot = plot
        self.visual_freq = visual_freq
        self.hand_comps_name = ['thumb', 'index', 'middle', 'fourth', 'small', 'palm']
        
    def intersection_detect(self, hand_mesh, obj_mesh):
        ContactDict = m2m_intersect(hand_mesh, 'hand', obj_mesh, 'obj')
        
        return ContactDict
    
    def cluster_rings(self, ContactDict, ObjMesh):
        # depth_array = np.array(ContactDict['depth'])
        ref_faces = ObjMesh.faces
        face_ids = ContactDict['face_index']['obj']
        face_ids = list(set(face_ids))
        faces = ref_faces[face_ids]
        
        import pdb;pdb.set_trace()
        
        faces_list = []
        st_verts_dict = {}
        faces_sort_dict = {}
        st = 0
        end = 0
        middle = 0
        for i in range(faces.shape[0]):
            face = faces[i]
            st_vert = face[0]
            end_vert = face[-1]
            if st_verts_dict is None:
                st_verts_dict[st_vert] = []
                st_verts_dict[st_vert].append(face.reshape(1, -1))
            else:
                st_vert_list = list(st_verts_dict.keys())
                if st_vert not in st_vert_list:
                    st_verts_dict[st_vert] = []
                st_verts_dict[st_vert].append(face.reshape(1, -1))
                
        st_vert_list = sorted(list(st_verts_dict.keys()))
        # form sorted face numpy array
            
        # visual_sort(depth_array, plot=self.plot)
        
        return


    def group_by_hand_comp(self, ContactDict):
        h_face_ids = ContactDict['face_index']['hand']
        o_face_ids = ContactDict['face_index']['obj']
        h_comp_face_ids = {'thumb':[], 'index':[], 'middle':[], 'fourth':[], 'small':[], 'palm':[]}
        o_comp_face_ids = {'thumb':[], 'index':[], 'middle':[], 'fourth':[], 'small':[], 'palm':[]}
        
        comps = ['thumb', 'index', 'middle', 'fourth', 'small']
        for idx, rh_fid in enumerate(h_face_ids):
            for comp in comps:
                count = 0
                if rh_fid in config.hand_comp[comp][0]:
                    count += 1
                    obj_fid = o_face_ids[idx]
                    if obj_fid not in o_comp_face_ids[comp]:
                        o_comp_face_ids[comp].append(obj_fid)
                        h_comp_face_ids[comp].append(rh_fid)
                    break
            if count == 0:
                if o_face_ids[idx] not in o_comp_face_ids['palm']:
                    o_comp_face_ids['palm'].append(obj_fid)
                    h_comp_face_ids['palm'].append(rh_fid)
        ContactDict['face_index']['hand'] = h_comp_face_ids
        ContactDict['face_index']['obj'] = o_comp_face_ids             
            
    @func_timer
    def fill_interior(self, hand_mesh, obj_mesh, face_ids_dict):
        """without cluster"""
        
        # 1. find the inner contour of the face by signed distance
        inner_vert_ids = []
        inner_vert_comps = []
        face_ids = [] # NOTE: needed！the total face list used for BFS
        for idx, comp in enumerate(self.hand_comps_name):
            # first get the non-repetitive faces & inner_vertices for each part
            # import pdb; pdb.set_trace()
            comp_face_ids = face_ids_dict[comp]
            if comp_face_ids:
                obj_faces = obj_mesh.faces[comp_face_ids]
                vert_ids = faces2verts_no_rep(obj_faces)
                inner_indices = inner_verts_detect(hand_mesh, obj_mesh, vert_ids)
                comp_inner_vert_ids = [vert_ids[id] for id in inner_indices] # NOTE: vert_ids is list, cannot be directly referenced with list
                # update vert list and face list
                inner_vert_ids += comp_inner_vert_ids
                face_ids += comp_face_ids
                # obtain comp marks for inner_vert
                inner_vert_comps += [idx] * len(comp_inner_vert_ids)
                # face_comps += [idx] * len(comp_face_ids)
            
        # 2. BFS search for interior faces;
        ## 更新对象： face_ids (list)
        verts_in_search = inner_vert_ids
        verts_comps = inner_vert_comps
        iter_num = 1
        # print("Start BFS search!")
        while(verts_in_search):
            new_vert_ids = [] # need a vert list
            new_vert_comps = []
            # for idx, vid in enumerate(tqdm(verts_in_search, desc=f'depth: {iter_num}')):
            for idx, vid in enumerate(verts_in_search):
                v_fids = obj_mesh.vertex_faces[vid] # returns: (m,) m = max number of faces for a single vertex in the mesh, padded with -1 
                v_fids = v_fids[v_fids > 0] # unpadded
                for fid in v_fids:
                    if fid not in face_ids:
                        face = obj_mesh.faces[fid]
                        vids = face.reshape(-1).tolist()
                        inners = inner_verts_detect(hand_mesh, obj_mesh, vids)
                        if len(inners) < 3: # 必须要3个点都被判定为在contour内部才能添加
                            continue
                        
                        # update comp list based on the comp of the current vert
                        comp = verts_comps[idx]
                        # update vert list
                        for v in vids:
                            if v not in vert_ids:
                                vert_ids.append(v) # NOTE: must update vert_ids, or there will be repetitive search!
                                new_vert_ids.append(v)
                                new_vert_comps.append(comp)
                        # update face_ids_dict directly
                        face_ids.append(fid)
                        face_ids_dict[self.hand_comps_name[comp]].append(fid)
                        
                        
            # import pdb; pdb.set_trace()
            if new_vert_ids:
                verts_in_search = new_vert_ids
                verts_comps = new_vert_comps
            else:
                verts_in_search = [] 
            
            iter_num += 1
            
        return face_ids_dict
    
    def run(self, hand_mesh, obj_mesh, frame_name, idx):
        # NOTE: I.-1.1 Mesh intersection detection on thumb region
        contact_dict = self.intersection_detect(hand_mesh, obj_mesh)
        
        
        visual_inter(hand_mesh, contact_dict['face_index']['hand'],'red',
                     obj_mesh, contact_dict['face_index']['obj'], 'blue', 
                     output_folder=self.visual_folder, 
                     frame_name=frame_name)
        
        # divide contact regions based on hand component
        self.group_by_hand_comp(contact_dict)
        
        # TODO: combine the obj_face_ids for BFS convenience, but with comp marks
        o_face_ids_dict = contact_dict['face_index']['obj']
            
        o_contact_face_ids_dict = self.fill_interior(hand_mesh, obj_mesh, o_face_ids_dict)
        
        
        # obj_contact_face_ids = {}
        # for comp in self.hand_comps_name:
        #     obj_face_ids = contact_dict['face_index']['obj'][comp]
        #     if obj_face_ids:
        #         # input 'face_ids' from obj that correspond to different components one by one
        #         # too slow!!! need to still do the full BFS for fast speed => full BFS but with comp mark?
        #         obj_comp_face_ids = self.fill_interior(hand_mesh, obj_mesh, obj_face_ids)
        #         obj_contact_face_ids[comp] = obj_comp_face_ids
        #     else:
        #         obj_contact_face_ids[comp] = []
        
        # NOTE: visualize
        if idx % self.visual_freq == 0:
            mark_colors = config.hand_comp_colors   
            h_face_ids = list(contact_dict['face_index']['hand'].values())
            obj_contact_face_ids = list(o_contact_face_ids_dict.values())
            visual_inter(hand_mesh, h_face_ids, mark_colors,
                        obj_mesh, obj_contact_face_ids, mark_colors, 
                        output_folder=self.visual_folder, 
                        frame_name=frame_name+'_filled')
            # import pdb; pdb.set_trace()
        
        # NOTE: I.-1.2 Clustering intersection points into intersection rings
        # self.cluster_rings(contact_dict, obj_mesh)
        
        
        # TODO: I.-1.3 Hand Mesh Segmentation based on rings
        
        # TODO: I.-1.4 Select the outer surface ring; all the surrounded -> contact faces
        
        
        # TODO: I.-2 If not intersecting with object, threshold signed distance -> contact faces
        
        
        return o_contact_face_ids_dict

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds_name', type=str, required=True)
    parser.add_argument('--visual_freq', type=int, default=500)
    parser.add_argument('--plot', action='store_true', default=False)
    args = parser.parse_args()
    
    dataset_root = config.DATASET_ROOT
    output_root = config.dataset_visual_dir
    mano_path = config.mano_dir
    rh_model = load(model_path = mano_path,
                             is_rhand = True,
                             num_pca_comps=45,
                             flat_hand_mean=True)
    
    dataset = GrabNetDataset_orig(dataset_root, ds_name=args.ds_name, dtype=np.float32)
    
    visual_folder = os.path.join(config.dataset_visual_dir, 'thumb_condition', 'intersect')
    makepath(visual_folder)
    
    create_annot_folders(dataset)
    annot_frame_names = [os.path.join(dataset.ds_root, fname.replace('data', 'data_annot').replace('npz', 'npy')) for fname in dataset.frame_names_orig]
    
    Conditionator = ThumbConditionator(visual_folder=visual_folder, annot_fnames=annot_frame_names, plot=args.plot, visual_freq=args.visual_freq)
    
    for idx in tqdm(range(dataset.__len__()), desc=f'{args.ds_name}'):
        
        sample = dataset.__getitem__(idx)
        hand_faces = rh_model.faces
        hand_verts = sample['verts_rhand']
        
        
        obj_name = dataset.frame_objs[idx]
        # import pdb; pdb.set_trace()
        #NOTE:下采样到2048点的物体顶点不能直接使用，需要从原模型获得全部顶点坐标
        ObjMesh = dataset.object_meshes[obj_name]
        ## 先从原物体模型获得原物体顶点
        obj_verts_orig = ObjMesh.vertices 
        obj_trans = sample['trans_obj']
        obj_rotmat = sample['root_orient_obj_rotmat'][0]
        ## 通过旋转、平移转换的矩阵操作获得实际顶点的坐标
        obj_verts = np.matmul(obj_verts_orig, obj_rotmat) + obj_trans
        obj_faces = ObjMesh.faces
        
        frame_name = str(idx) + '_' + obj_name
        Conditionator.run(hand_verts, hand_faces, obj_verts, obj_faces, frame_name, idx)
        
        
        
        
    