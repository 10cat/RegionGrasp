import argparse
import os
import sys
sys.path.append('.')
sys.path.append('..')
import pickle
from tqdm import tqdm
import config
import mano
from mano import load
import numpy as np
import torch
import trimesh
from sklearn.neighbors import KDTree
from utils.utils import func_timer, makepath
from dataset.Dataset_origin import GrabNetDataset_orig
from dataset.data_utils import faces2verts_no_rep
from dataset.data_utils import signed_distance

class GrabNetResample(GrabNetDataset_orig):
    def __init__(self, dataset_root, ds_name='train', frame_names_file='frame_names.npz', grabnet_thumb=False, obj_meshes_folder='contact_meshes', output_root=None, dtype=torch.float32, only_params=False, load_on_ram=False, resample_num=8192):
        super().__init__(dataset_root, ds_name, frame_names_file, grabnet_thumb, obj_meshes_folder, output_root, dtype, only_params, load_on_ram)
        self.resample_root = os.path.join(self.ts_root, 'object_meshes', obj_meshes_folder+'_resampled')
        makepath(self.resample_root)
        self.resample_dir = os.path.join(self.resample_root, f'N_{resample_num}')
        makepath(self.resample_dir)
        self.resampled_objs = self.resample_obj_mesh(N=resample_num)
        
        
    def resampling_surface(self, obj_mesh, N):
        return trimesh.sample.sample_surface(obj_mesh, N)
        
    def resample_obj_mesh(self, N):
        resampled = {}
        resample_paths = []
        for name, obj_mesh in tqdm(self.object_meshes.items(), desc='Loading object resampled points'):
            resampled_points_path = os.path.join(self.resample_dir, name + '.npy')
            resampled_faces_path = os.path.join(self.resample_dir, name + '_face_id.npy')
            if os.path.exists(resampled_points_path) and os.path.exists(resampled_faces_path) and not config.force_resample:
                obj_xyz_resampled = np.load(resampled_points_path)
                face_id = np.load(resampled_faces_path)
            else:
                obj_xyz_resampled, face_id = self.resampling_surface(obj_mesh, N)
                np.save(resampled_points_path, obj_xyz_resampled)
                np.save(resampled_faces_path, face_id)
                
            resampled[name] = {'points':obj_xyz_resampled, 'faces':face_id}
        return resampled
    
class GrabNetThumb(GrabNetResample):
    def __init__(self, dataset_root, ds_name='train', mano_path=None, frame_names_file='frame_names.npz', grabnet_thumb=False, obj_meshes_folder='contact_meshes', output_root=None, dtype=torch.float32, only_params=False, load_on_ram=False, resample_num=8192):
        super().__init__(dataset_root, ds_name, frame_names_file, grabnet_thumb, obj_meshes_folder, output_root, dtype, only_params, load_on_ram, resample_num)
        self.obj_rotmat = self.ds['root_orient_obj_rotmat']
        self.obj_trans = self.ds['trans_obj']
        self.mano_path = mano_path
        self.rh_model = load(model_path=self.mano_path, 
                             is_rhand=True, 
                             num_pca_comps=45, 
                             flat_hand_mean=True)
        
        self.thumb_vertices_ids = faces2verts_no_rep(self.rh_model.faces[config.thumb_center])
        
        self.contact_data_path = os.path.join(self.ds_path, 'data_contact')
        makepath(self.contact_data_path)
        
    @func_timer
    def get_obj_data(self, data, idx):
        obj_name = self.frame_objs[idx]
        obj_mesh = self.object_meshes[obj_name]
        obj_verts = obj_mesh.vertices
        
        obj_resp_points = self.resampled_objs[obj_name]['points']
        obj_resp_faces = self.resampled_objs[obj_name]['faces']
        rot_mat_np = np.array(self.obj_rotmat[idx][0])
        trans_np = np.array(self.obj_trans[idx])
        
        obj_verts_trans = np.matmul(obj_verts, rot_mat_np) + trans_np
        obj_resp_points_trans = np.matmul(obj_resp_points, rot_mat_np) + trans_np
        
        obj_mesh = trimesh.Trimesh(vertices=obj_verts_trans, faces=obj_mesh.faces)
        
        return obj_mesh, obj_verts_trans, obj_resp_points_trans, obj_resp_faces
    
    def get_hand_data(self, data):
        hand_verts = data['verts_rhand']
        hand_faces = self.rh_model.faces
        hand_mesh = trimesh.Trimesh(vertices=hand_verts, faces=hand_faces)
        return hand_verts, hand_mesh
    
    # @func_timer
    def thumb_query_point(self, HandMesh, ObjMesh, pene_th=0.002, contact_th=-0.005):
        thumb_vertices = HandMesh.vertices[self.thumb_vertices_ids]
        # ObjQuery = trimesh.proximity.ProximityQuery(ObjMesh)
        #  -- 以thumb_vertices_ids为query计算signed distances并返回相对应的closest faces
        #  the on_surface return is not signed_dists, 所以需要专门计算signed dists， 再用on_surface返回obj上最近的面
        # h2o_signed_dists = ObjQuery.signed_distance(thumb_vertices)
        # _, _, h2o_closest_fid = ObjQuery.on_surface(thumb_vertices)
        # TODO: trimesh.proximity.signed_distance包含了closest_points而没有输出triangle_ids,改写使输出triangle_ids -- 这样只用计算一次closest_points
        h2o_signed_dists, h2o_closest_fid = signed_distance(ObjMesh, thumb_vertices)
        # -- 用sdf_th阈值进一步选取thumb上真正的contact部分
        # NOTE: OUTSIDE mesh -> NEG； INSIDE the mesh -> POS
        penet_flag = h2o_signed_dists < pene_th
        contact_flag = h2o_signed_dists > contact_th
        flag = penet_flag & contact_flag
        obj_contact_fids = h2o_closest_fid[flag]
        # import pdb; pdb.set_trace()
        if obj_contact_fids.shape[0] == 0:
            return None, None
        elif obj_contact_fids.shape[0] == 1:
            point = ObjMesh.triangles_center[obj_contact_fids[0]]
        else:
            # import pdb; pdb.set_trace()
            tri_centers = np.array([ObjMesh.triangles_center[fid] for fid in obj_contact_fids])
            # TODO mean of the tri_centers
            point =  np.mean(tri_centers, axis=0)
            
        return point, obj_contact_fids
    
    # @func_timer
    def get_KNN_in_pc(self, PC, point_q, K=410):
        PC_tree = KDTree(PC)
        distance, indices = PC_tree.query(point_q.reshape(1, -1), K)
        distance = distance.reshape(-1)
        indices = indices.reshape(-1)
        return distance, indices
    
    def divide_pointcloud(self, PC, indices):
        N = PC.shape[0]
        indices_neg = list(set(range(N)) - set(indices))
        pc = [PC[i] for i in indices]
        rem_pc = [PC[i] for i in indices_neg]
        
        pc = np.array(pc)
        rem_pc = np.array(rem_pc)
        return pc, rem_pc
    
    def input_pc_sample(self, idx, PC, M=2048):
        N = PC.shape[0]
        np.random.seed(idx)
        indices = np.floor(np.random.random(M)*N).astype(np.int32)
        sample_pc = [PC[int(index)] for index in indices]
        sample_pc = np.array(sample_pc)
        return sample_pc, indices
        
    def __getitem__(self, idx):
        ## 直接通过self.ds获取这一个sample对应的data params
        data_out = {k: self.ds[k][idx] for k in self.ds.keys()}
        # import pdb; pdb.set_trace()
        annot = {}
        if not self.only_params:
            data = self.get_frames_data(idx, self.frame_names)
            # import pdb; pdb.set_trace()
            data_out.update(data)
            obj_mesh, obj_verts, obj_points, _= self.get_obj_data(data_out, idx)
            
            hand_verts, hand_mesh = self.get_hand_data(data_out)
            
            point_contact, obj_contact_fids = self.thumb_query_point(hand_mesh, obj_mesh)
            if point_contact is None:
                    # DONE: 筛掉没有手接触的sample
                return data_out, None, None
            
            dists, contact_indices = self.get_KNN_in_pc(obj_points, point_contact)
            
            # contact_pc, input_pc_hr = self.divide_pointcloud(obj_points, contact_indices)
            
            # input_pc = self.input_pc_sample(idx, input_pc_hr)
            
            # DONE:region_visual --> 820有点太大了，先取一半吧410
            # PC_contact = trimesh.PointCloud(vertices=contact_ps, colors=colors_like(config.colors['yellow']))
            # PC_rem = trimesh.PointCloud(vertices=rem_ps, colors=colors_like(config.colors['green']))
            
            # PC_contact.export('test_pc_contact.ply')
            # PC_rem.export('test_pc_rem.ply')
            # ObjMesh.export('test_mesh.ply')
            # HandMesh.export('hand_mesh.ply')
            
            # import pdb; pdb.set_trace()
            contact_mask = np.zeros((obj_points.shape[0], 1))
            contact_mask[contact_indices] = 1.
            
            annot['pmask'] = contact_mask
            annot['center_point'] = np.array(point_contact)
            annot['contact_faces'] = np.array(obj_contact_fids)
            
            path = os.path.join(self.contact_data_path, f'{idx}.pkl')
            with open(path, 'wb') as f:
                pickle.dump(annot, f)
                
            
        return data_out, np.array(point_contact), np.array(obj_contact_fids)
    
def readpkl(dataset, idx):
    path = os.path.join(dataset.contact_data_path, f'{idx}.pkl')
    with open(path, 'rb') as f:
        annot = pickle.load(f)
    contact_center = annot['center_point']
    contact_faces = annot['contact_faces']
    return contact_center, contact_faces
        
def get_thumb_condition(ds_root, args):
    dataset = GrabNetThumb(dataset_root = ds_root,
                           ds_name = args.split,
                           dtype=np.float32,
                           )
    frame_names_thumb_list = []
    ds_orig = dataset.ds
    ds_thumb = {k:[] for k in list(ds_orig.keys())}
    ds_thumb['contact_center'] = []
    # ds_thumb['contact_faces'] = []
    sample_ids = []
    
    if args.use_cache:
        indices = [int(dir.split('/')[-1].split('.')[0]) for dir in os.listdir(dataset.contact_data_path)]
        indices.sort()
        pbar = tqdm(indices, desc=f'Loading the saved pkl files')
    else:
        pbar = tqdm(range(args.start, dataset.__len__()), desc=f'Annotating thumb contact in {dataset.ds_name} set')
        
    # import pdb; pdb.set_trace()
        
    for idx in pbar:
        # print(idx)
        if args.use_cache:
            center_point, obj_faces = readpkl(dataset, idx)
        else:
            data_out, center_point, obj_faces = dataset.__getitem__(idx)
            
        if center_point is None:
            continue
        
        frame_names_thumb_list.append(dataset.frame_names_orig[idx])
        sample_ids.append(idx)
        for key in list(ds_thumb.keys()):
            if key not in ['contact_center']:
                ds_thumb[key].append(ds_orig[key][idx])
            elif key == 'contact_center':
                ds_thumb[key].append(center_point)
        
        # if idx > 5:     
        #     break
                
    
    # NOTE: output 1) frame_names_thumb: 存储每个frame对应文件路径名的文件 2）d_thumb:frame基本数据+新标注的contact_mask
    frame_names_thumb = np.asarray(frame_names_thumb_list)
    frame_names_path = os.path.join(dataset.ds_path, 'frame_names_thumb_N.npz')
    np.savez(frame_names_path, frame_names=frame_names_thumb)
    ds_thumb = {k: np.array(ds_thumb[k]) for k in list(ds_thumb.keys())}
    ds_thumb_path = os.path.join(dataset.ds_path, f'grabnet_{dataset.ds_name}_thumb_N.npz')
    np.savez(ds_thumb_path, 
             global_orient_rhand_rotmat = ds_thumb['global_orient_rhand_rotmat'], 
             fpose_rhand_rotmat = ds_thumb['fpose_rhand_rotmat'], 
             trans_rhand = ds_thumb['trans_rhand'], 
             trans_obj = ds_thumb['trans_obj'], 
             root_orient_obj_rotmat = ds_thumb['root_orient_obj_rotmat'], 
             global_orient_rhand_rotmat_f = ds_thumb['global_orient_rhand_rotmat_f'], 
             fpose_rhand_rotmat_f = ds_thumb['fpose_rhand_rotmat_f'], 
             trans_rhand_f = ds_thumb['trans_rhand_f'],
             contact_center = ds_thumb['contact_center'])
    
    list_path = os.path.join(dataset.ds_path, 'samples_id.npy')
    np.save(list_path, np.array(sample_ids))
    

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--use_cache', action='store_true')
    args = parser.parse_args()
    import psutil
    p = psutil.Process()
    cpu_list = p.cpu_affinity()
    print(cpu_list)
    p.cpu_affinity(cpu_list)
    
    get_thumb_condition(config.DATASET_ROOT, args)