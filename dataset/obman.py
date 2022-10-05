import os
import sys

from tools.cfg_parser import makepath
sys.path.append('.')
sys.path.append('..')
import pickle
from collections import Counter

import cv2
import numpy as np
from tqdm import tqdm

import tools
from tools.obman_utils import fast_load_obj


class ObMan():
    def __init__(self, 
                 root,
                 shapenet_root,
                 split = 'train',
                 joint_nb = 21,
                 mini_factor = None, # ?
                 use_cache = False, # ?
                 root_palm = False, # ?
                 mode = 'all', # rgb_folder: [all|obj|hand]
                 segment = False, # ?
                 use_external_points = True, # ?
                 apply_obj_transform = True):
        
        self.split = split
        self.root_palm = root_palm
        self.mode = mode
        self.segment = segment
        self.root = os.path.join(root, split)
        self.apply_obj_transform = apply_obj_transform

        if shapenet_root.endswith('/'):
            shapenet_root = shapenet_root[:-1]
        self.shapenet_root = shapenet_root

        self.use_external_points = use_external_points

        if mode == 'all':
            self.rgb_folder = os.path.join(self.root, "rgb")
        elif mode == 'obj':
            self.rgb_folder = os.path.join(self.root, "rgb_obj")
        elif mode == 'hand':
            self.rgb_folder = os.path.join(self.root, "rgb_hand")
        else:
            raise ValueError(
                'Mode should be in [all|obj|hand], got {}'.format(mode)
            )

        
        # Cache information
        self.use_cache = use_cache
        self.name = 'obman'
        self.cache_folder = os.path.join('data', 'cache',
                                         '{}'.format(self.name))
        os.makedirs(self.cache_folder, exist_ok=True)

        # common parameters configuration
        self.mini_factor = mini_factor
        self.cam_intr = np.array([[480., 0., 128.], [0., 480., 128.],
                                  [0., 0., 1.]]).astype(np.float32)
        self.cam_extr = np.array([[1., 0., 0., 0.], [0., -1., 0., 0.],
                                  [0., 0., -1., 0.]]).astype(np.float32)
        
        self.joint_nb = joint_nb
        self.segm_folder = os.path.join(self.root, 'segm')
        self.prefix_template = '{:08d}'
        self.meta_folder = os.path.join(self.root, 'meta')
        self.coord2d_folder = os.path.join(self.root, "coords2d")

        # Define links on skeleton 结点标注序号不同数据集不尽相同
        self.links = [(0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12),
                      (0, 13, 14, 15, 16), (0, 17, 18, 19, 20)]
        
        # Load mano faces
        self.faces = {}
        for side in ['left', 'right']:
            with open('mano_faces_{}.pkl'.format(side), 'rb') as p_f:
                self.faces[side] = pickle.load(p_f)
        
        self.shapenet_template = self.shapenet_root + '/{}/{}/models/model_normalized.pkl'
        self.load_dataset()

    def _get_image_path(self, prefix):
        image_path = os.path.join(self.rgb_folder, '{}.jpg'.format(prefix))

        return image_path

    def _get_obj_path(self, class_id, sample_id):
        shapenet_path = self.shapenet_template.format(class_id, sample_id)
        return shapenet_path
    
    def load_dataset(self):
        
        """
        According to the requirements of our tasks, this function only includes the loading and the processing of 3D mesh information.
        """

        # prepare to read the data file in the provided folder
        cache_path = os.path.join(
            self.cache_folder, '{}_{}_mode_{}.pkl'.format(
                self.split, self.mini_factor, self.mode))
        if os.path.exists(cache_path) and self.use_cache:
            with open(cache_path, 'rb') as cache_f:
                annotations = pickle.load(cache_f)
            print('Cached information for dataset {} loaded from {}'.format(
                self.name, cache_path))
        else:
            idxs = [
                int(imgname.split('.')[0])
                for imgname in sorted(os.listdir(self.meta_folder))
            ]

            if self.mini_factor:
                mini_nb = int(len(idxs) * self.mini_factor)
                idxs = idxs[:mini_nb]
            prefixes = [self.prefix_template.format(idx) for idx in idxs]
            print('Got {} samples for split {}'.format(len(idxs), self.split))

            image_names = []
            all_joints3d = []
            hand_sides = []
            hand_poses = []
            hand_pcas = []
            hand_verts3d = []
            obj_paths = []
            obj_transforms = []
            meta_infos = []
            depth_infos = []

            for idx, prefix in enumerate(tqdm(prefixes)):
                meta_path = os.path.join(self.meta_folder,
                                         '{}.pkl'.format(prefix))
                with open(meta_path, 'rb') as meta_f:
                    meta_info = pickle.load(meta_f)
                    image_path = self._get_image_path(prefix)
                    image_names.append(image_path)
                    # loading hand info
                    all_joints3d.append(meta_info['coords_3d'])
                    hand_verts3d.append(meta_info['verts_3d'])
                    hand_sides.append(meta_info['side'])
                    hand_poses.append(meta_info['hand_pose'])
                    hand_pcas.append(meta_info['pca_pose'])
                    depth_infos.append({
                        'depth_min':
                        meta_info['depth_min'],
                        'depth_max':
                        meta_info['depth_max'],
                        'hand_depth_min':
                        meta_info['hand_depth_min'],
                        'hand_depth_max':
                        meta_info['hand_depth_max'],
                        'obj_depth_min':
                        meta_info['obj_depth_min'],
                        'obj_depth_max':
                        meta_info['obj_depth_max']
                    })

                    # loading object info
                    obj_path = self._get_obj_path(meta_info['class_id'],        meta_info['sample_id'])
                    obj_paths.append(obj_path)
                    obj_transforms.append(meta_info['affine_transform'])

                    meta_info_full = {
                        'obj_scale': meta_info['obj_scale'],
                        'obj_class_id': meta_info['class_id'],
                        'obj_sample_id': meta_info['sample_id']
                    }
                    if 'grasp_quality' in meta_info:
                        meta_info_full['grasp_quality'] = meta_info['grasp_quality']
                        meta_info_full['grasp_epsilon'] = meta_info['grasp_epsilon']
                        meta_info_full['grasp_volume'] = meta_info['grasp_volume']
                    meta_infos.append(meta_info_full)
            
            # annotations load
            annotations = {
                'depth_infos': depth_infos,
                'joints3d': all_joints3d,
                'hand_sides': hand_sides,
                'hand_poses': hand_poses,
                'hand_pcas': hand_pcas,
                'hand_verts3d': hand_verts3d,
                'obj_paths': obj_paths,
                'obj_transforms': obj_transforms,
                'meta_infos': meta_infos
            }

            # print the basic information of loaded annotations:
            print('object class number: {}'.format(
                np.unique(
                    [(meta_info['obj_class_id']) for meta_info in meta_infos], axis=0
                ).shape
            ) )
            print('total sample number (each class * sample): {}'.format(
                np.unique(
                    [(meta_info['obj_class_id'], meta_info['obj_sample_id'])
                     for meta_info in meta_infos],
                     axis=0).shape
            ))
            with open(cache_path, 'wb') as fid:
                pickle.dump(annotations, fid)
            print('Wrote cache for dataset {} to {}'.format(
                self.name, cache_path))

            # Set dataset attributes
            all_objects = [
                obj[:-7].split('/')[-1].split('_')[0]
                for obj in annotations['obj_paths']
            ]
            selected_idxs = list(range(len(all_objects)))
            obj_paths = [annotations['obj_paths'][idx] for idx in selected_idxs]
            joints3d = [annotations['joints3d'][idx] for idx in selected_idxs]
            hand_sides = [annotations['hand_sides'][idx] for idx in selected_idxs]
            hand_pcas = [annotations['hand_pcas'][idx] for idx in selected_idxs]
            hand_verts3d = [
                annotations['hand_verts3d'][idx] for idx in selected_idxs
            ]
            obj_transforms = [
                annotations['obj_transforms'][idx] for idx in selected_idxs
            ]
            meta_infos = [annotations['meta_infos'][idx] for idx in selected_idxs]
            if 'depth_infos' in annotations:
                has_depth_info = True
                depth_infos = [
                    annotations['depth_infos'][idx] for idx in selected_idxs
                ]
            else:
                has_depth_info = False

            objects = [
                obj.split('_')[0]
                for obj in set([obj[:-7].split('/')[-1] for obj in obj_paths])
            ]

            unique_objects = set(objects)
            print('Got {} out instances of {} unique objects {}'.format(
                len(objects), len(unique_objects), unique_objects))
            freqs = Counter(objects)
            print(freqs)
            if has_depth_info:
                self.depth_infos = depth_infos
            self.image_names = image_names
            self.joints3d = joints3d
            self.hand_sides = hand_sides
            self.hand_pcas = hand_pcas
            self.hand_verts3d = hand_verts3d
            self.obj_paths = obj_paths
            self.obj_transforms = obj_transforms
            self.meta_infos = meta_infos
            # Initialize cache for center and scale in case objects are used
            self.center_scale_cache = {}

    def get_obj_verts_faces(self, idx):
        model_path = self.obj_paths[idx]
        model_path = model_path.replace(
            '/sequoia/data2/dataset/shapenet/ShapeNetCore.v2',
            self.shapenet_root)
        model_path_obj = model_path.replace('.pkl', '.obj')
        if os.path.exists(model_path):
            with open(model_path, 'rb') as obj_f:
                mesh = pickle.load(obj_f)
        elif os.path.exists(model_path_obj):
            with open(model_path_obj, 'r') as m_f:
                mesh = fast_load_obj(m_f)[0]
        else:
            raise ValueError(
                'Could not find model pdl or obj file at {}'.format(
                    model_path.split('.')[-2]
                )
            )
        verts = mesh['vertices']

        # Apply transforms for object vertices with extrinsic camera model
        obj_transform = self.obj_transforms[idx]
        hom_verts = np.concatenate([verts, np.ones([verts.shape[0], 1])], axis = 1)
        trans_verts = obj_transform.dot(hom_verts.T).T[:, :3]
        trans_verts = self.cam_extr[:3, :3].dot(
            trans_verts.transpose()).transpose()
        return np.array(trans_verts).astype(np.float32), np.array(
            mesh['faces']).astype(np.int16)

    def get_objpoints3d(self, idx, point_nb=600):
        """
        Function: ? Directly read the surface points of the object mesh from surface_points.pkl file?
        """
        model_path = self.obj_paths[idx].replace('model_normalized.obj',
                                                 'surface_points.pkl')
        with open(model_path, 'rb') as obj_f:
            points = pickle.load(obj_f)

        # Filter very far outlier points from modelnet/shapenet !!
        point_nb_or = points.shape[0]
        points = points[np.linalg.norm(points, 2, 1) < 20 * 
                        np.median(np.linalg.norm(points, 2, 1))] # ? Is 20 a selected hyper-parameter?
        if points.shape[0] < point_nb_or:
            #TODO get image_name  / leave out the print out message
            print('Filtering {} points out of {} for sample {} from split {}'.
                  format(point_nb_or - points.shape[0], point_nb_or,
                         self.image_names[idx], self.split)) 
        # TODO Change the below sampling strategy to a face optimized one
        idxs = np.random.choice(points.shape[0], point_nb)
        points = points[idxs]
        # Apply transforms for object surface points
        if self.apply_obj_transform:
            obj_transform = self.obj_transforms[idx]
            hom_points = np.concatenate(
                [ points, np.ones( [ points.shape[0], 1 ] ) ], axis = 1
            )
            trans_points = obj_transform.dot(hom_points.T).T[:, :3]
            trans_points = self.cam_extr[:3, :3].dot(
                trans_points.transpose()).transpose()
        else:
            trans_points = points
        return trans_points.astype(np.float32)

    def get_sides(self, idx):
        return self.hand_sides[idx]

    def get_camintr(self, idx):
        return self.cam_intr

    def get_verts3d(self, idx):
        verts3d = self.hand_verts3d[idx]
        verts3d = self.cam_extr[:3, :3].dot(verts3d.transpose()).transpose()
        return verts3d
    
    def get_faces3d(self, idx):
        faces = self.faces[self.get_sides(idx)]
        return faces

    def get_joints3d(self, idx):
        joints3d = self.joints3d[idx]
        if self.root_palm:
            # Replace wrist with palm
            verts3d = self.hand_verts3d[idx]
            palm = (verts3d[95] + verts3d[218]) / 2
            joints3d = np.concatenate([palm[np.newaxis, :], joints3d[1:]])
        # No hom coordinates needed because no translation ==> hand centric coordinate!!
        assert np.linalg.norm(
            self.cam_extr[:, 3]) == 0, 'extr camera should have no translation'

        joints3d = self.cam_extr[:3, :3].dot(joints3d.transpose()).transpose()
        return joints3d


class RayCastingContact():
    def __init__(self,
                 dir_sample_num = 50):
        #self.dir_sample_num = dir_sample_num
        """
        生成从原点发出、全方位扫描的射线
        方向采样数default=100 => 1000,000 rays, stride 0.01 in every direction
        """
        values_dx = np.arange(-dir_sample_num, dir_sample_num, 1) / dir_sample_num
        values_dy = np.arange(-dir_sample_num, dir_sample_num, 1) / dir_sample_num
        values_dz = np.arange(-dir_sample_num, dir_sample_num, 1) / dir_sample_num
        print(f"direction_x_value: {values_dx}; direction_y_value: {values_dy}; direction_z_value: {values_dz};")
        ray_directions = []
        #ray_sources = []
        for xval in values_dx:
            for yval in values_dy:
                for zval in values_dz:
                    ray_directions.append(np.array([[xval, yval, zval]])) #TODO check 坐标order
                    #ray_sources.append(np.array([[0, 0, 0]])) #TODO check 这个点需要是物体中心，按道理object centric的原点是物体中心

        self.ray_directions = np.concatenate(ray_directions, axis=0)
        # self.ray_sources = np.concatenate(ray_sources, axis=0)
        # print(f"ray direction shape: {self.ray_directions.shape}; ray sources shape: {self.ray_sources.shape}")

    def initialzie(self, 
                   ray_source_point = [0, 0, 0],
                   HandMesh=None,
                   ObjMesh=None):
        ray_source = np.array(ray_source_point) # (1,3)
        self.ray_sources = np.repeat(self.ray_direcions.shape[0], axis=0)
        self.ObjRaycaster = trimesh.ray.ray_pyembree.RayMeshIntersector(ObjMesh)
        self.HandRaycaster = trimesh.ray.ray_pyembree.RayMeshIntersector(HandMesh)

    def get_contact(self,
                    direction_map=None):

        hand_tri_idxs, hand_ray_idxs, hand_points = self.HandRaycaster.intersects_id(ray_origins=self.ray_sources, ray_directions=self.ray_directions, return_locations=True)
        

        obj_tri_idxs, obj_ray_idxs, obj_points = self.ObjRaycaster.intersects_id(ray_origins=self.ray_sources, ray_directions=self.ray_directions, return_locations=True)

        hoi_ray_idxs, hoi_hand_idx, hoi_obj_idx = np.intersect1d(hand_ray_idxs, obj_ray_idxs, return_indices=True) 
        # np.intersect1d可以通过return_indices返回相应的编号

        hoi_objfaces_idxs = obj_tri_idxs[hoi_obj_idx]
        hoi_objpoints = obj_points[hoi_obj_idx]
        ObjMesh.set_face_colors(colors['pink'], face_ids=hoi_objfaces_idxs)

    
    
if __name__ == '__main__':
    import smplx
    import trimesh
    import cv2
    import argparse
    import config as cfg
    from tqdm import tqdm

    from utils.meshviewer import Mesh, MeshViewer, points2sphere, colors
    from utils.objectmodel import ObjectModel

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, help="Path to dataset root")
    parser.add_argument(
        '--shapenet_root', required=True, help="Path to root of ShapeNetCore.v2")
    parser.add_argument(
        '--split', type=str, default='train', help='Usually [train|test]')
    parser.add_argument(
        '--mode',
        default='all',
        choices=['all', 'obj', 'hand'],
        help='Mode for synthgrasp dataset')
    parser.add_argument('--segment', action='store_true')
    parser.add_argument(
        '--mini_factor', type=float, help='Ratio in data to use (in ]0, 1[)')
    parser.add_argument(
        '--root_palm', action='store_true', help='Use palm as root')
    
    parser.add_argument(
        '--idx_0', type=int, default = 1, help='idx of the first image to display')
    parser.add_argument(
        '--idx_end', type=int, default = 0, help='idx of the last image to display')
    parser.add_argument('--use_cache', action='store_true', help='Use cache')
    parser.add_argument(
        '--idx_step', type=int, default = 1, help='idx of the first image to display')
    parser.add_argument(
        '--output_path', type=str, help='output path of the display experiment')
    args = parser.parse_args()

    obman = ObMan(
        args.root,
        args.shapenet_root,
        split = args.split,
        use_cache = args.use_cache,
        root_palm = args.root_palm,
        mini_factor = args.mini_factor,
        mode = args.mode,
        segment = args.segment)

    if args.idx_end == 0:
        idx_end = obman.__len__()
    else:
        idx_end = args.idx_end
    
    sample_indices = np.arange(args.idx_0, idx_end, args.idx_step)
    for idx in tqdm(sample_indices):
        image_name = obman.image_names[idx]
        hand_verts3d = obman.get_verts3d(idx)
        hand_faces = obman.get_faces3d(idx)
        obj_verts3d, obj_faces = obman.get_obj_verts_faces(idx)
        hand_joints3d = obman.get_joints3d(idx)

        # initialize the mesh of hand, object and the origin sphere
        HandMesh = Mesh(
            vertices=hand_verts3d,
            faces=hand_faces,
            vc=cfg.colors['skin']
        )

        ObjMesh = Mesh(
            vertices=obj_verts3d,
            faces=obj_faces,
            vc=cfg.colors['grey']
        )

        originSphere = trimesh.primitives.Sphere(radius=0.001, center=[0,0,0])

        prefix_jpg = image_name.split('/')[-1]
        prefix = prefix_jpg.split('.')[0]

        makepath(args.output_path)
        name = args.output_path + prefix
        HandMesh.export( name + '_hand.ply')
        ObjMesh.export( name + '_obj.ply')
        originSphere.export( name + '_origin.ply')



        

