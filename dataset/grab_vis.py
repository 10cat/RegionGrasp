from fileinput import filename
from socket import INADDR_MAX_LOCAL_GROUP
import sys
sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch
import os, glob
import smplx
import argparse
# from bps_torch.bps import bps_torch
# from bps import bps

from tqdm import tqdm
from tools.objectmodel import ObjectModel
from tools.cfg_parser import Config
from tools.utils import makepath, makelogger
from mano.utils import Mesh
from tools.utils import parse_npz
from tools.utils import params2torch
from tools.utils import prepare_params
from tools.utils import to_cpu
from tools.utils import append2dict
from tools.utils import np2torch

INTENTS = ['lift', 'pass', 'offhand', 'use', 'all']

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
JOINTS_NUM = 15

class GRABvisual(object):
    def __init__(self, cfg, logger=None, **params):
        self.cfg = cfg
        self.grab_path = cfg.grab_path
        self.out_path = cfg.out_path

        makepath(self.out_path)

        assert cfg.intent in INTENTS

        self.intent = cfg.intent

        if cfg.splits is None:
            # train/val/test split by the category of objects
            self.splits = {'test': ['mug', 'wineglass', 'camera', 'binoculars', 'fryingpan', 'toothpaste'],
                            'val': ['apple', 'toothbrush', 'elephant', 'hand'],
                            'train': []}
        else:
            assert isinstance(cfg.splits, dict)
            self.splits = cfg.splits

        self.all_seqs = glob.glob(self.grab_path + '/*/*.npz')

        self.selected_seqs = []
        self.obj_based_seqs = {}
        self.sbj_based_seqs = {}
        self.split_seqs = {'test': [],
                            'val': [],
                            'train': []
                            }
        
        self.process_sequences() # obtain self.selected_sequence / self.split_seqs

    def process_sequences(self):
        for sequence in self.all_seqs:
            subject_id = sequence.split('/')[-2]
            action_name = os.path.basename(sequence)
            object_name = action_name.split('_')[0]

            if self.intent == 'all':
                pass
            elif self.intent == 'use' and any (intnt in action_name for intnt in INTENTS[:3]):
                continue
            elif self.intent not in action_name:
                continue

            if object_name not in self.obj_based_seqs:
                self.obj_based_seqs[object_name] = [sequence]
            else:
                self.obj_based_seqs[object_name].append(sequence)
            
            if subject_id not in self.sbj_based_seqs:
                self.sbj_based_seqs[subject_id] = [sequence]
            else:
                self.sbj_based_seqs[subject_id].append(sequence)
            
            self.selected_seqs.append(sequence)

            if object_name in self.splits['test']:
                self.split_seqs['test'].append(sequence)
            elif object_name in self.splits['val']:
                self.split_seqs['val'].append(sequence)
            else:
                self.split_seqs['train'].append(sequence)
                if object_name not in self.splits['train']:
                    self.splits['train'].append(object_name)

    def filter_contact_rhand_frames(self, seq_data):
        if self.cfg.only_contact:
            frame_mask = (seq_data['contact']['object']>40).any(axis=1)
        else:
            frame_mask = (seq_data['contact']['object']>-1).any(axis=1)
        
        return frame_mask

    def load_obj_info(self, obj_name, seq_data):
        mesh_path = os.path.join(self.grab_path, '..', seq_data.object.object_mesh)
        if obj_name not in self.obj_info:
            obj_mesh = Mesh(filename=mesh_path)
            faces_obj = np.array(obj_mesh.faces)
            verts_obj = np.array(obj_mesh.vertices)

            self.obj_info[obj_name] = {'verts': verts_obj,
                                        'faces': faces_obj,
                                        'obj_mesh_file': mesh_path}
        
        return self.obj_info[obj_name]


    def data_visualize(self, cfg):

        self.subject_mesh = {}
        self.obj_info = {}
        self.sbj_info = {}

        for split in self.split_seqs.keys():
            frame_names = []
            split_output_path = os.path.join(self.out_path, split)
            makepath(split_output_path)
            
            for sequence in tqdm(self.split_seqs[split]):

                seq_data = parse_npz(sequence)

                #TODO filter the frames with contact points and exclude those without any hoi contact
                frame_mask = self.filter_contact_rhand_frames (seq_data)

                T = frame_mask.sum()
                if T < 100: # sequence中右手的contact少于100帧就放弃这条sequence，可调数值
                    continue # if no frame is selected continue to the next sequence

                # if in contact with rhand, extract the basic information for 

                action_name = os.path.basename(sequence)
                action_name = action_name.split('.')[0]
                obj_name = seq_data.obj_name
                sbj_id = seq_data.sbj_id
                n_comps = seq_data.n_comps
                gender = seq_data.gender

                sbj_output_path = os.path.join(split_output_path, sbj_id)
                makepath(sbj_output_path)
                action_output_path = os.path.join(sbj_output_path, action_name)
                makepath(action_output_path)

                rh_params = prepare_params(seq_data.rhand.params, frame_mask)
                obj_params = prepare_params(seq_data.object.params, frame_mask)

                ## for right hand
                
                rh_mesh = os.path.join(grab_path, '..', seq_data.rhand.vtemp)
                rh_vtemp = np.array(Mesh(filename=rh_mesh).vertices)

                rh_m = smplx.create(model_path = cfg.model_path,
                                    model_type = 'mano',
                                    is_rhand = True,
                                    v_template = rh_vtemp,
                                    num_pca_comps = n_comps,
                                    flat_hand_mean = True,
                                    batch_size = T)

                rh_parms = params2torch(rh_params)
                verts_rh = to_cpu(rh_m(**rh_parms).vertices)

                ## for object
                obj_info = self.load_obj_info(obj_name, seq_data)
                obj_m = ObjectModel(v_template=obj_info['verts'],
                                    batch_size = T)
                obj_parms = params2torch(obj_params)
                verts_obj = to_cpu(obj_m(**obj_parms).vertices) # transformed vertices of the whole sequence
                

                # contact data
                contact_data = seq_data.contact.object[frame_mask]

                #TODO visualize the sequence with 10 sampled frames

                sample_indices = np.arange(0, T, int(T / 10))

                for idx in sample_indices:
                    ## for rhand
                    frame_verts_rh = verts_rh[idx]
                    frame_verts_obj = verts_obj[idx]
                    frame_contact = contact_data[idx]

                    RhandMesh = Mesh(vertices=frame_verts_rh, faces=rh_m.faces, vc=colors['skin'])
                    ObjMesh = Mesh(vertices=frame_verts_obj, faces=obj_info['faces'], vc=colors['grey'])
                    ObjMesh.set_vertex_colors(vc=colors['pink'], vertex_ids=frame_contact)

                    frame_name = 'frame_' + str(idx)

                    RhandMesh.export(os.path.join(action_output_path, frame_name + '_rhand.ply'))
                    ObjMesh.export(os.path.join(action_output_path, frame_name + '_obj.ply'))




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GRAB-vertices')

    parser.add_argument('--grab-path', required=True, type=str,
                        help='The path to the downloaded grab data')
    parser.add_argument('--out-path', default=None, type=str,
                        help='The path to the folder to save the processed data')
    parser.add_argument('--model-path', required=True, type=str,
                        help='The path to the folder containing smplx models')
    parser.add_argument('--process-id', required=True, type=str,
                        help='ID for the processed data')

    args = parser.parse_args()

    grab_path = args.grab_path
    out_path = args.out_path
    model_path = args.model_path
    process_id = args.process_id

    grab_splits = { 'test': ['mug', 'wineglass', 'camera', 'binoculars', 'fryingpan', 'toothpaste'],
                    'val': ['apple', 'toothbrush', 'elephant', 'hand'],
                    'train': []}

    cfg = {

        'intent':'all', # from 'all', 'use' , 'pass', 'lift' , 'offhand'
        'only_contact':True, # if True, returns only frames with contact
        'save_body_verts': False, # if True, will compute and save the body vertices
        'save_lhand_verts': True, # if True, will compute and save the body vertices
        'save_rhand_verts': True, # if True, will compute and save the body vertices
        'save_object_verts': True,

        'save_contact': True, # if True, will add the contact info to the saved data

        # splits
        'splits':grab_splits,

        #IO path
        'grab_path': grab_path,
        'out_path': os.path.join(out_path, process_id),

        # number of vertices samples for each object
        'n_verts_sample': 2048,

        # body and hand model path
        'model_path':model_path,
    }

    cwd = os.getcwd()
    default_cfg_path = os.path.join(cwd, '../configs/grab_preprocessing_cfg.yaml')
    cfg = Config(default_cfg_path=default_cfg_path, **cfg)
    cfg.write_cfg(write_path=cfg.out_path+'/grab_preprocessing_cfg.yaml')

    log_dir = os.path.join(cfg.out_path, 'grab_processing.log')
    logger = makelogger(log_dir=log_dir, mode='a').info

    # logger(instructions)

    visualizer = GRABvisual(cfg, logger)

    visualizer.data_visualize(cfg)