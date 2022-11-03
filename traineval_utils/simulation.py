import os
import pickle
from subprocess import Popen
import shutil
import time
import tempfile
import numpy as np
import pybullet as p
import skvideo.io as skvio


def process_sample(sample_idx, sample, save_gif_folder=None, save_obj_folder=None, vhacd_exe=None, use_gui=False, wait_time=0, sample_vis_freq=10, save_all_steps=True):
    if use_gui:
        conn_id = p.connect(p.GUI)
    else:
        conn_id = p.connect(p.DIRECT)

    if sample_idx % sample_vis_freq == 0:
        save_video = True
        save_video_path = os.path.join(save_gif_folder, "{:08d}.gif".format(sample_idx))
        save_obj_path = os.path.join(save_obj_folder, "{:08d}_obj.obj".format(sample_idx))
        save_hand_path = os.path.join(save_obj_folder, "{:08d}_hand.obj".format(sample_idx))

def run_simulation(hand_verts, hand_faces, obj_verts, obj_faces,
                   conn_id=None, vhacd_exe=None, sample_idx=None,
                   save_video=False, save_video_path=None,
                   simulation_step=1 / 240, num_iterations=35,
                   object_friction=3, hand_friction=3, # friction parameter
                   hand_restitution=0, object_restitution=0.5,
                   object_mass=1, verbose=False, vhacd_resolution=1000, wait_time=0,
                   save_hand_path=None, save_obj_path=None,
                   save_simul_folder=None, use_gui=False):
    if conn_id is None:
        if use_gui:
            conn_id = p.connect(p.GUI)
        else:
            conn_id = p.connect(p.DIRECT)
    hand_indicies = hand_faces.flatten().tolist() # hand_verts -- (list)
    p.resetSimulation(physicsClientId=conn_id)
    p.setPhysicsEngineParameter(enableFileCaching=0, physicsClientId=conn_id)
    return



