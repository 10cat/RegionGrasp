import os
import sys
sys.path.append('.')
sys.path.append('..')
import config
import numpy as np
import torch
import trimesh
import mano
from mano.model import load

from dataset.Dataset_origin import GrabNetDataset_orig
from dataset.data_utils import m2m_intersect, visual_inter, visual_mesh, visual_mesh_region
from utils.utils import func_timer, makepath
import matplotlib.pyplot as plt
from tqdm import tqdm


if __name__ == "__main__":
    import argparse
    import random
    # random.seed(1024)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--ds_name', type=str, required=True)
    
    
    args = parser.parse_args()
    
    if args.dataset== "GrabNet":
        dataset_root = config.GrabNet_ROOT
    output_root = os.path.join(config.visual_root, f"{args.dataset}_visual") 
    mano_path = config.mano_dir
    
    rh_model = load(model_path=mano_path, 
                    is_rhand=True, 
                    num_pca_comps=45, 
                    flat_hand_mean=True)
    if args.dataset == "GrabNet":
        dataset = GrabNetDataset_orig(dataset_root, ds_name=args.ds_name, dtype=np.float32)
    visual_folder = os.path.join(output_root, 'thumb_condition', 'Annotations', args.ds_name)
    makepath(visual_folder) 
    
    for idx in tqdm(range(dataset.__len__()), desc=f"{args.ds_name}set in {args.dataset}"):
        # TODO: to see the ratio of containing thumb contact annotations in the original GrabNet annotations
        
        sample = dataset.__getitem__(idx)
        import pdb; pdb.set_trace()
        
        sample.keys()