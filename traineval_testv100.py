import os
import sys

from tqdm import tqdm

# from dataloader import Batchsampler_testv100
sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch
from torch.utils import data
import torch.optim as optim

import argparse
import config
# from option import MyOptions
from dataset.Dataset import GrabNetDataset

# from epochbase import TrainEpoch, ValEpoch
import random
from utils.utils import set_random_seed

# from dataset.obman_preprocess import ObManObj
# from models.ConditionNet import ConditionMAE, ConditionTrans, ConditionBERT
# from models.cGrasp_vae import cGraspvae
# from traineval_utils.loss import ChamferDistanceL2Loss, PointCloudCompletionLoss, cGraspvaeLoss
# from utils.optim import *
# from utils.datasets import get_dataset
# from utils.epoch_utils import EpochVAE_comp, ValEpochVAE_comp, EpochVAE_mae, ValEpochVAE_mae,  MetersMonitor, model_update, PretrainEpoch

if __name__ == "__main__":
    # import wandb
    import argparse
    # from omegaconf import OmegaConf
    from easydict import EasyDict
    import utils.cfgs as cfgsu
    
    # import psutil
    # p = psutil.Process()
    # p.cpu_affinity(range(16))
    # print(p.cpu_affinity())
    
    # import pdb; pdb.set_trace()
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfgs', type=str, default='thumb_mae_grabnet_og')
    parser.add_argument('--cfgs_fodler', type=str, default='cgrasp')
    parser.add_argument('--exp_name', type=str, default='thumb_mae_grabnet_og')
    parser.add_argument('--cuda_id', type=str, default="0")
    parser.add_argument('--machine', type=str, default='208')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--same', action='store_true') # add this to set same 32 data samples to every batch
    parser.add_argument('--batch_size', type=int, default=32)

    args = parser.parse_args()

    conf = cfgsu.get_config(args, args.cfgs_fodler)
    conf.update(cfgsu.config_exp_name(args))
    conf.update(cfgsu.config_paths(args.machine, conf['exp_name']))
    conf.update(args.__dict__)
    
    cfg = EasyDict()
    # DONE: transform the dict config to easydict
    cfg = cfgsu.merge_new_config(cfg, conf)
    # conf = OmegaConf.structured(cfg)
    # import pdb; pdb.set_trace()
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.cuda_id
    os.environ['OMP_NUM_THREAD'] = '1'
    torch.set_num_threads(1)
    
    cfgsu.save_experiment_config(cfg)
    
    
    ds_root = cfg.grabnet_root
    configs = cfg.dataset['train']._base_.kwargs
    dataset = GrabNetDataset(dataset_root=ds_root, 
                            ds_name='train',
                            mano_path=cfg.mano_rh_path,
                            sample_same=cfg.same,
                            batch_size=cfg.batch_size,
                            **configs)
    # trainset = get_dataset(cfg, mode='train')
    dataloader = data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)       
    
    for idx, sample in enumerate(tqdm(dataloader)):
        obj_input_pc = sample['input_pc']
        gt_rhand_vs = sample['hand_verts'].transpose(2, 1)
        mask_centers = sample['contact_center']
        # import pdb; pdb.set_trace()
        sample_ids = sample['sample_id']
        
        # test cuda
        # obj_input_pc = sample['input_pc'].to('cuda')
        # gt_rhand_vs = sample['hand_verts'].transpose(2, 1).to('cuda')
        # mask_centers = sample['contact_center'].to('cuda')
        # # import pdb; pdb.set_trace()
        # sample_ids = sample['sample_id'].to('cuda')
        
        
    
    

    


