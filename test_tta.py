import os
import sys

from tqdm import tqdm
from zmq import device

# from Codes.Hand.ConditionHOI.GraspTTA.network.affordanceNet_obman_mano_vertex import affordanceNet

sys.path.append('.')
sys.path.append('..')

import argparse
import random

import config
import numpy as np
import torch
import torch.optim as optim
# from option import MyOptions
from dataset.Dataset import GrabNetDataset
from dataset.obman_preprocess import ObManObj
from epochbase import TrainEpoch, ValEpoch
from GraspTTA.network.affordanceNet_obman_mano_vertex import affordanceNet
from GraspTTA.network.cmapnet_objhand import pointnet_reg
from models.cGrasp_vae import cGraspvae
from models.ConditionNet import (ConditionBERT, ConditionMAE,
                                 ConditionMAE_origin, ConditionTrans)
from models.pointnet_encoder import ObjRegionConditionEncoder
from torch.utils import data
from traineval_utils.loss import (ChamferDistanceL2Loss,
                                  PointCloudCompletionLoss, cGraspvaeLoss)
from utils.datasets import get_dataset
from utils.epoch_utils import (EpochVAE_mae, EvalEpochVAE_mae, MetersMonitor,
                               PretrainEpoch, TestTTAEpoch, model_update)
from utils.optim import *
from utils.utils import set_random_seed


def test_tta(cfg=None):
    device = torch.device("cuda")
    affordance_model = affordanceNet()
    cmap_model = pointnet_reg(with_rgb=False)
    
    checkpoint_affordance = torch.load('../tta_checkpoints/model_affordance_best_full.pth', map_location=torch.device('cpu'))['network']
    affordance_model.load_state_dict(checkpoint_affordance)
    affordance_model = affordance_model.to(device)
    
    checkpoint_cmap = torch.load('../tta_checkpoints/model_cmap_best.pth', map_location=torch.device('cpu'))['network']
    cmap_model.load_state_dict(checkpoint_cmap)
    cmap_model = cmap_model.to(device)
    
    cgrasp_loss = cGraspvaeLoss(device, cfg)
    cgrasp_loss.to(device)
    
    testset = get_dataset(cfg, mode=cfg.eval_ds)
    testloader = data.DataLoader(testset, batch_size=32, shuffle=False)
    
    kwargs_cmap = cmap_model if cfg.refine else None
    
    testepoch = TestTTAEpoch(cgrasp_loss, testset, output_dir=cfg.output_dir, mode='test', cfg=cfg)
    
    _, _ = testepoch(testloader, 0, affordance_model, None, None, save_pred=True, cmap_model=kwargs_cmap)
    
    

if __name__ == "__main__":
    import argparse

    import utils.cfgs as cfgsu
    import wandb
    from easydict import EasyDict
    from omegaconf import OmegaConf

    # import psutil
    # p = psutil.Process()
    # p.cpu_affinity(range(16))
    # print(p.cpu_affinity())
    # import pdb; pdb.set_trace()
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfgs', type=str, default='config')
    parser.add_argument('--cfgs_fodler', type=str, default='cgrasp')
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--cuda_id', type=str, default="0")
    parser.add_argument('--machine', type=str, default='41')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--resume', action='store_true')
    
    
    parser.add_argument('--mae', action='store_true')
    parser.add_argument('--comp', action='store_true')
    parser.add_argument('--base', action='store_true')

    parser.add_argument('--chkpt', type=str, default=None)
    
    parser.add_argument('--start_epoch', type=int, default=None)
    parser.add_argument('--eval_ds', type=str, default='test')
    parser.add_argument('--batch_intv', type=str, default=1)
    parser.add_argument('--sample_intv', type=str, default=None)
    parser.add_argument('--run_mode', type=str, default='train')
    parser.add_argument('--pt_model', type=str, default='trans')
    parser.add_argument('--run_check', action='store_true')
    parser.add_argument('--no_save', action='store_true')
    
    # parser.add_argument('--no_loss_edge', action='store_true')
    # parser.add_argument('--no_loss_mesh_rec', action='store_true')
    # parser.add_argument('--no_loss_dist_h', action='store_true')
    # parser.add_argument('--no_loss_dist_o', action='store_true')
    # parser.add_argument('--loss_penetr', action='store_false')
    # parser.add_argument('--loss_mano', action='store_false')
    parser.add_argument('--dloss_type', type=str, default=None)
    parser.add_argument('--eval_iter', type=int, default=10)
    parser.add_argument('--cmae_orig', action='store_true')
    parser.add_argument('--use_pos', action='store_true')
    parser.add_argument('--refine', action='store_true')
    parser.add_argument('--rand', action='store_true')
    parser.add_argument('--tta', type=bool, default=True)
    parser.add_argument('--grabnet', action='store_true')
    parser.add_argument('--grabnet_rnum', type=int, default=2048)
    
    args = parser.parse_args()

    set_random_seed(3407)
    
    # cfg = MyOptions()
    # DONE: 读取配置文件并转化成字典，同时加入args的配置
    
    exp_name = cfgsu.config_exp_name(args)
    paths = cfgsu.config_paths(args.machine, exp_name['exp_name'])
    
    # import pdb; pdb.set_trace()
    
    conf = cfgsu.get_config(args, paths, args.cfgs_fodler)
    conf.update(exp_name)
    conf.update(paths)
    conf.update(args.__dict__) # args的配置也记录下来
    
    cfg = EasyDict()
    # DONE: transform the dict config to easydict
    cfg = cfgsu.merge_new_config(cfg, conf)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.cuda_id
    os.environ['OMP_NUM_THREAD'] = '1'
    torch.set_num_threads(1)
    
    cfgsu.save_experiment_config(cfg)
    

    if cfg.wandb:
        wandb.login()
        wandb.init(project=cfg.project_name,
                name=cfg.exp_name,
                config=conf,
                dir=os.path.join(cfg.output_root, 'wandb')) # omegaconf: resolve=True即可填写自动变量
                # dir: set the absolute path for storing the metadata of each runs
                
    print(f"================ testTTA experiment running! ================") # NOTE: Checkpoint! 提醒一下当前实验的属性
        
    test_tta(cfg)