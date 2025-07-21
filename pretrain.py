import os
import sys

from tqdm import tqdm

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
from models.cGrasp_vae import cGraspvae
from models.ConditionNet import ConditionBERT, ConditionTrans
from models.PointMAE import PointMAE, PointMAE_orig, PointMAE_PC
from torch.utils import data
from traineval_utils.loss import (ChamferDistanceL2Loss, MPMLoss,
                                  PointCloudCompletionLoss, cGraspvaeLoss)
from utils.datasets import get_dataset
from utils.epoch_utils import (MetersMonitor, PretrainEpoch, PretrainMAEEpoch,
                               model_update)
from utils.optim import *
from utils.utils import set_random_seed


def obj_comp(cfg=None):
    mode = cfg.run_mode
    model = cfg.model.type
    bs = cfg.batch_size
    
    model = ConditionBERT if model == 'bert' else ConditionTrans
    net = model(**cfg.model.kwargs)
    if mode == 'train':
        trainset = get_dataset(cfg, mode='train')
        valset = get_dataset(cfg, mode='val')
         
        trainloader = data.DataLoader(trainset, batch_size=bs, shuffle=True)
        valloader = data.DataLoader(valset, batch_size=bs, shuffle=False)
        
        optimizer, scheduler = build_optim_sche( net, cfg=cfg)
        chloss = PointCloudCompletionLoss()
        
        net = net.to('cuda')
        chloss = chloss.to('cuda')
        
        trainepoch = PretrainEpoch(chloss, optimizer, scheduler, output_dir=cfg.output_dir, cfg=cfg)
        valepoch = PretrainEpoch(chloss, optimizer, scheduler, output_dir=cfg.output_dir, mode='val', cfg=cfg)
        
        for epoch in range(cfg.num_epoch):
            net, _ = trainepoch(trainloader, epoch, net)
            if epoch % cfg.check_interval:
                _, stop_flag = valepoch(valloader, epoch, net)
            if stop_flag:
                print("Early stopping occur!")
                break
            
def mae(cfg=None):
    mode = cfg.run_mode
    bs = cfg.batch_size
    
    if cfg.cdec:
        net = PointMAE_PC(cfg.model)
    elif cfg.cpred:
        net = PointMAE(cfg.model)
    else:
        net = PointMAE_orig(cfg.model)
    
    if mode == 'train':
        trainset = get_dataset(cfg, mode='train')
        valset = get_dataset(cfg, mode='val')
        
        trainloader = data.DataLoader(trainset, batch_size=bs, shuffle=True)
        valloader = data.DataLoader(valset, batch_size=bs, shuffle=False)
        
        optimizer, scheduler = build_optim_sche(net, cfg=cfg)
        
        loss = MPMLoss()
        
        net = net.to('cuda')
        loss = loss.to('cuda')
        
        trainepoch = PretrainMAEEpoch(loss, output_dir=cfg.output_dir, cfg=cfg)
        valepoch = PretrainMAEEpoch(loss, output_dir=cfg.output_dir, mode='val', cfg=cfg)
        
        stopflag = False
        for epoch in range(cfg.num_epoch):
            net, _ = trainepoch(trainloader, epoch, net, optimizer, scheduler)
            _, stop_flag = valepoch(valloader, epoch, net, optimizer, scheduler)
            if stop_flag:
                print("Early stopping occur!")
                break
    

if __name__ == "__main__":
    import argparse

    import utils.cfgs as cfgsu
    import wandb
    from easydict import EasyDict
    from omegaconf import OmegaConf

    # import pdb; pdb.set_trace()
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfgs', type=str, required=True)
    parser.add_argument('--cfgs_fodler', type=str, default='pretrain')
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--cuda_id', type=str, default="0")
    # parser.add_argument('--no_cuda', action='store_false')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--mae', action='store_true')
    parser.add_argument('--comp', action='store_true')
    
    parser.add_argument('--machine', type=str, default='41')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--cdec', action='store_true', help='whether decode center of masked regions for position embedding or not')
    parser.add_argument('--cpred', action='store_true', help='whether predict center of masked regions or not (but use gt for position embedding)')
    
    
    parser.add_argument('--start_epoch', type=int, default=None)
    parser.add_argument('--eval_ds', type=str, default=None)
    parser.add_argument('--run_mode', type=str, default='train')
    parser.add_argument('--run_check', action='store_true')

    args = parser.parse_args()

    set_random_seed(1024)
    
    exp_name = cfgsu.config_exp_name(args)
    paths = cfgsu.config_paths(args.machine, exp_name['exp_name'])
    
    conf = cfgsu.get_config(args, paths, args.cfgs_fodler)
    conf.update(exp_name)
    conf.update(paths)
    conf.update(args.__dict__) # args的配置也记录下来
    
    cfg = EasyDict()
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
                dir=os.path.join(cfg.output_root, 'wandb')) 
                
    print(f"================ {cfg.run_type} experiment running! ================")
        
    if cfg.mae: 
        mae(cfg = cfg)
    else:
        obj_comp(cfg=cfg)
        

    


