import os
import sys

from tqdm import tqdm
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

from epochbase import TrainEpoch, ValEpoch
import random
from utils.utils import set_random_seed

from dataset.obman_preprocess import ObManObj
from models.ConditionNet import ConditionTrans, ConditionBERT
from models.PointMAE import PointMAE
from models.cGrasp_vae import cGraspvae
from traineval_utils.loss import ChamferDistanceL2Loss, PointCloudCompletionLoss, cGraspvaeLoss, MPMLoss
from utils.optim import *
from utils.datasets import get_dataset
from utils.epoch_utils import MetersMonitor, model_update, PretrainEpoch, PretrainMAEEpoch, EpochVAE, ValEpochVAE
    
    
def obj_comp(cfg=None):
    mode = cfg.run_mode
    model = cfg.model.type
    bs = cfg.batch_size
    
    model = ConditionBERT if model == 'bert' else ConditionTrans
    # net = model(embed_dim=cfg.embed_dim, num_heads=cfg.num_heads, mlp_ratio=cfg.mlp_ratio, glob_feat_dim=cfg.glob_feat_dim, depth={'encoder':cfg.depth, 'decoder':cfg.depth}, knn_layer=cfg.knn_layer_num, fps=True)
    net = model(**cfg.model.kwargs)
    if mode == 'train':
        # TODO: dataset传入cfg参数
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
        # import pdb; pdb.set_trace()
        
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
    
    net = PointMAE(cfg.model)
    
    if mode == 'train':
        trainset = get_dataset(cfg, mode='train')
        valset = get_dataset(cfg, mode='val')
        
        trainloader = data.DataLoader(trainset, batch_size=bs, shuffle=True)
        valloader = data.DataLoader(valset, batch_size=bs, shuffle=False)
        
        optimizer, scheduler = build_optim_sche( net, cfg=cfg)
        
        loss = MPMLoss()
        
        net = net.to('cuda')
        loss = loss.to('cuda')
        
        trainepoch = PretrainMAEEpoch(loss, optimizer, scheduler, output_dir=cfg.output_dir, cfg=cfg)
        valepoch = PretrainMAEEpoch(loss, optimizer, scheduler, output_dir=cfg.output_dir, mode='val', cfg=cfg)
        
        stopflag = False
        for epoch in range(cfg.num_epoch):
            net, _ = trainepoch(trainloader, epoch, net)
            if epoch % cfg.check_interval == 0:
                _, stop_flag = valepoch(valloader, epoch, net)
            if stop_flag:
                print("Early stopping occur!")
                break
    
    
    


if __name__ == "__main__":
    import wandb
    import argparse
    from omegaconf import OmegaConf
    from easydict import EasyDict
    import utils.cfgs as cfgsu
    
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
    
    parser.add_argument('--start_epoch', type=int, default=None)
    parser.add_argument('--eval_ds', type=str, default=None)
    parser.add_argument('--run_mode', type=str, default='train')

    args = parser.parse_args()

    set_random_seed(1024)
    
    # cfg = MyOptions()
    # DONE: 读取配置文件并转化成字典，同时加入args的配置
    conf = cfgsu.get_config(args, args.cfgs_fodler)
    conf.update(cfgsu.config_exp_name(args))
    conf.update(cfgsu.config_paths(args.machine, conf['exp_name']))
    conf.update(args.__dict__) # args的配置也记录下来
    
    cfg = EasyDict()
    # DONE: transform the dict config to easydict
    cfg = cfgsu.merge_new_config(cfg, conf)
    # conf = OmegaConf.structured(cfg)
    # import pdb; pdb.set_trace()
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.cuda_id
    
    

    if cfg.wandb:
        wandb.login()
        wandb.init(project=cfg.project_name,
                name=cfg.exp_name,
                config=conf,
                dir=os.path.join(cfg.output_root, 'wandb')) # omegaconf: resolve=True即可填写自动变量
                # dir: set the absolute path for storing the metadata of each runs
                
    print(f"================ {cfg.run_type} experiment running! ================") # NOTE: Checkpoint! 提醒一下当前实验的属性
        
    if cfg.mae: 
        # TODO: pretrain--传入cfg参数
        mae(cfg = cfg)
    else:
        obj_comp(cfg=cfg)
        

    


