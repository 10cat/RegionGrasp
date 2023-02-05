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
from models.ConditionNet import ConditionMAE, ConditionTrans, ConditionBERT
from models.cGrasp_vae import cGraspvae
from traineval_utils.loss import ChamferDistanceL2Loss, PointCloudCompletionLoss, cGraspvaeLoss
from utils.optim import *
from utils.datasets import get_dataset
from utils.epoch_utils import EpochVAE_comp, ValEpochVAE_comp, EpochVAE_mae, ValEpochVAE_mae,  MetersMonitor, model_update, PretrainEpoch
    
def cgrasp_comp(cfg=None):
    mode = cfg.run_mode
    model_type = cfg.model.cnet.type
    bs = cfg.batch_size
    
    # load the condition net first
    model = ConditionBERT if model_type == 'bert' else ConditionTrans
    cnet = model(**cfg.model.cnet.kwargs)
    # CHECK: load the checkpoint
    if cfg.model.cnet.chkpt_path:
        checkpoint = torch.load(cfg.model.cnet.chkpt_path)
        cnet.load_state_dict(checkpoint['state_dict'])
        print('checkpoint for cnet loaded!')
        
    model = cGraspvae(cnet,
                      **cfg.model.vae.kwargs, cfg=cfg)
    
    # model.named_parameters()
    # import pdb; pdb.set_trace()
    
    if mode == 'train':
        # DONE: dataset -- obj_points / obj_point_normals / obj_trans / input_pc / hand_verts
        valset = get_dataset(cfg, mode='val')
        trainset = get_dataset(cfg, mode='train')
        
        
        trainloader = data.DataLoader(trainset, batch_size=bs, shuffle=True)
        valloader = data.DataLoader(valset, batch_size=bs, shuffle=False)
        
        # TODO: cnet/vae其他参数设置不同学习率
        optimizer, scheduler = build_optim_sche_grasp(model, part_model={'cnet': model.cnet}, cfg=cfg)
        
        # TODO: loss改写
        device = 'cuda' if cfg.use_cuda else 'cpu'
        model = model.to(device)
        cgrasp_loss = cGraspvaeLoss(device, cfg)
        cgrasp_loss.to(device)
        
        
        trainepoch = EpochVAE_comp(cgrasp_loss, trainset, optimizer, scheduler, output_dir=cfg.output_dir, mode='train', cfg=cfg)
        valepoch = ValEpochVAE_comp(cgrasp_loss, valset, optimizer, scheduler, output_dir=cfg.output_dir, mode='val', cfg=cfg)
        
        for epoch in range(cfg.num_epoch):
            model, _ = trainepoch(trainloader, epoch, model)
            _, stop_flag = valepoch(valloader, epoch, model)
            if stop_flag:
                print("Early stopping occur!")
                break
            
def cgrasp_mae(cfg=None):
    mode = cfg.run_mode
    # model_type = cfg.model.cnet.type
    bs = cfg.batch_size
    
    cnet = ConditionMAE(cfg.model.cnet.kwargs)
    
    
        
    if cfg.model.cnet.chkpt_path:
        checkpoint = torch.load(cfg.model.cnet.chkpt_path)
        # 只载入state_dict = 'MAE_encoder'部分的参数 至 state_dict = 'MAE_encoder'
        # import pdb; pdb.set_trace()
        mae_state_dict = {}
        for key, param in checkpoint['state_dict'].items():
            if key.startswith('MAE_encoder.'):
                key = key.replace('MAE_encoder.', '')
                mae_state_dict[key] = param
        cnet.MAE_encoder.load_state_dict(mae_state_dict)
        print('checkpoint for MAE_encoder in cnet loaded!')
    
    model = cGraspvae(cnet,
                      **cfg.model.vae.kwargs, cfg=cfg)
    
    if cfg.resume:
        assert cfg.model.chkpt is not None, "Checkpoint not configured!"
        checkpoint = torch.load(cfg.model.chkpt)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Resuming the exp from {cfg.model.chkpt} ")
    
    if mode == 'train':
        # DONE: dataset -- obj_points / obj_point_normals / obj_trans / input_pc / hand_verts
        valset = get_dataset(cfg, mode='val')
        trainset = get_dataset(cfg, mode='train')
        
        
        trainloader = data.DataLoader(trainset, batch_size=bs, shuffle=True)
        valloader = data.DataLoader(valset, batch_size=bs, shuffle=False)
        
        # TODO: cnet/vae其他参数设置不同学习率
        optimizer, scheduler = build_optim_sche_grasp(model, part_model={'cnet_mae': model.cnet.MAE_encoder}, cfg=cfg)
        
        # TODO: loss改写
        device = 'cuda' if cfg.use_cuda else 'cpu'
        model = model.to(device)
        cgrasp_loss = cGraspvaeLoss(device, cfg)
        cgrasp_loss.to(device)
        
        trainepoch = EpochVAE_mae(cgrasp_loss, trainset, optimizer, scheduler, output_dir=cfg.output_dir, mode='train', cfg=cfg)
        valepoch = ValEpochVAE_mae(cgrasp_loss, valset, optimizer, scheduler, output_dir=cfg.output_dir, mode='val', cfg=cfg)
        
        for epoch in range(cfg.num_epoch):
            model, _ = trainepoch(trainloader, epoch, model)
            _, stop_flag = valepoch(valloader, epoch, model)
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
    parser.add_argument('--cfgs_fodler', type=str, default='cgrasp')
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--cuda_id', type=str, default="0")
    parser.add_argument('--machine', type=str, default='41')
    parser.add_argument('--wandb', action='store_true')
    
    
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--mae', action='store_true')
    parser.add_argument('--comp', action='store_true')
    parser.add_argument('--grasp', action='store_true')

    parser.add_argument('--checkpoint', type=str, default=None)
    
    parser.add_argument('--start_epoch', type=int, default=None)
    parser.add_argument('--eval_ds', type=str, default=None)
    parser.add_argument('--run_mode', type=str, default='train')
    parser.add_argument('--pt_model', type=str, default='trans')

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
    os.environ['OMP_NUM_THREAD'] = '1'
    torch.set_num_threads(1)
    

    if cfg.wandb:
        wandb.login()
        wandb.init(project=cfg.project_name,
                name=cfg.exp_name,
                config=conf,
                dir=os.path.join(cfg.output_root, 'wandb')) # omegaconf: resolve=True即可填写自动变量
                # dir: set the absolute path for storing the metadata of each runs
                
    print(f"================ {cfg.run_type} experiment running! ================") # NOTE: Checkpoint! 提醒一下当前实验的属性
        
    if cfg.run_type == 'cgrasp':
        if cfg.comp:
            cgrasp_comp(cfg=cfg)
        elif cfg.mae:
            cgrasp_mae(cfg=cfg)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    


